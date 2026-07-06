"""Extended GLB export for TRELLIS.2.

Based on o_voxel.postprocess.to_glb, which only bakes base color and
metallic/roughness. This version additionally bakes:
  - a tangent-space normal map from the original high-res mesh (the detail
    lost by decimation/remeshing), exported as glTF normalTexture;
  - an ambient occlusion map via hemisphere ray casting against the high-res
    mesh (cuBVH.ray_trace), packed into the R channel of the ORM texture and
    exported as glTF occlusionTexture.

Runs inside the isolated worker venv (needs cumesh / nvdiffrast / flex_gemm).
"""

from typing import Dict, Union
import math

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh

_RAY_CHUNK = 4_000_000


def _orthonormal_frame(n: torch.Tensor, t_raw: torch.Tensor, b_raw: torch.Tensor):
    """Gram-Schmidt orthonormalization of a per-texel (T, B) pair against N,
    with fallbacks for degenerate UV triangles."""
    up = torch.where(
        n[:, 2:3].abs() < 0.99,
        torch.tensor([0.0, 0.0, 1.0], device=n.device).expand_as(n),
        torch.tensor([1.0, 0.0, 0.0], device=n.device).expand_as(n),
    )
    t = t_raw - n * (t_raw * n).sum(-1, keepdim=True)
    t_norm = t.norm(dim=-1, keepdim=True)
    t_fallback = F.normalize(torch.cross(up, n, dim=-1), dim=-1)
    t = torch.where(t_norm > 1e-6, t / t_norm.clamp_min(1e-12), t_fallback)

    b = b_raw - n * (b_raw * n).sum(-1, keepdim=True) - t * (b_raw * t).sum(-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True)
    b_fallback = torch.cross(n, t, dim=-1)
    b = torch.where(b_norm > 1e-6, b / b_norm.clamp_min(1e-12), b_fallback)
    return t, b


def _compute_smooth_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted smooth vertex normals."""
    faces_l = faces.long()
    tri = vertices[faces_l]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    vn = torch.zeros_like(vertices)
    vn.index_add_(0, faces_l.reshape(-1), fn.repeat_interleave(3, dim=0))
    return F.normalize(vn, dim=-1)


def _bake_ao(
    origins: torch.Tensor,
    normals: torch.Tensor,
    bvh,
    samples: int,
    max_distance: float,
    seed: int = 42,
) -> torch.Tensor:
    """Cosine-weighted hemisphere visibility, 1 = fully open, 0 = fully occluded."""
    n_pts = origins.shape[0]
    device = origins.device
    up = torch.where(
        normals[:, 2:3].abs() < 0.99,
        torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(normals),
        torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(normals),
    )
    t1 = F.normalize(torch.cross(up, normals, dim=-1), dim=-1)
    t2 = torch.cross(normals, t1, dim=-1)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    visible = torch.zeros(n_pts, device=device)
    for _ in range(samples):
        r1 = torch.rand(n_pts, device=device, generator=gen)
        r2 = torch.rand(n_pts, device=device, generator=gen)
        phi = 2.0 * math.pi * r1
        sin_theta = torch.sqrt(r2)
        dirs = (
            t1 * (torch.cos(phi) * sin_theta).unsqueeze(-1)
            + t2 * (torch.sin(phi) * sin_theta).unsqueeze(-1)
            + normals * torch.sqrt((1.0 - r2).clamp_min(0.0)).unsqueeze(-1)
        )
        for i in range(0, n_pts, _RAY_CHUNK):
            _, _, depth = bvh.ray_trace(origins[i:i + _RAY_CHUNK], dirs[i:i + _RAY_CHUNK])
            # rays that miss return a huge depth -> counted as visible
            visible[i:i + _RAY_CHUNK] += (depth > max_distance).float()
    return visible / samples


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    bake_normal_map: bool = True,
    bake_ao: bool = True,
    ao_samples: int = 32,
    ao_max_distance: float = 0.25,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Same pipeline as o_voxel.postprocess.to_glb plus normal map / AO baking."""
    # --- Input normalization (AABB, voxel size, grid size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor) and aabb.shape == (2, 3)

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    vertices = vertices.cuda()
    faces = faces.cuda()

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()

    # BVH over the high-res mesh: used to re-project texels, bake normals and AO
    bvh = cumesh.cuBVH(vertices, faces)
    hi_vertex_normals = _compute_smooth_normals(vertices, faces) if (bake_normal_map or bake_ao) else None

    # --- Simplification / remeshing (identical to o_voxel) ---
    if not remesh:
        mesh.simplify(decimation_target * 3, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center=center,
            scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution,
            band=remesh_band,
            project_back=remesh_project,
            verbose=verbose,
            bvh=bvh,
        ))
        mesh.simplify(decimation_target, verbose=verbose)

    # --- UV parameterization ---
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals().cuda()[out_vmaps]

    # --- Rasterize in UV space ---
    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([
        out_uvs * 2 - 1,
        torch.zeros_like(out_uvs[:, :1]),
        torch.ones_like(out_uvs[:, :1]),
    ], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i + 100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)

    mask_t = rast[0, ..., 3] > 0

    # Map every valid texel back to the high-res surface
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask_t]
    _, hit_face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    hit_face_id = hit_face_id.long()
    orig_tri_verts = vertices[faces[hit_face_id]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    # --- Sample PBR attributes from the sparse voxel volume ---
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask_t] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )

    # --- High-res shading normal per texel (shared by normal map and AO) ---
    hi_normals = None
    n_lo_valid = None
    if bake_normal_map or bake_ao:
        hi_normals = F.normalize(
            (hi_vertex_normals[faces[hit_face_id]] * uvw.unsqueeze(-1)).sum(dim=1), dim=-1
        )
        n_lo = dr.interpolate(out_normals.contiguous().unsqueeze(0), rast, out_faces)[0][0]
        n_lo_valid = F.normalize(n_lo[mask_t], dim=-1)
        # Guard against inconsistent winding in the raw extracted mesh
        sign = torch.sign((hi_normals * n_lo_valid).sum(-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        hi_normals = hi_normals * sign

    # --- Bake tangent-space normal map ---
    normal_map_np = None
    if bake_normal_map:
        # Tangents from the final UV orientation (V flipped at export) so that
        # viewers deriving tangents from UVs decode the map consistently.
        uv_flip = out_uvs.clone()
        uv_flip[:, 1] = 1 - uv_flip[:, 1]
        faces_l = out_faces.long()
        fp = out_vertices[faces_l]
        fuv = uv_flip[faces_l]
        e1 = fp[:, 1] - fp[:, 0]
        e2 = fp[:, 2] - fp[:, 0]
        duv1 = fuv[:, 1] - fuv[:, 0]
        duv2 = fuv[:, 2] - fuv[:, 0]
        det = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        inv_det = torch.where(det.abs() > 1e-12, 1.0 / det, torch.zeros_like(det)).unsqueeze(-1)
        face_tangent = (e1 * duv2[:, 1:2] - e2 * duv1[:, 1:2]) * inv_det
        face_bitangent = (e2 * duv1[:, 0:1] - e1 * duv2[:, 0:1]) * inv_det

        tex_face_id = (rast[0, ..., 3][mask_t].long() - 1)
        t_tex, b_tex = _orthonormal_frame(
            n_lo_valid, face_tangent[tex_face_id], face_bitangent[tex_face_id]
        )
        nm = torch.stack([
            (hi_normals * t_tex).sum(-1),
            (hi_normals * b_tex).sum(-1),
            (hi_normals * n_lo_valid).sum(-1),
        ], dim=-1)
        nm = F.normalize(nm, dim=-1)
        normal_map_np = np.full((texture_size, texture_size, 3), (128, 128, 255), dtype=np.uint8)
        normal_map_np[mask_t.cpu().numpy()] = np.clip(
            (nm.cpu().numpy() * 0.5 + 0.5) * 255, 0, 255
        ).astype(np.uint8)

    # --- Bake ambient occlusion ---
    ao_np = None
    if bake_ao:
        eps = float(voxel_size.min().item()) * 0.5
        ao_valid = _bake_ao(
            valid_pos + hi_normals * eps,
            hi_normals,
            bvh,
            samples=ao_samples,
            max_distance=ao_max_distance,
        )
        ao_np = np.full((texture_size, texture_size, 1), 255, dtype=np.uint8)
        ao_np[mask_t.cpu().numpy(), 0] = np.clip(
            ao_valid.cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)

    # --- Texture post-processing & material construction ---
    mask = mask_t.cpu().numpy()
    mask_inv = (~mask).astype(np.uint8)

    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    normal_texture = None
    if normal_map_np is not None:
        normal_map_np = cv2.inpaint(normal_map_np, mask_inv, 3, cv2.INPAINT_TELEA)
        normal_texture = Image.fromarray(normal_map_np)

    if ao_np is not None:
        occlusion = cv2.inpaint(ao_np, mask_inv, 3, cv2.INPAINT_TELEA)[..., None]
    else:
        occlusion = np.full_like(metallic, 255)

    # glTF ORM packing: R = occlusion, G = roughness, B = metallic
    orm_texture = Image.fromarray(np.concatenate([occlusion, roughness, metallic], axis=-1))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=orm_texture,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        occlusionTexture=orm_texture if ao_np is not None else None,
        normalTexture=normal_texture,
        alphaMode='OPAQUE',
        doubleSided=True if not remesh else False,
    )

    # --- Coordinate system conversion & final object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()

    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1]

    return trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )
