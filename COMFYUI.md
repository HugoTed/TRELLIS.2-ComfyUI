# TRELLIS.2 ComfyUI Plugin

ComfyUI 插件：单图生成带 PBR 材质的 3D 模型（GLB），架构参考 [SkinTokens-ComfyUI](https://github.com/HugoTed/SkinTokens-ComfyUI)。

## 安装（WSL2 四步）

### 1. 克隆到 ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/HugoTed/TRELLIS.2-ComfyUI.git
```

> 必须 `--recursive`，否则 `o-voxel` 子模块缺失。

### 2. 一次性系统依赖（需要 sudo，Setup 节点无法代劳）

```bash
cd TRELLIS.2-ComfyUI
sudo bash scripts/setup_wsl_prereqs.sh
```

自动完成：build-essential / python3-venv、gcc-13（新 Ubuntu 默认 GCC 过新）、
CUDA toolkit（nvcc + dev 头文件，缺失时自动添加 NVIDIA WSL 源）、
glibc ≥ 2.38 的 CUDA 头文件补丁。幂等，可重复运行。

### 3. 启动 ComfyUI（WSL 建议加参数）

```bash
python main.py --disable-pinned-memory
```

> WSL 上 ComfyUI 默认的 pinned memory 会锁定大量系统内存，挤压 Worker 的 GPU 分配额度。

### 4. 首次使用

在 ComfyUI 中任选其一：

- 运行一次 **TRELLIS.2 Setup (Install Worker)** 节点，或
- 直接运行 **TRELLIS.2 Image to 3D**，首次会自动安装 Worker

插件会在插件目录下创建独立虚拟环境 `.trellis2-venv`，自动安装固定版本 PyTorch（2.6.0+cu124）、
编译全部 CUDA 扩展（nvdiffrast / CuMesh / FlexGEMM / o-voxel 等）并验证导入，然后启动 Worker 子进程。
**无需手动配置 Python 路径。** 首次安装需编译扩展，约 10–30 分钟。

### WSL 内存配比（重要）

WSL 里每笔 GPU 显存分配都需要 Windows 宿主侧等量的提交内存做后备，配比不当会出现
"显存大量空闲却 CUDA OOM"：

- `.wslconfig` 的 `memory` 建议设为物理内存的一半（64GB 机器设 `memory=32GB`），**不要贪多**
- Windows 页面文件设大一些（如初始 32GB / 最大 64GB），改完重启 Windows

### 可选：flash-attn 加速

Worker 默认自动选择注意力后端（flash_attn → xformers → sdpa，均可工作）。
手动安装 flash-attn 时**必须用匹配 torch 2.6 的预编译 wheel 并加 `--no-deps`**，
否则 pip 会升级 torch、破坏已编译的 CUDA 扩展：

```bash
.trellis2-venv/bin/pip install --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## 工作流节点（category: `3d/trellis2`）

| 节点 | 作用 |
|------|------|
| **TRELLIS.2 Setup (Install Worker)** | 安装 Worker venv、CUDA 扩展、启动 Worker |
| **TRELLIS.2 Load Model** | 预加载模型到 GPU（可选） |
| **TRELLIS.2 Image to 3D** | 单图 → GLB，输出 `model_file` + `FILE_3D_GLB` |
| **TRELLIS.2 Multi-Image to 3D** | 多视角图（2–8 张） → GLB，融合多图条件提升背面/几何精度 |

### 多图（多视角）节点

`TRELLIS.2 Multi-Image to 3D` 使用官方 tuning-free 多图算法（同 TRELLIS `run_multi_image`）：

- **输入方式**：`images` 接一个 IMAGE batch（同尺寸图可用 Image Batch 节点合批），
  不同尺寸的图用可选的 `image_2` / `image_3` / `image_4` 口各接一张
- **multi_image_mode**：
  - `stochastic`：每步轮换一张图作条件，速度快、省显存
  - `multidiffusion`（默认）：每步对所有图的预测取平均，质量更高、更慢
- 所有视角应为**同一物体**，最好为透明背景或可自动抠图的干净图片；
  视角间姿态不一致时结果可能变形

### PBR 贴图导出（法线 / AO / 金属度粗糙度）

两个生成节点均输出完整 PBR 材质的 GLB：

- **baseColorTexture**：基础色（含 alpha 通道，默认 OPAQUE，不激活透明）
- **metallicRoughnessTexture / occlusionTexture**：ORM 打包贴图（R=AO, G=Roughness, B=Metallic）
- **normalTexture**：切线空间法线贴图，从简化前的高模烘焙，保留 decimation 丢失的细节

相关选项：

- `bake_normal_map`（默认开）：烘焙法线贴图
- `bake_ao`（默认开）：半球射线投射烘焙环境光遮蔽，写入 ORM 的 R 通道
- `ao_samples`（默认 32）：AO 每纹素采样数，越大越平滑越慢
- `texture_format`（默认 `png`）：`webp` 体积小但走 `EXT_texture_webp` 扩展，
  Windows 3D 查看器、旧版 Blender 等许多软件不支持，会表现为**下载后贴图全丢**；
  默认 `png` 兼容一切 glTF 查看器

> 注意：官方 demo 预览器里的 "Normal" 等模式是**实时渲染的显示模式**,官方 `to_glb`
> 导出的 GLB 本身并不含法线/AO 贴图——本插件的烘焙是在官方流程之上额外实现的。

### Preview3D 连接

```
Load Image → TRELLIS.2 Image to 3D → Preview 3D（内置）
                      ↓
            model_file 或 model_3d (FILE_3D_GLB)
```

新版 ComfyUI 可直接用 `model_3d` 输出接 **Preview 3D**；旧版用 `model_file`（STRING 路径）。

## 可选配置

```bash
cp config.json.example config.json
```

可自定义端口、venv 路径、PyTorch 索引 URL 等。也可用环境变量 `TRELLIS2_PYTHON` 指定 Worker 解释器。

## 手动安装 Worker（无 ComfyUI）

```bash
# Linux
bash scripts/install_worker.sh

# Windows
powershell -ExecutionPolicy Bypass -File scripts/install_worker.ps1

# 或
python trellis2_bootstrap.py
```

Linux 前置依赖：

```bash
sudo apt install python3.10 python3.10-venv git
# CUDA 扩展编译可能还需要 CUDA Toolkit 12.4
```

## 架构

```
ComfyUI 主进程（任意 Python 版本，仅 requests）
  ├── nodes/              ← 轻量节点
  ├── trellis2_client/        ← HTTP 客户端，自动拉起 Worker
  └── trellis2_bootstrap.py    ← 创建 .trellis2-venv

.trellis2-venv（独立 Python 3.10）
  └── worker/server.py    ← 常驻 HTTP 服务
        └── trellis2 + o-voxel 推理
```

## 要求

- NVIDIA GPU，≥ 24GB 显存
- **Linux 推荐**（官方仅测试 Linux；CUDA 扩展在 Windows 上可能编译失败）
- 首次推理会从 Hugging Face 下载 `microsoft/TRELLIS.2-4B` 权重

## 故障排除

- **节点列表里找不到 TRELLIS.2 节点 / IMPORT FAILED**：若同时安装了 SkinTokens 等 Worker 插件，**禁止**使用通用模块名（`nodes`、`client`、`bootstrap`、`config`）。本插件已全部改为 `trellis2_*` 前缀。请 `git pull` 最新代码并重启 ComfyUI，在节点菜单搜索 `TRELLIS`，分类 **3d → trellis2**。
- **Worker 启动失败**：查看插件目录下 `trellis2-worker.log`
- **flash-attn 安装失败**：在 Worker 环境中 `pip install xformers`，并设置 `ATTN_BACKEND=xformers`
- **ComfyUI 禁止 subprocess**：启动时加 `--allow-subprocess`
- **CUDA OOM 但显存明明有大量空闲（WSL 特有）**：WSL 的 GPU 分配需要 Windows 宿主提交内存做后备。按顺序排查：① `.wslconfig` 的 `memory` 不要超过物理内存一半；② 加大 Windows 页面文件并重启；③ ComfyUI 加 `--disable-pinned-memory`；④ 关闭 Windows 侧占 GPU/内存的程序（Unity、浏览器等）。内核证据：`dmesg | grep dxg` 出现 `create_allocation failed`。
- **真·显存不足**：节点用 `resolution=512`、`max_num_tokens=8192`、`texture_size=1024`、`remesh=off`。Worker 健康检查（`curl http://127.0.0.1:18188/health`）会返回 `vram_free_gb`。
- **装完 flash-attn/xformers 后扩展报 `undefined symbol`**：pip 把 torch 换了版本。恢复：`.trellis2-venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`，再用 `--no-deps` 装匹配 wheel（见上文）。

## 与 SkinTokens-ComfyUI 的对应关系

| SkinTokens | TRELLIS.2 |
|------------|-----------|
| `.tokenrig-venv` | `.trellis2-venv` |
| `bootstrap.py` | `bootstrap.py` |
| `TokenRig Setup` 节点 | `TRELLIS.2 Setup` 节点 |
| 首次 Generate 自动安装 | 首次 Image to 3D 自动安装 |
| `FILE_3D_GLB` 输出 | `FILE_3D_GLB` 输出 |
