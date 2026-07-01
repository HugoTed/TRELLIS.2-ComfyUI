# TRELLIS.2 ComfyUI Plugin

ComfyUI 插件：单图生成带 PBR 材质的 3D 模型（GLB），架构参考 [SkinTokens-ComfyUI](https://github.com/HugoTed/SkinTokens-ComfyUI)。

## 安装（三步）

### 1. 克隆到 ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/your/TRELLIS.2-ComfyUI.git
```

> 必须 `--recursive`，否则 `o-voxel` 子模块缺失。

### 2. 重启 ComfyUI

ComfyUI 主环境只需 `requests`（通常 ComfyUI-Manager 会自动装 `requirements-comfyui.txt`）。

### 3. 首次使用

在 ComfyUI 中任选其一：

- 运行一次 **TRELLIS.2 Setup (Install Worker)** 节点，或
- 直接运行 **TRELLIS.2 Image to 3D**，首次会自动安装 Worker

插件会在插件目录下创建独立虚拟环境 `.trellis2-venv`（Python 3.10+），自动安装 PyTorch、CUDA 扩展和 TRELLIS.2 依赖，并启动 Worker 子进程。**无需手动配置 Python 路径。**

## 工作流节点（category: `3d/trellis2`）

| 节点 | 作用 |
|------|------|
| **TRELLIS.2 Setup (Install Worker)** | 安装 Worker venv、CUDA 扩展、启动 Worker |
| **TRELLIS.2 Load Model** | 预加载模型到 GPU（可选） |
| **TRELLIS.2 Image to 3D** | 单图 → GLB，输出 `model_file` + `FILE_3D_GLB` |

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
python bootstrap.py
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
  ├── client/             ← HTTP 客户端，自动拉起 Worker
  └── bootstrap.py        ← 创建 .trellis2-venv

.trellis2-venv（独立 Python 3.10）
  └── worker/server.py    ← 常驻 HTTP 服务
        └── trellis2 + o-voxel 推理
```

## 要求

- NVIDIA GPU，≥ 24GB 显存
- **Linux 推荐**（官方仅测试 Linux；CUDA 扩展在 Windows 上可能编译失败）
- 首次推理会从 Hugging Face 下载 `microsoft/TRELLIS.2-4B` 权重

## 故障排除

- **节点列表里找不到 TRELLIS.2 节点**：确保已 `git pull` 最新代码（旧版 `from nodes import` 会与 ComfyUI 内置 `nodes.py` 冲突，导致静默注册失败）。重启后在节点菜单搜索 `TRELLIS`，分类为 **3d → trellis2**。
- **Worker 启动失败**：查看插件目录下 `trellis2-worker.log`
- **flash-attn 安装失败**：在 Worker 环境中 `pip install xformers`，并设置 `ATTN_BACKEND=xformers`
- **ComfyUI 禁止 subprocess**：启动时加 `--allow-subprocess`

## 与 SkinTokens-ComfyUI 的对应关系

| SkinTokens | TRELLIS.2 |
|------------|-----------|
| `.tokenrig-venv` | `.trellis2-venv` |
| `bootstrap.py` | `bootstrap.py` |
| `TokenRig Setup` 节点 | `TRELLIS.2 Setup` 节点 |
| 首次 Generate 自动安装 | 首次 Image to 3D 自动安装 |
| `FILE_3D_GLB` 输出 | `FILE_3D_GLB` 输出 |
