# SmartStudio

SmartStudio 是一个开源实时虚拟演播室项目。它支持普通视频、摄像头和绿幕输入，可在浏览器中完成人像抠图、背景替换、多人补全及描边、辉光、阴影、文字等演播室特效。

```text
视频 / 摄像头
  -> 自动场景识别
  -> 普通场景：RVM + 关键帧光流传播
  -> 绿幕场景：专用 Chroma Key + 去绿溢色
  -> Alpha 时序稳定与人物连通修复
  -> 背景、调色和特效合成
  -> Web 实时预览 / 图片与视频导出
```

## 功能

- RVM-MobileNetV3 视频人像抠图，保留循环状态以改善时间连续性。
- 自动识别绿幕，切换到更快的色键、边缘羽化、形态学清理和去绿溢色路径。
- 关键帧推理和低分辨率光流传播；实时模式只保留最新帧，避免延迟累积。
- 可选 YOLO11n-seg 多人先验，用于补救较小或靠后的漏检人物。
- 图片、视频、纯色、模糊和透明背景。
- 描边、辉光、阴影、文字、Logo 以及人物缩放和定位。
- Web 工作台、Qt 桌面界面、图片/视频 CLI 和评测脚本。
- 内置办公室、会议室、舞台和城市背景，Web 端可预览并选择。
- 基于 faster-whisper 的多语言实时字幕；可独立选择视频音轨、具体麦克风或电脑播放设备，支持自动语言检测。

## 系统要求

推荐使用 Windows 10/11 或 Linux、Miniconda/Anaconda，以及支持 CUDA 12.1 的 NVIDIA GPU（建议 6 GB 以上显存）。项目可在 CPU 上运行，但普通场景中的 RVM 通常无法达到实时速度；绿幕专用路径不依赖神经网络，CPU 也能获得较高速度。

## 快速开始

### 1. 获取代码

```bash
git clone <YOUR_REPOSITORY_URL>
cd SmartStudio
```

发布后请把 `<YOUR_REPOSITORY_URL>` 替换为真实仓库地址。

### 2. 创建环境

```powershell
conda env create -f environment.yml
conda activate smartstudio
```

已有同名环境时更新：

```powershell
conda env update -n smartstudio -f environment.yml --prune
conda activate smartstudio
```

检查 GPU：

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果不用 Conda，可在已正确安装 PyTorch 的 Python 3.10 环境中执行：

```bash
python -m pip install -r requirements.txt
```

### 3. 下载 RVM 模型

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail --retry 3 `
  -o models\rvm_mobilenetv3.pth `
  "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
```

Linux/macOS：

```bash
mkdir -p models
curl -L --fail --retry 3 \
  -o models/rvm_mobilenetv3.pth \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
```

最终路径必须为 `models/rvm_mobilenetv3.pth`。Windows 用户也可以执行：

```powershell
.\setup.ps1 -CreateEnv
```

### 4. 下载实时字幕模型

字幕默认使用 `dropbox-dash/faster-whisper-large-v3-turbo`，保存到 `models/faster-whisper-large-v3-turbo`：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('dropbox-dash/faster-whisper-large-v3-turbo', local_dir='models/faster-whisper-large-v3-turbo')"
```

无法访问 Hugging Face 官方节点时，可临时设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python -c "from huggingface_hub import snapshot_download; snapshot_download('dropbox-dash/faster-whisper-large-v3-turbo', local_dir='models/faster-whisper-large-v3-turbo')"
```

模型约 464 MiB。字幕默认关闭，只有在 Web 端启用后才加载模型或访问麦克风。配置为 `device: auto` 时优先使用 NVIDIA GPU 和 FP16，不可用时自动回退到 CPU int8。

## 启动 Web 演播室

使用内置绿幕演示视频：

```powershell
conda activate smartstudio
python web_server.py
```

打开 <http://127.0.0.1:5000>。

使用自己的视频：

```powershell
python web_server.py --source "D:\videos\input.mp4"
```

使用摄像头：

```powershell
python web_server.py --camera 0
```

允许局域网访问：

```powershell
python web_server.py --host 0.0.0.0 --port 5000
```

然后通过本机局域网 IP 访问，例如 `http://192.168.1.10:5000`。不要将 Flask 开发服务器直接暴露到公网。

## Web 操作

1. 在“输入源”中填写服务器本机的视频路径，或选择摄像头。
2. 选择背景类型；图片背景可直接点击场景缩略图。
3. 根据需要启用描边、辉光、阴影和文字。
4. 点击“应用设置”。
5. 状态区会显示运行模式、处理 FPS、延迟、帧号和实时保护状态。
6. 点击“下载截图”保存当前合成画面。
7. 在“实时字幕”中启用识别并选择“自动检测”或指定语言。
8. “音频来源”可选“跟随画面”“视频音轨”“麦克风”或“电脑音频”；后两项可继续选择具体设备。

“跟随画面”保持默认行为：视频输入读取文件音轨，摄像头输入读取系统默认麦克风。选择“电脑音频”时，Windows 使用 WASAPI loopback 直接采集所选耳机或扬声器正在播放的声音，不需要外放后再由麦克风收音。

| 背景类型 | 说明 |
| --- | --- |
| `solid` | 纯色背景 |
| `image` | 静态场景图片 |
| `video` | 循环视频背景 |
| `blur` | 原画面背景模糊 |
| `transparent` | 黑色预览画布，用于 Alpha 工作流 |

场景图片位于 `web/backgrounds/`。加入新的 `.jpg` 或 `.png` 后重启服务，图片会自动出现在图库中。

## 绿幕与普通视频

系统默认自动识别场景：

- `mode: chroma`：检测到稳定绿幕，跳过 RVM，使用专用色键、软边缘和去绿溢色处理。
- `mode: rvm`：普通自然场景，使用 RVM 人像抠图和关键帧传播。

当普通场景中存在大面积绿色物体并发生误判时，可关闭绿幕优化：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/config `
  -ContentType "application/json" `
  -Body '{"green_screen_optimized":false}'
```

## 性能档位

默认实时档：

```powershell
python web_server.py --processing-scale 0.4 --inference-stride 3
```

- `processing-scale` 是算法工作分辨率，范围 `0.25` 到 `1.0`，输出仍保持源视频尺寸。
- `inference-stride` 表示普通场景每隔多少帧运行一次 RVM，中间帧使用光流传播。
- `--person-prior` 启用 YOLO 多人补救，可能改善漏检，但会显著降低速度。

| 用途 | 参数 | 说明 |
| --- | --- | --- |
| 实时预览 | `--processing-scale 0.4 --inference-stride 3` | 优先达到 30 FPS |
| 平衡质量 | `--processing-scale 0.5 --inference-stride 2` | 边缘更稳定，速度下降 |
| 高质量导出 | `--processing-scale 1.0 --inference-stride 1` | 每帧 RVM，不保证实时 |
| 困难多人 | 追加 `--person-prior` | 使用 YOLO11n-seg 补救人物 |

首次进入 RVM 模式需要加载模型，第一帧可能需要数秒；这不代表稳态延迟。绿幕段和非绿幕段的速度也不能直接比较。

## 命令行处理

处理图片：

```powershell
python main.py --image assets/demo/input.png --output output/result.png
```

会生成 `output/result.png` 和 `output/result_alpha.png`。

处理视频：

```powershell
python main.py --video assets/demo/live_studio_test.mp4 --output output/result.mp4
```

启动 Qt 桌面端：

```powershell
python main.py --gui
```

## HTTP API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/status` | FPS、延迟、当前模式和输入源 |
| `GET` | `/api/config` | 当前配置 |
| `GET` | `/api/audio/devices` | 可用麦克风和电脑播放设备 |
| `POST` | `/api/config` | 修改背景、质量、特效和字幕 |
| `POST` | `/api/source` | 切换视频或摄像头 |
| `POST` | `/api/reset` | 重置 RVM 和时序状态 |
| `POST` | `/api/snapshot` | 下载当前合成帧 |
| `GET` | `/video_feed` | MJPEG 实时视频流 |
| `GET` | `/get_frame` | 当前 JPEG 帧 |

查看状态：

```powershell
curl.exe http://127.0.0.1:5000/api/status
```

通过 API 启用自动语言字幕：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/config `
  -ContentType "application/json" `
  -Body '{"subtitles":{"enabled":true,"language":"auto","audio_source":"system","system_audio_device":"default"}}'
```

`audio_source` 支持 `auto`、`video`、`microphone` 和 `system`。设备编号可从 `/api/audio/devices` 获取；使用 `default` 会跟随 Windows 当前默认设备。

`/api/status` 的 `subtitle` 字段会返回 `status`、`text`、`language`、`language_probability`、`latency_ms`、实际音频来源和错误信息。识别线程独立于 `rvm`/`chroma` 画面线程，切换抠图模式不会清空字幕。

切换背景：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/config `
  -ContentType "application/json" `
  -Body '{"background":{"kind":"image","value":"web/backgrounds/meeting_room.jpg"}}'
```

保存截图：

```powershell
curl.exe -X POST `
  http://127.0.0.1:5000/api/snapshot `
  -o output\web_snapshots\acceptance.jpg
```

## 测试与验收

```powershell
python -m pytest -q
python web_server.py
curl.exe http://127.0.0.1:5000/api/status
```

默认实时档建议检查：

- `running` 为 `true`。
- 绿幕段 `mode` 为 `chroma`，普通片段为 `rvm`。
- 目标机器上 `processing_fps >= 30`。
- 正常情况下 `quality_guard` 为 `false`。
- 输出分辨率与输入一致，人物主体没有明显缩放模糊。

本项目曾在 RTX 4060 Laptop GPU 上对内置演示绿幕段测得约 `60.4 FPS / 12.98 ms`。该数字只代表对应机器、视频段和配置，不是对其他硬件的性能保证。

100 张 Aisegment Matting Human 数据集评测：

```powershell
$env:KAGGLE_API_TOKEN="YOUR_KAGGLE_TOKEN"
python download_subset_100.py
python benchmarks/full_evaluation.py
```

数据集不会提交到仓库，结果默认写入 `output/full_evaluation/`。

## 常见问题

### CUDA 显示为 False

确认激活了 `smartstudio` 环境，并检查驱动与 PyTorch：

```powershell
nvidia-smi
conda list pytorch
```

### 找不到 RVM checkpoint

```powershell
Test-Path models\rvm_mobilenetv3.pth
```

路径和文件名必须完全匹配。

### FPS 低或延迟高

- 关闭辉光、阴影、颜色协调和多人先验。
- 使用 `--processing-scale 0.4 --inference-stride 3`。
- 关闭其他占用 GPU 的应用。
- 等待首次模型加载完成后再看稳态 FPS。
- 确认没有多个 `web_server.py` 同时运行。

Windows 查看进程：

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*web_server.py*' } |
  Select-Object ProcessId, CommandLine
```

### 绿幕边缘有绿色光晕

保证绿幕照明均匀，让人物与幕布保持距离，避免绿色反光落在头发和衣服上。算法会执行去绿溢色，但严重的物理反光无法完全从单帧颜色恢复。

### 后方人物遗漏

普通自然视频可追加 `--person-prior`。绿幕视频应优先改善灯光和色键范围，而不是开启 YOLO，因为绿幕路径会按颜色提取全部前景物体。

### GUI 选择图片后出现 `NoneType ... shape`

这表示图片读取失败。请先更新到最新版；GUI 已兼容 Windows 中文路径，并会对无效图片显示明确提示。`--gui` 当前用于打开和处理单张 JPG/PNG 图片，不负责播放视频。

视频处理参数区分大小写，标准写法是小写 `--video`，而且后面必须提供文件路径：

```powershell
python main.py --video "D:\videos\input.mp4" --output "output\result.mp4"
```

不能只执行 `python main.py --VIDEO`。需要实时视频预览时应启动：

```powershell
python web_server.py --source "D:\videos\input.mp4"
```

## 项目结构

```text
SmartStudio/
├── assets/demo/             演示图片和视频
├── benchmarks/              性能、质量和演示生成脚本
├── configs/default.yaml     默认算法与特效配置
├── models/                  RVM 权重目录
├── src/matting/             RVM、绿幕、多人先验和时序模块
├── src/compositing/         背景、合成与颜色协调
├── src/effects/             描边、辉光、阴影、文字和 Logo
├── tests/                   自动测试
├── web/backgrounds/         Web 场景图库
├── web/templates/           Web 前端
├── main.py                  CLI 与 Qt 入口
└── web_server.py            Web 服务入口
```

## 模型、数据与许可

- RVM 来源：[PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)，请遵守其许可证和模型发布条款。
- 可选 `yolo11n-seg.pt` 由 Ultralytics 加载。公开发布或商用前请检查当前 Ultralytics 许可证。
- Aisegment 数据集仅用于本地评测，使用者需要遵守 Kaggle 数据集许可。
- 场景图库来源记录在 [`web/backgrounds/SOURCES.md`](web/backgrounds/SOURCES.md)，重新分发前请核对许可和署名要求。
- 演示视频也应在发布前确认来源和再分发许可。

仓库当前没有根目录 `LICENSE` 文件。正式开源前必须由项目所有者选择并添加许可证；在此之前，代码默认不会自动获得开源使用授权。

## 开源发布检查

- 添加明确的根目录 `LICENSE`。
- 将 `<YOUR_REPOSITORY_URL>` 替换为真实地址。
- 不提交 `datasets/`、`output/`、日志、缓存和任何 API Token。
- 决定是否直接分发模型权重、大型演示视频和第三方背景素材。
- 在全新机器或干净 Conda 环境中完成一次安装、启动和测试。

欢迎通过 Issue 提交可复现的问题。请附上操作系统、GPU、CUDA/PyTorch 版本、启动命令以及 `/api/status` 输出。
### Real-time face accessories

The web preview supports MediaPipe Face Mesh tracking with professional transparent OpenMoji stickers:

- `glasses`: transparent eyeglasses overlay
- `sunglasses`: tinted sunglasses overlay
- `hat`: transparent hat overlay

Assets are stored in `assets/face_effects/` and loaded once at startup. They are distributed under OpenMoji CC BY-SA 4.0; see https://openmoji.org/ for attribution and license terms.

Enable an accessory in the browser under "人脸饰品" and enable "启用跟踪". The accessory follows eye/forehead landmarks, including scale and head rotation. For headwear, the sticker is intentionally allowed to extend outside the person alpha so the brim is not clipped.
