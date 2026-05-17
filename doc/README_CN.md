# TrackFPS — FPS 游戏小地图选手轨迹提取与分析

[English](../README.md)

从 FPS 游戏 1080p60fps 录屏中自动提取选手移动轨迹并分析比赛事件。本工程基于 EasyOCR 与 OpenCV-Python。由于设备显存与时间限制，未使用深度学习方法。

> **基础任务结果**：Brakkesh_Game2 View 2 的示例输出见 [`/TaskResult`](../TaskResult/)。
> 为获得最佳检测效果，推荐使用从 Bilibili 下载的高清高码率视频源：[百度网盘](https://pan.baidu.com/s/19zYqO1Wo7dG0KjOlXZJzag?pwd=TOTK) 提取码：TOTK。

## 说明

本工程使用游戏百科中的地图资源与视频中截取的图标作为检测辅助，相关资源见 `./assets/`。

## 整体工程流程

```
视频 → 地图视角检测 → 帧提取 → 地图点跟踪 → 事件检测 → 输出
```

### 1. 地图视角检测 (`map_extractor.py`)

扫描视频中地图视角出现的片段，分别截取游戏画面与地图画面。结果分段输出至 `output/{地图名}/Game{N}/View{M}/`。

### 2. 玩家点位跟踪 (`dot_tracker.py`)

从地图视角中提取轨迹与方向信息。方法：

中值背景减法（使用当前视口对应的地图区域）+ HoughCircles 圆检测 + 颜色过滤。小地图上玩家圆点约 14×14 像素。通过识别圆点上方 20px 处 98×20 的名称标签（EasyOCR + 模糊匹配）关联选手身份。贪心最近邻匹配保证帧间轨迹平滑并过滤误检。

输出至 `output/{地图名}/Game{N}/View{M}/trajectory/`：速度热力图、轨迹图（含转向与停留标注）、选手统计 `stats_{队名}.json`。

### 3. 游戏 UI 分析 (`game_analyzer.py`)

处理游戏画面 UI 与右上角击杀播报，提取对局事件。

游戏 UI：读取左侧队伍表格（6 行 × 3 人）。色条使用色调直方图峰值检测，选手状态使用 7×9 模板匹配（存活/击倒/击败/淘汰）。正中间计时器 OCR（Tesseract PSM 11）对齐全局比赛时间。

击杀播报：检测右上角 (546,44) 447×18 像素区域的击杀信息。在玩家状态变化帧附近做帧差检测，EasyOCR + 编辑距离模糊匹配已知队名与选手名。

## 主要挑战与解决

- **地图噪声**：道路与标签与玩家点相似。中值背景减法 + 前景面积过滤解决。
- **颜色分类**：半透明色条被游戏画面干扰。15 帧时序平滑 + 多数投票 + RANSAC 离群值剔除。
- **选手识别**：98×20px 名称标签 OCR。模糊编辑距离匹配 + 排除法补全未识别轨迹。

## 尚未解决的问题

- **纯视觉匹配有固有局限**。例如仅在白队（INK）轨迹处理中效果较好，当大量选手聚集或地图视口过大时，轨迹不够准确。
- **视频分辨率限制模板匹配**。1080p 分屏视角下，UI 队员状态图标仅 7×9px，击杀信息仅 18px 高——极难 OCR 或模板匹配。当前方案使用 UI + 播报双重验证来弥补。
- **半透明 UI 导致信息丢失**。许多 UI 元素半透明且直接重叠（如地图圆点上方的名称标签），近距离多人交战时无法分辨。因此选择直接提供 `config/teams.json` 中的队伍信息；视觉 OCR 方案作为备选。

## 环境配置

依赖 Python 3.12、conda 与 tesseract OCR。

**Linux (Ubuntu/Debian)：**

```bash
# OCR 系统包
sudo apt install tesseract-ocr tesseract-ocr-chi-sim

# Python 环境
conda create -n trackfps python=3.12 -y
conda activate trackfps
pip install -r requirements.txt
```

**Windows：**

```bash
# 1. 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装 tesseract
#    （安装时勾选 Chinese Simplified）
# 2. 添加到 PATH 或在代码中指定路径：
#    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Python 环境
conda create -n trackfps python=3.12 -y
conda activate trackfps
pip install -r requirements.txt
```

## 使用

```bash
conda activate trackfps

# 首次运行
python main.py

# 后续运行
python main.py --extract-only                     # 仅提取帧
python main.py --track-only                       # 跳过提取，仅分析与跟踪
python main.py -c config/custom.yaml              # 自定义配置
```

## 配置文件说明 (`config/default.yaml`)

```yaml
input:
  video_path: "input/Final/Brakkesh_Game2.mp4"

map_dir: "assets/map"                  # 参考地图目录
teams_config: "config/teams.json"      # 队伍颜色、队名、选手名单

output:
  dir: "output"

pipeline:
  skip_extraction: false               # 已提取过帧时设为 true
  analyze_ui: true                     # 生成 events.csv
  track_teams: "INK"                   # "all"、"INK"、或 ["INK","ESG"]

scanning:
  scan_step: 300                       # 每 N 帧检测一次地图视角
  min_segment_frames: 120              # 有效片段的最低帧数

trajectory:
  turn_threshold: 90                   # 标记转向的角度阈值（度）
  stop_speed: 2                        # 低于此速度视为停留（px/s）
  stop_min_duration: 0.5               # 停留最短持续时间（秒）
```

## 输出结构

```
output/{地图名}/Game{N}/View{M}/
├── map/                    # 地图裁剪帧
├── game/                   # 游戏画面裁剪帧
├── map_region.png          # 参考地图对应区域
├── metadata.json           # 视口坐标、单应矩阵、帧范围
├── game_analysis.json      # 逐帧 UI 状态 + 事件时间线
├── events.csv              # 人类可读事件表
└── trajectory/
    ├── track_{队名}_{选手}.jpg     # 个人轨迹图
    ├── heatmap_{队名}_{选手}.jpg  # 速度热力图
    └── stats_{队名}.json          # 速度、转向、距离统计
```

## events.csv 事件表格式

`events.csv` 各列含义：

| 列 | 说明 |
|----|------|
| `frame` | 提取帧号 |
| `video_time` | 视频时间戳 (M:SS) |
| `game_time` | 游戏倒计时 (M:SS) |
| `type` | `player_status`（选手状态）或 `ranking_change`（排名变化） |

**player_status 行：**

| 列 | 说明 |
|----|------|
| `team` | 队名(颜色) |
| `player` | 选手名（对应 teams.json） |
| `from_status` | 变化前状态：alive / knocked / defeated / eliminated |
| `to_status` | 变化后状态 |

**ranking_change 行：**

| 列 | 说明 |
|----|------|
| `team` | 移动到新排名的队伍 |
| `player` | 排名 #N（从 1 开始） |
| `from_status` | 之前该排名的队伍 |
| `to_status` | 现在该排名的队伍 |

合法选手状态转移：`alive→knocked`、`knocked→alive`、`knocked→defeated`、`defeated→alive`、`*→eliminated`。

## 开发环境

- 系统：Ubuntu 22.04 (WSL2) / Windows 11
- Python：3.12（conda 环境 `trackfps`）
- 编辑器：VS Code + Claude Code 插件
- 主要依赖：opencv-python 4.13、easyocr 1.7、numpy 1.24+、pytesseract 0.3
