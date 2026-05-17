# TrackFPS — FPS 游戏小地图选手轨迹提取与分析

从 FPS 游戏 1080p60fps 录屏中自动提取选手移动轨迹并分析比赛事件。

## 管线

```
视频 → 地图视角检测 → 帧提取 → 地图点跟踪 → 事件检测 → 输出
```

### 1. 地图视角检测 (`map_extractor.py`)

扫描视频寻找右侧小地图覆盖层。地图左边框在 x=1008 处，裁剪得到地图区（854×852）和游戏画面区（998×559）。ORB 特征匹配 + 单应矩阵对齐参考地图。跳过无计时器的帧（加载/背包画面）。

### 2. 玩家点位跟踪 (`dot_tracker.py`)

中值背景减法（地图静止不滚动）+ HoughCircles 圆检测 + 颜色过滤。圆点约 14×14 像素，贪心最近邻匹配（帧间最大位移 30px）保证轨迹平滑。通过识别圆点上方 20px 处的名称标签（OCR + 模糊匹配）将轨迹关联到具体选手。

### 3. 游戏 UI 分析 (`game_analyzer.py`)

读取左侧队伍表格（6 行 × 3 人），色条用色调直方图峰值检测，选手状态用 7×9 模板匹配（存活/击倒/击败/淘汰）。计时器 OCR（Tesseract PSM 11）对齐全局比赛时间。

### 4. 击杀播报 (`killreport.py`)

检测右上角 (546,44) 447×18 的右对齐击杀信息。在玩家状态变化帧附近做帧差检测，EasyOCR + 编辑距离模糊匹配已知队名/选手名。

## 主要挑战与解决

- **地图噪声**：道路、标签与玩家点相似。中值背景减法 + 前景面积过滤（>100px）解决。
- **颜色分类**：半透明色条被游戏画面干扰。15 帧时序平滑 + 多数投票 + RANSAC 离群值剔除。
- **选手识别**：98×20px 名称标签 OCR。模糊编辑距离匹配 + 排除法补全未识别轨迹。
- **计时器 OCR 漂移**：PSM 6 将 19 误读为 13 导致 6 分钟偏差。改用 PSM 11 + 全 View 全局 RANSAC 拟合。

## 使用

```bash
conda activate trackfps
python main.py                                    # 全流程
python main.py --extract-only                     # 仅提取帧
python main.py --track-only                       # 跳过提取，仅分析
python main.py -v input/Final/Dum_Game1.mp4       # 指定视频
python main.py -c config/custom.yaml              # 自定义配置
```

## 输出结构

```
output/{地图名}/Game{N}/View{M}/
├── map/                    # 地图裁剪帧 (854×852)
├── game/                   # 游戏画面裁剪帧 (998×559)
├── map_region.png          # 参考地图对应区域
├── metadata.json           # 视口坐标、单应矩阵、帧范围
├── game_analysis.json      # 逐帧 UI 状态 + 事件时间线
├── events.csv              # 人类可读事件表
└── trajectory/
    ├── track_{队名}_{选手}.jpg     # 个人轨迹图
    ├── heatmap_{队名}_{选手}.jpg  # 速度热力图
    └── stats_{队名}.json          # 速度、转向、距离统计
```

## 配置

`config/teams.json` — 队伍颜色、队名、选手名单（3 人/队 × 6 队）。
`config/default.yaml` — 视频路径、管线开关、跟踪参数。
