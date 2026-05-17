# TrackFPS

FPS游戏2D俯视视角视频 — 选手轨迹提取与行为分析

## 功能

自动从FPS游戏顶视角录像中：
1. 检测并跟踪目标队伍选手
2. 提取完整移动轨迹
3. 识别关键事件（方向变化、射击、拾取等）
4. 生成带时序标注的可视化路线图

## 环境配置

```bash
conda create -n trackfps python=3.12 -y
conda activate trackfps
pip install -r requirements.txt
```

## 运行

```bash
python main.py                           # 使用默认配置
python main.py -c config/custom.yaml     # 使用自定义配置
```

## 项目结构

```
TrackFPS/
├── main.py                  # 入口
├── config/
│   └── default.yaml         # 默认配置
├── src/
│   ├── preprocessing.py     # 视频抽帧与图像增强
│   ├── detection.py         # YOLO选手检测 + 队伍颜色分类
│   ├── tracking.py          # SORT多目标跟踪 + 卡尔曼滤波
│   ├── events.py            # 事件检测（方向/射击/拾取）
│   ├── mapping.py           # 像素→地图坐标转换
│   ├── visualization.py     # 轨迹图与速度热力图
│   └── pipeline.py          # 管道编排
├── input/                   # 输入视频
├── output/                  # 所有输出结果
├── notebooks/               # 实验性Notebook
└── requirements.txt
```

## 输出

| 文件 | 位置 | 格式 |
|---|---|---|
| 轨迹数据 | `output/trajectories/` | CSV + GeoJSON |
| 事件时间表 | `output/events/` | CSV + JSON |
| 轨迹可视化图 | `output/visualizations/` | PNG |
| 速度热力图 | `output/visualizations/` | PNG（可选） |

## 配置要点

编辑 `config/default.yaml`：

- `input.video_path` — 视频路径
- `target.team_color` — 目标队伍颜色（white/red/blue/green等）
- `detection.model_path` — YOLO模型路径（先用预训练，建议用俯视视角数据微调）
- `detection.conf_threshold` — 检测置信度阈值（小目标调低至0.25-0.35）
- `tracking.max_age` — 跟踪丢失后保留帧数（遮挡多时调大）
- `events` — 各事件检测的阈值参数
