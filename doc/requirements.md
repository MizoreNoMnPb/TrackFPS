# 任务需求与实现对照

## 0. 整体框架

| 需求 | 实现 | 状态 |
|------|------|------|
| 输入视频为 1080p60fps 的 FPS 游戏录屏（分屏：左侧游戏画面 + 右侧小地图） | `map_extractor.py` — VIEWPORT 硬编码边界裁剪 | ✓ |
| 视频文件名格式 `{MapName}_{GameNum}.mp4` 解析地图名和对局编号 | `main.py` — `stem.split("_")` | ✓ |
| 参考地图在 `assets/map/{MapName}_unlabeled.png` | `map_extractor.py` — ORB 特征匹配 + 单应矩阵 | ✓ |
| 分段提取地图视角（地图时隐时现），每次连续出现为一个 View | `map_extractor.py` — 扫描 + 二分精确定界 | ✓ |
| 输出结构 `output/{MapName}/Game{N}/View{M}/` | 内含 `map/`, `game/`, `map_region.png`, `metadata.json`, `trajectory/` | ✓ |

## 1. 游戏 UI 信息检测

> game 视角的正中间会有一个绿色的 Match{Num} 字样，Match 字样正下方为比赛剩余时间。Match 与时间被一个绿色 [] 括起来。

| 需求 | 实现 | 状态 |
|------|------|------|
| 检测绿色 `[]` 括号包围的计时器 | `game_analyzer.py` — `_read_timer()`，绿色像素比例 >0.02 | ✓ |
| OCR 读取比赛剩余时间 | `_ocr_timer_sec()` — Tesseract PSM 11 | ✓ |
| 全局计时器对齐（比赛从 20:00 倒计时） | `_build_global_timer_map()` — RANSAC 离群值剔除 | ✓ |
| 与 `[]` 下端水平平行的 game 视角右端会弹出对局提示（击杀播报） | `killreport.py` — (546,44) 447×18 检测 | ✓ |
| 击杀播报格式：`TeamName \| PlayerName [武器] [爆头/击倒] TeamName \| PlayerName` | `parse_event()` — EasyOCR + 模糊匹配 | △ OCR 噪声大 |
| 救援：同队 `TeamName \| PlayerName [Rescue] TeamName \| PlayerName` | 同上 — 同队判定为 rescue | △ |
| 灭队：`TeamName \| PlayerName Wipeout TeamName`（Wipeout 红色） | 同上 — 检测 "wipeout" 关键词 | △ 未充分测试 |

## 2. 队员状态获取

> game 视角的左侧中间会包含一个全部队伍信息的表格，表格最左侧是队伍对应的颜色与右边的队伍名。队伍名右侧为队员情况：白色上半身人像=存活，折线=击倒，十字=击败，X=淘汰。

| 需求 | 实现 | 状态 |
|------|------|------|
| 6 行队伍表格，颜色条 4px 宽，队名 30px 宽 | `game_analyzer.py` — TABLE_X=8, TABLE_Y=208, ROW_H=23 | ✓ |
| 18px 表头行 | `HEADER_H = 18`，`DATA_START_Y = TABLE_Y + HEADER_H` | ✓ |
| 色条颜色分类 | `_classify_color_peak()` — 色调直方图峰值检测 | ✓ |
| 橙色=黄色统一 | `COLOR_ALIASES = {"orange": "yellow"}` | ✓ |
| 3 个队员图标（7×9px），x=[93, 102, 110] | `PLAYER_X`，`_read_players()` — 模板匹配 4 状态图标 | ✓ |
| 存活→击倒→击败→淘汰，淘汰不可逆 | `VALID_TRANSITIONS` — 7 种合法转移 | ✓ |
| 队员状态变化事件记录为 CSV | `_export_csv()` — `events.csv` 含 frame/video_time/game_time/team/player/from/to | ✓ |
| 队伍排名会变化（表格行顺序改变） | `_update_ranking()` — 排名变更事件记录 | ✓ |
| 队伍信息来自固定配置（非 OCR） | `config/teams.json` — 颜色/队名/选手名列表 | ✓ |

## 3. 核心道具 Mandel Brick

> 橙色图标在计时器右侧，未获取时无文字，获取后显示 `{TeamName} Picked up`，破译中显示 `{TeamName} Decoding...` 加剩余时间。

| 需求 | 实现 | 状态 |
|------|------|------|
| 橙色图标 33×33，距计时器右侧 5px | `_read_mandel()` — MANDEL_ICON_X/Y 常量 | ⚠ 未充分测试 |
| 状态文字区在图标右侧 | `MANDEL_TEXT_X/Y` — 未 OCR | ⚠ |
| 获取道具的队伍行右侧显示橙色图标 | `_read_row_mandel()` — MANDEL_ROW_X=174 | ⚠ |

## 4. 地图点跟踪

> 玩家小点是严格的 14×14 的圆点，用队伍颜色填充。每队 3 人。小点上方 20px（圆心起算）为一个 98×20 的矩形下边沿，矩形水平对齐，半透明队伍颜色，文字为 `TeamName \| PlayerName`。

| 需求 | 实现 | 状态 |
|------|------|------|
| 14×14 彩色圆点检测 | `dot_tracker.py` — 中值背景减法 + HoughCircles + 颜色过滤 | ✓ |
| fg_area > 100 过滤地图噪声 | `detect_dots()` — 前景面积阈值 | ✓ |
| 贪心最近邻匹配（帧间最大 30px） | `track_dots()` — 逐帧匹配 | ✓ |
| OCR 识别圆点上方的名称标签（98×20） | `_identify_players()` — EasyOCR + 模糊匹配 | ✓ |
| 排除法补全未识别轨迹 | 3 条轨迹中 2 条已命名 → 第 3 条为剩余选手 | ✓ |
| 轨迹图输出（labeled 地图底图、队名标签、速度热力图） | `track_team()` — track_INK_BabyB.jpg 等 | ✓ |
| 轨迹统计：速度、转向、停留、距离 | `compute_trajectory_stats()` — stats_INK.json | ✓ |
| 时间戳标注（每 10s） | `draw` 中视频时间标注 | ✓ |
| 停止合并（15px 内视为同一停止） | `draw` 中合并逻辑 | ✓ |
| 被击败玩家在地图上的图标 | `assets/player/defeated_onmap.png` | ⚠ 未使用 |

## 5. 队伍/选手信息

> 视频开头出现「入局信息对比」页面，表格 6 行 4 列，包含队伍颜色、队名、选手 ID 和收益数字。

| 需求 | 实现 | 状态 |
|------|------|------|
| 检测 (192,16) 230×35 区域是否含"入局信息对比" | 未自动化 | ✗ |
| OCR 读取表格（1,66）616×1014 | 手动标注后写入 `config/teams.json` | △ 半自动 |
| 选手 ID（白色文字，可变长，含数字） | 手动标注 | △ |

## 6. 其他

| 需求 | 实现 | 状态 |
|------|------|------|
| Windows 兼容 | `encoding="utf-8"` 所有 `open()` + tesseract 路径提示 | ✓ |
| 无效帧跳过（计时器不可见=加载/背包画面） | 提取阶段检测计时器，不可见则跳过 | ✓ |
| 首次运行一键输出 | `python main.py` — 提取→分析→跟踪全流程 | ✓ |

## 图例

✓ 已实现   △ 部分实现/效果受限   ⚠ 已编码但未充分验证   ✗ 未实现
