# 源码实现流程说明

## `main.py` — 入口

```
main()
 ├─ 读取 config/default.yaml
 ├─ 解析视频文件名 → MapName, GameNum
 ├─ 加载 config/teams.json
 ├─ Step 1: extract_frames()          [map_extractor.py]
 ├─ Step 2: GameAnalyzer.analyze_view()[game_analyzer.py]
 └─ Step 3: track_team()              [dot_tracker.py]
```

可选参数：`--extract-only` / `--track-only` / `--config` / `--video`。
流水线开关在 `pipeline:` 配置节：`skip_extraction`, `analyze_ui`, `track_teams`。

---

## `map_extractor.py` — 地图视角检测与帧提取

```
run(video_path)
 ├─ MapExtractor.__init__()
 │   ├─ 解析文件名 → map_name, game_num
 │   ├─ 加载参考地图 assets/map/{MapName}_unlabeled.png
 │   └─ ORB 预计算参考地图特征点
 │
 └─ scan(scan_step=300)
     ├─ 粗扫描：每 300 帧检测一次地图是否可见
     │   └─ _detect_viewport(frame)
     │       ├─ 检测地图左边界 x=1008 处的亮度跳变 (>30)
     │       └─ 检测地图区域的 Canny 边缘密度 (>10)
     │
     ├─ _group_segments() → 连续 map-positive 帧合并为 View 段
     ├─ _refine_all_boundaries() → 二分搜索精确定界 (±1.2帧)
     └─ 输出 segment: {start_frame, end_frame, viewport_rects}
         └─ VIEWPORT 硬编码: map=(1008,112,854,852), game=(40,258,998,559)

match_and_extract(segment)
 ├─ 取 segment 中间帧 → 裁剪 map viewport → ORB 匹配参考地图 → 单应矩阵 H
 ├─ 从参考地图裁剪对应区域 → map_region.png
 ├─ _extract_frames() → 遍历帧范围 [start, end] step=skip
 │   ├─ 计时器检测：game view (454,8) 92×36 绿色像素>0.02 → 不可见则跳过
 │   ├─ 裁剪 map 区域 → map/frame_{fn}.jpg
 │   └─ 裁剪 game 区域 → game/frame_{fn}.jpg
 └─ 输出 metadata.json (viewport, homography, frame_range, duration)
```

**关键常量**：`VIEWPORT` 硬编码边界，`map_view` 左边界 x=1008 处亮度梯度 >30 作为检测信号。

---

## `game_analyzer.py` — 游戏 UI 分析

```
GameAnalyzer(teams_config="config/teams.json")
 ├─ 加载 4 个玩家状态模板(7×8→7×9): alive/knocked/defeated/eliminated
 └─ 可选: 加载 teams.json (跳过 OCR)

analyze_view(view_dir)
 ├─ _find_warmup_end() → 扫描前 800 帧找计时器首次出现 → 跳过加载过渡
 ├─ _build_color_name_map() → 从 teams.json 载入颜色→队名映射
 │
 ├─ 每帧处理循环:
 │   ├─ _read_timer() → 计时器 ROI (454,8) 92×36 → 绿色像素比例
 │   │   └─ 不可见 → 跳过 (背包/加载画面)
 │   │
 │   ├─ 队伍检测:
 │   │   ├─ 对 6 行 (DATA_START_Y + i*25):
 │   │   │   ├─ _classify_color_peak() → 色条区域 (16,4px) 色调直方图峰值
 │   │   │   │   ├─ 先检查 white (S<40, V>180)
 │   │   │   │   └─ 饱和像素 >50 阈值 → 色调直方图 → 最近 HUE_CENTER
 │   │   │   ├─ _read_players() → 3 个 7×9 窗口 → 模板匹配 (TM_CCOEFF_NORMED)
 │   │   │   └─ _read_row_mandel() → 橙色图标检测 x=174
 │   │   └─ 时序平滑: 15 帧窗口多数投票
 │   │
 │   ├─ 排名变化: _update_ranking()
 │   │   └─ 当前排名与锁定排名不同 → 持续 5 帧确认 → 发出 ranking_change 事件
 │   │
 │   └─ 玩家状态变化: 与上一帧 cur_players 比较 → 发出 player_status 事件
 │
 ├─ _build_global_timer_map() → 所有 View 的 25%/50%/75% 帧 OCR → RANSAC 拟合
 │   └─ game_time = slope × video_fn + intercept
 │
 └─ _export_csv() → events.csv
     ├─ VALID_TRANSITIONS 过滤非法状态转移
     ├─ player_status 行: frame, time, team, player, from→to
     └─ ranking_change 行: frame, time, team → Rank #N
```

**状态转移**：`alive→knocked`, `knocked↔alive`, `knocked→defeated`, `defeated→alive`, `*→eliminated`。

**颜色别名**：`orange → yellow`（同一队伍）。

---

## `dot_tracker.py` — 地图玩家点跟踪

```
track_team(map_dir, team_color, players, team_name)
 ├─ build_background() → 每 10 帧采样 → 中值合成干净地图背景
 │
 ├─ track_dots() → 逐帧检测 + 最近邻匹配
 │   ├─ detect_dots(frame, bg, color_lower, color_upper)
 │   │   ├─ 前景 = |当前帧 - 中值背景| > 20 → 阈值二值化 → 形态学开运算
 │   │   ├─ 颜色过滤: HSV inRange(lower, upper)
 │   │   ├─ HoughCircles: param1=40, param2=10, r=5-9
 │   │   ├─ 面积过滤: fg_area > 100 (14×14 圆 ~154px, 噪声 <80)
 │   │   └─ 去重: <10px 内重复检测合并
 │   │
 │   ├─ 最近邻匹配:
 │   │   ├─ 活跃轨迹按长度排序(优先匹配长轨迹)
 │   │   ├─ 每个轨迹找最近检测点 (<30px)
 │   │   └─ 未匹配的检测点 → 新建轨迹
 │   │
 │   └─ 轨迹过滤: len>100, 总位移>5px
 │
 ├─ _identify_players_for_team()
 │   ├─ EasyOCR 读取圆点上方 20px 处 98×20 名称标签
 │   ├─ 在轨迹 25%/50%/75% 位置采样
 │   ├─ _fuzzy_match(): 首字符必须匹配, 允许 1 错
 │   └─ 排除法: 2/3 已识别 → 第 3 条为剩余选手
 │
 ├─ compute_trajectory_stats() → 平滑 (5帧) → 速度/转向(>60°) → stats.json
 │
 └─ 绘图: labeled 地图底图 + 轨迹线 + 转向弧线 + 停留标记 + 时间戳
```

**HSV 颜色范围**：白(0,0,160)-(180,50,255)，黄(20,40,80)-(40,255,255)，绿(35,30,50)-(85,255,255)，红双区间，蓝(95,30,50)-(135,255,255)，紫(125,30,50)-(165,255,255)。

---

## `killreport.py` — 击杀播报检测

```
scan_view(view_dir)
 ├─ 从 game_analysis.json 读取 player_status 事件
 ├─ 对每个事件 ±3 帧内检测:
 │   ├─ 帧差: |当前ROI - 前一帧ROI|
 │   ├─ 信号: diff_mean > 8 且 high_change_px > 100
 │   │
 │   ├─ _ocr_killfeed() → EasyOCR 读 ROI (546,44) 447×58
 │   │
 │   └─ parse_event()
 │       ├─ 正则提取字母数字 token
 │       ├─ fuzzy_match() → 编辑距离匹配 KNOWN_NAMES
 │       ├─ 队伍归属: PLAYER_TEAMS 权威 (非 OCR)
 │       ├─ 分类:
 │       │   ├─ same_player → invalid
 │       │   ├─ same_team → rescue
 │       │   ├─ 含 "wipeout" → wipeout
 │       │   ├─ _has_knock_icon() → knock
 │       │   └─ 否则 → kill
 │       └─ 输出 event dict
 │
 └─ 去重: 连续事件相近帧合并
```

**检测位置**：game view (546,44)，单行 18px 高，最多 3 行(含间距 2px)。仅 text 区域约 234px 宽，右对齐。

---

## `map_matcher.py` — 地图匹配

```
MapMatcher(reference_map_path)
 ├─ 加载参考地图 → CLAHE 增强 → ORB 特征提取(3000 关键点)
 │
 └─ match(viewport)
     ├─ viewport CLAHE 增强 → ORB 特征提取
     ├─ BFMatcher(NORM_HAMMING, crossCheck=True) → 筛选 distance<50 的匹配
     ├─ findHomography(RANSAC, 5.0px)
     └─ 返回 3×3 单应矩阵 (viewport px → ref map px)
```

---

## `heatmap.py` — 速度热力图

```
generate_heatmap(trajectories, map_path, output_path)
 ├─ 载入 labeled 地图底图
 ├─ 逐轨迹累加速度:
 │   └─ speed = sqrt(dx²+dy²) / dt → cv2.line() 绘制到 float32 画布
 ├─ 按权重归一化 → GaussianBlur(25) → JET colormap
 ├─ 与底图混合 (addWeighted 0.3/0.7)
 └─ 右侧色条标注速度范围 (px/s)
```

---

## 公共模式

- **所有文件 I/O**：`open(path, encoding="utf-8")` 保证 Windows GBK 兼容
- **OCR 容错**：Tesseract 找不到时输出警告 (Windows 路径提示)，不会崩溃
- **颜色统一**：`COLOR_ALIASES = {"orange": "yellow"}` 消除色条检测歧义
- **时序平滑**：多数投票 (颜色)、5 帧平滑 (轨迹)、5 帧确认 (排名变化)
