# Failure Replay Workbench
## E2E 失败区间复现、Expert 接管与训练数据采集说明

> 当前基线：CARLA 0.9.15 + Bench2Drive/Leaderboard/ScenarioRunner + Python 3.7 collector 环境  
> 当前 Expert：CARLA Garage PDM-Lite  
> 当前 GUI：`tools/failure_replay_gui.py`  
> 当前流程定位：**人工发现失败区间，不做自动 failure mining；人工指定 Record Start / Handoff / Record End，再由 Expert 接管并导出 correction dataset。**

---

# 1. 设计理念

## 1.1 核心目标

这个项目要解决的问题不是“重新实现一个 E2E 算法”，而是把任意已经能够在 CARLA / Bench2Drive 中运行的 E2E 算法当成一个黑盒驾驶员：

```text
任意 E2E Agent
      │
      │ 正常控制 CARLA hero
      ▼
CARLA World
      │
      ├── Probe 旁路监听真实已执行动作与状态
      │
      ▼
Behavior Tape
      │
      ├── 人工复盘
      ├── 选择失败/弱表现区间
      ▼
Intervention Case
      │
      ▼
确定性 Ego Replay
      │
      ├── Expert Shadow warm-up
      ▼
Handoff
      │
      ▼
Expert Control
      │
      ▼
Base Dataset
```

因此本工程的核心思想是：

**监听 E2E，而不是侵入 E2E。**

SparseDriveV2 只是目前用来验证整条链路的一个 E2E 实例，不是 Failure Replay 的硬编码前提。

---

## 1.2 不修改源 E2E 的内部推理

Probe Listener 不需要：

- 修改 SparseDriveV2 的网络；
- 修改模型 forward；
- 修改 checkpoint；
- 在模型内部增加 hook；
- 从模型内部拿 action；
- 修改 VAD / DiffusionDrive / TransFuser 的规划头；
- 要求源 E2E 输出额外的 replay 文件。

Probe 的信息源是 CARLA 世界本身。

最重要的两类信息是：

```text
CARLA world timestamp
CARLA hero actor 当前真实状态 / 已施加 VehicleControl
```

因此只要一个 E2E Agent 最终通过 CARLA/Leaderboard 控制同一个 `hero` actor，它就可以被同一套 Probe 监听。

---

## 1.3 Behavior Tape 与训练数据必须分开

第一次 E2E 运行的主要目的不是直接生产最终训练数据，而是产生一个轻量、可复现的 **Behavior Tape**。

Behavior Tape 负责：

- 精确记录 E2E 实际执行过的 control；
- 记录 CARLA world 时间轴；
- 保存 ego 状态作为 replay fidelity reference；
- 保存必要的场景/actor 信息用于人工 review；
- 为人工选取 failure interval 提供依据。

第二次运行才由 Base Collector 采集真正用于训练的 rich dataset。

这样做的原因是：

```text
第一次运行：
E2E 自己需要什么传感器，就继续用什么传感器。
Probe 只旁路监听，不改变它的运行条件。

第二次运行：
Ego 按原 E2E Behavior Tape 重放。
到指定 Handoff 后，Expert 接管。
Base Collector 可以挂载训练需要的完整辅助传感器。
```

这避免了为了“录训练集”而改变原始 E2E 第一次运行时的感知负载和执行行为。

---

## 1.4 Replay 使用 CARLA World Time，而不是 Leaderboard Game Time

已经实机验证，Leaderboard `run_step(timestamp)` 和：

```python
world.get_snapshot().timestamp.elapsed_seconds
```

可能存在固定时间差。

因此 Failure Replay 的统一时间基准是：

```text
CARLA world elapsed_seconds
```

当前已经验证并冻结的 control causal mapping：

```text
control_target_time
=
CARLA_world_time
+
case.time_offset_seconds
+
1 * tape_dt
```

其中：

```text
case.time_offset_seconds = 0.0
causal_shift             = +1 tick
```

`+1 tick` 已经由：

```text
src/b2d_collector/failure_replay/agent.py
```

实现。

**不要在 case 中再人为写 `time_offset_seconds = 0.1`。**

---

## 1.5 Handoff 的含义

人工定义三个时间点：

```text
Record Start
Handoff
Record End
```

语义为：

```text
                         Record Start
                              │
E2E Replay + PDM Shadow       │ Collector ON
──────────────────────────────┼────────────────────
                              │
                              │
                         Handoff
                              │
                              ▼
                         PDM Expert
                              │
                              │ Collector ON
                              ▼
                         Record End
```

在 Handoff tick：

1. 当前 ego 物理状态仍然是原 E2E Replay 到达的状态；
2. 从这一帧开始 `control_source = pdm_expert`；
3. Expert 返回的 control 影响下一个 CARLA physics tick。

所以它形成的是一个明确的 counterfactual branch：

```text
相同的失败前历史
        │
        ├── 原始分支：E2E 继续
        │
        └── 新分支：Expert 接管
```

---

## 1.6 Expert Shadow Warm-up

PDM-Lite 在正式 Handoff 前可以持续读取正常输入并更新自己的 planner/controller 状态，但不能改变 ego control。

因此 replay 阶段是：

```text
E2E Replay     -> 真正控制车辆
PDM Shadow     -> 只更新内部状态
```

到 Handoff：

```text
E2E Replay OFF
PDM Shadow OFF
PDM Control ON
```

当前实现还隔离了 PDM 对 `CarlaDataProvider.active_scenarios` 的修改，避免 shadow 过程改变 ScenarioRunner 的真实状态。

---

## 1.7 当前对“场景复现”的准确表述

当前已经严格验证的是：

```text
Ego replay 可以达到 0.000 m / 0.000 deg 级别的一致性
```

周围车辆不是从 Behavior Tape 用 `set_transform()` 逐帧强制播放。

它们由：

```text
相同 Bench2Drive Route
+ 相同 ScenarioRunner
+ 相同 Traffic Manager seed
+ 相同 ego 行为历史
```

重新自然执行。

因此当前设计属于：

**ego action deterministic replay + same scenario re-execution**

而不是：

**full-world actor forced playback**

这样可以保持 ScenarioRunner / Traffic Manager 的内部状态自然演化，避免 Handoff 后出现“actor 外部位置被强制对齐但内部控制器历史不一致”的问题。

---

# 2. 数据分层设计

系统中实际上有两类完全不同的数据。

## 2.1 Behavior Tape：用于 Replay

它必须保持算法无关。

最低必要信息：

```text
world frame / timestamp
ego transform
ego velocity
ego applied VehicleControl
route/run identity
```

推荐同时保留：

```text
ego acceleration
ego angular velocity
nearby actor state
traffic information
run metadata
```

当前关键字段示意：

```text
frames.jsonl
└── world.elapsed_seconds
└── ego.state.transform
└── ego.state.velocity
└── ego.applied_control
└── ...
```

Replay 真正执行的是记录下来的 ego control。

ego state reference 用于检查 Replay 是否发散。

---

## 2.2 Base Dataset：用于训练

Expert Recovery 运行时 Base Collector 才负责采集 rich data。

当前工程的 full 数据能力包括：

- 多视角 RGB；
- depth；
- semantic segmentation；
- instance segmentation；
- LiDAR；
- Radar；
- GPS；
- IMU；
- speed；
- ego pose / velocity / acceleration；
- navigation command；
- applied control；
- nearby actor 3D annotation；
- sensor intrinsic / extrinsic；
- `control_source`；
- failure/intervention metadata。

核心原则：

> **Base Dataset 尽量保存模型无关的 raw superset；VAD / DiffusionDrive / TransFuser 的差异放到后处理 exporter / index builder 中，而不是重新设计 Replay。**

---

# 3. 当前目录结构

推荐仓库根目录：

```text
b2d-custom-collector/
├── README_FAILURE_REPLAY_WORKBENCH.md
├── requirements.txt
├── pyproject.toml
├── environment.yml
├── LICENSE
├── .gitignore
│
├── replay_to_pdm_agent.py
│
├── configs/
│   ├── replay_to_pdm.example.yaml
│   └── dataset_profiles/
│       └── diffusiondrive_b2d_v1.json
│
├── scripts/
│   ├── run_replay_to_pdm.sh
│   └── run_failure_replay_gui.sh
│
├── tools/
│   ├── failure_replay_gui.py
│   ├── e2e_probe_listener.py
│   ├── validate_probe.py
│   ├── review_probe.py
│   ├── make_intervention_spec.py
│   ├── preflight_recovery.py
│   └── compare_probe_runs.py
│
├── src/
│   └── b2d_collector/
│       ├── __init__.py
│       ├── agent.py
│       ├── annotations.py
│       ├── config.py
│       ├── controller.py
│       ├── navigation.py
│       ├── rig.py
│       ├── route_xml.py
│       ├── sensors.py
│       ├── writer.py
│       ├── validate_dataset.py
│       ├── dataset_profiles.py
│       ├── build_diffusiondrive_index.py
│       │
│       ├── experts/
│       │   ├── __init__.py
│       │   ├── template.py
│       │   └── pdm_lite.py
│       │
│       └── failure_replay/
│           ├── __init__.py
│           ├── tape.py
│           ├── spec.py
│           ├── config.py
│           └── agent.py
│
├── third_party/
│   └── carla_garage/
│       ├── LICENSE
│       └── team_code/
│           └── ...
│
└── tests/
    └── ...
```

---

# 4. 必须提交到 Git 的文件

## 4.1 当前 Failure Replay / GUI 主链必须提交

这些文件决定当前系统能否在另一台机器重新搭起来：

```text
.gitignore
LICENSE
pyproject.toml
environment.yml
requirements.txt
README_FAILURE_REPLAY_WORKBENCH.md

replay_to_pdm_agent.py

configs/replay_to_pdm.example.yaml

scripts/run_replay_to_pdm.sh
scripts/run_failure_replay_gui.sh

tools/failure_replay_gui.py
tools/e2e_probe_listener.py
tools/validate_probe.py
tools/review_probe.py
tools/make_intervention_spec.py
tools/preflight_recovery.py
tools/compare_probe_runs.py

src/b2d_collector/__init__.py
src/b2d_collector/agent.py
src/b2d_collector/annotations.py
src/b2d_collector/config.py
src/b2d_collector/controller.py
src/b2d_collector/navigation.py
src/b2d_collector/rig.py
src/b2d_collector/route_xml.py
src/b2d_collector/sensors.py
src/b2d_collector/writer.py
src/b2d_collector/validate_dataset.py

src/b2d_collector/experts/__init__.py
src/b2d_collector/experts/template.py
src/b2d_collector/experts/pdm_lite.py

src/b2d_collector/failure_replay/__init__.py
src/b2d_collector/failure_replay/tape.py
src/b2d_collector/failure_replay/spec.py
src/b2d_collector/failure_replay/config.py
src/b2d_collector/failure_replay/agent.py
```

---

## 4.2 如果继续使用当前 PDM-Lite Expert，也必须保留

当前 `PdmLiteExpert` 依赖项目内：

```text
third_party/carla_garage/team_code/
```

并且当前 `autopilot.py` 含有用于 Shadow 安全性的本地保护。

因此不能只提交：

```text
autopilot.py
```

然后假设其他机器会自动拥有其依赖。

推荐两种方式二选一：

### 方式 A：直接 vendor

提交当前 PDM-Lite 运行真正使用到的：

```text
third_party/carla_garage/
```

并保留：

```text
third_party/carla_garage/LICENSE
```

这是当前最简单、最不容易出现版本漂移的方式。

### 方式 B：Git submodule / fork

如果未来清理仓库：

```text
third_party/carla_garage
```

可以改为固定 commit 的 submodule 或自己的 fork。

但必须保证：

```text
AutoPilot shadow-safe patch
```

仍然存在。

---

## 4.3 模型适配/训练导出相关，建议一起提交

当前项目已经存在 DiffusionDrive 数据 profile 的情况下，应提交：

```text
configs/dataset_profiles/diffusiondrive_b2d_v1.json
src/b2d_collector/dataset_profiles.py
src/b2d_collector/build_diffusiondrive_index.py
src/b2d_collector/build_e2e_index.py
```

这样 Base Dataset 与具体训练模型之间仍保持独立。

---

## 4.4 Tests 必须提交

推荐直接提交：

```text
tests/
```

但删除：

```text
tests/__pycache__/
```

测试不是运行时必须，但对于当前这种高度依赖时间对齐和数据 schema 的工程，Git 中没有测试会非常危险。

---

# 5. 不应该提交到 Git 的内容

以下都是机器本地数据或大文件：

```text
probe_runs/
outputs/
outputs_recovery/
outputs_recovery_data/
logs/
checkpoints/
close_loop_log/
__pycache__/
.pytest_cache/
*.pyc
*.pyo
```

也不要提交：

```text
CARLA_0.9.15/
SparseDriveV2/
Bench2Drive/
ScenarioRunner 独立安装目录
*.egg
*.whl
E2E 大模型 checkpoint
```

---

## 5.1 `replay_to_pdm.yaml` 不建议直接提交

实际运行的：

```text
configs/replay_to_pdm.yaml
```

通常包含机器绝对路径，例如：

```text
/home/user/...
```

因此 Git 中应提交：

```text
configs/replay_to_pdm.example.yaml
```

真实配置由 GUI 在本机生成/更新。

---

## 5.2 生成的 case 默认也不应直接作为代码提交

例如：

```text
cases/case_1711_001.json
```

当前 `source_run` 可能包含绝对路径。

如果某个 case 要作为论文/实验基准保存，建议：

1. 将路径转换为相对路径或逻辑 ID；
2. 放入：

```text
cases/examples/
```

或：

```text
cases/golden/
```

再提交。

普通临时 case 不需要进入 Git。

---

# 6. 推荐 `.gitignore`

至少应包含：

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.venv/
*.egg-info/

dist/
build/

probe_runs/
outputs/
outputs_recovery/
outputs_recovery_data/
logs/
close_loop_log/
checkpoints/

configs/replay_to_pdm.yaml
cases/case_*.json

*.egg
*.whl
```

如果需要保存 curated golden case，可以通过反向规则单独保留：

```gitignore
!cases/golden/
!cases/golden/*.json
```

---

# 7. 环境与外部依赖

当前 collector 基线：

```text
Python 3.7.x
CARLA 0.9.15
Bench2Drive / Leaderboard / ScenarioRunner
10 Hz synchronous simulation
```

本项目 `requirements.txt` 只负责 Python 层的通用依赖。

以下依赖不要简单通过普通 requirements 自动安装：

## CARLA Python API

使用 CARLA 安装包自带的 Python 3.7 egg：

```text
CARLA_0.9.15/PythonAPI/carla/dist/
carla-0.9.15-py3.7-linux-x86_64.egg
```

不要误用：

```text
cp27
py2.7
```

---

## Torch / CUDA

Torch 版本与：

- GPU；
- CUDA；
- E2E repo；
- PDM/CARLA Garage 环境

强耦合。

因此不在通用 `requirements.txt` 中强制 pin 一个 CUDA build。

应按机器和算法环境安装。

---

## Tkinter

GUI 使用 Tkinter。

如果：

```bash
python -c "import tkinter"
```

失败，在 conda 环境中可以安装：

```bash
conda install tk
```

---

# 8. GUI 操作流程

启动：

```bash
conda activate b2d-collector37
cd /path/to/b2d-custom-collector
bash scripts/run_failure_replay_gui.sh
```

GUI 分四页：

```text
1. Setup
2. Route / E2E Probe
3. Review / Case
4. Expert Recovery / Export
```

---

# 9. Step 1 — Setup

第一次换机器时配置：

```text
Collector project root
CARLA root
Bench2Drive / SparseDrive root
E2E Python executable
Route XML
E2E agent
E2E config
E2E checkpoint
Recovery YAML
CARLA port
Traffic Manager port
GPU rank
```

点击：

```text
Auto-fill derived paths
```

再点击：

```text
Save settings
Check installation
```

## 这一步在干什么？

它只解决“不同机器路径不同”的问题。

GUI 会把本机配置保存到：

```text
~/.failure_replay_workbench.json
```

这些配置不进入 Git。

Replay 算法本身不依赖这些绝对路径。

---

# 10. Step 2 — 选择 Route

GUI 从：

```text
bench2drive220.xml
```

读取 Route ID。

用户从下拉框选择，例如：

```text
1711
2091
...
```

## 这一步在干什么？

Route ID 决定：

- Town；
- route geometry；
- ScenarioRunner 场景；
- traffic setup；
- weather/scenario 配置。

之后原 E2E 和 Recovery 必须运行同一个 Route。

---

# 11. Step 3 — Start CARLA

点击：

```text
Start CARLA
```

## 这一步在干什么？

GUI 等价于启动：

```bash
CarlaUE4.sh -carla-port=23000 -quality-level=Low
```

CARLA 是唯一真实 simulation world。

后续 Probe、E2E、Recovery 都连接这个 server。

---

# 12. Step 4 — Start Probe Listener

在原始 E2E 启动前点击：

```text
Start Probe Listener
```

## 这一步在干什么？

Probe 等待：

```text
role_name = hero
```

出现。

E2E 一旦创建 ego vehicle，Probe 开始旁路监听。

重要的是：

```text
Probe 不参与 E2E 推理
Probe 不返回 control
Probe 不修改 E2E
```

它只观察 CARLA 中已经发生的结果。

---

# 13. Step 5 — Run Original E2E

当前 GUI v1 默认提供 SparseDrive 风格的启动参数：

```text
E2E Python
agent.py
config.py
checkpoint
Route ID
```

点击：

```text
Run selected Route
```

## 这一步在干什么？

原 E2E 按自己的正常流程驾驶。

Probe 同时生成：

```text
probe_runs/probe_*/
```

这里得到的是原始 E2E 的 Behavior Tape。

---

# 14. Step 6 — Validate Probe

原 E2E 完成后：

```text
Recorded probes
-> 选择最新 Probe
-> Validate
```

目标：

```text
frame_gaps         = []
non_monotonic_time = []
missing_control    = []
COMPLETE           = True
manifest_integrity = True
RESULT             = PASS
```

## 这一步在干什么？

它检查的是 Replay 输入是否可信。

如果：

- 时间戳断裂；
- control 丢失；
- Probe 没有正常结束；

那么不要进入 Recovery。

因为错误的 Behavior Tape 会导致后续所有数据失去意义。

---

# 15. Step 7 — Review Probe

点击：

```text
Load in Review
```

GUI 会：

- 读取 `frames.jsonl`；
- 绘制 ego trajectory；
- 显示当前 frame；
- 显示 simulation time；
- 显示 speed；
- 显示 throttle / brake / steer。

用户可以：

- 拖动时间滑块；
- 点击轨迹位置；
- 手工复盘失败/弱表现。

---

# 16. Step 8 — 定义 Intervention Window

选择当前 frame 后依次：

```text
Set Record Start
Set Handoff
Set Record End
```

推荐概念：

```text
Record Start
=
失败真正发生前一段上下文

Handoff
=
希望 Expert 开始改写未来的位置

Record End
=
专家纠正完成并恢复稳定后
```

例如：

```text
Record Start = 8.5 s
Handoff      = 13.5 s
Record End   = 20.4 s
```

## 为什么不只录 Handoff 以后？

训练 correction policy 时，失败前 context 往往很重要。

所以建议 clip 包含：

```text
失败前上下文
+
失败/犹豫发展
+
Expert 接管
+
纠正
+
短暂恢复
```

---

# 17. Step 9 — Save Case

点击：

```text
Save Case JSON
```

生成：

```text
cases/case_<route>_<id>.json
```

关键结构：

```json
{
  "record_start": {
    "sim_time_s": 8.5
  },
  "handoff": {
    "sim_time_s": 13.5
  },
  "record_end": {
    "sim_time_s": 20.4
  },
  "replay": {
    "time_offset_seconds": 0.0
  }
}
```

## 这一步在干什么？

Case 不包含训练数据。

它只是定义：

> 对哪一个 Reference Tape，在什么时间开始录、什么时间换驾驶员、什么时间结束录。

同一条 Behavior Tape 可以有多个 Case：

```text
probe_A
├── case_A_001
├── case_A_002
└── case_A_003
```

无需重新跑原 E2E。

---

# 18. Step 10 — Write Recovery YAML

进入：

```text
Expert Recovery / Export
```

点击：

```text
Write/Update recovery YAML
```

GUI 自动填写：

```text
Route ID
Reference Probe
Case JSON
output_root
shadow_pdm = true
sensor_warmup_seconds
require_complete_tape
abort_after_tape
```

## 这一步在干什么？

把：

```text
机器级路径
+
本次实验参数
```

组装成 Recovery Agent 能读取的配置。

不改变 case 时间规则。

---

# 19. Step 11 — Preflight

点击：

```text
Run Preflight
```

必须：

```text
RESULT = PASS
```

## 这一步在干什么？

它在 CARLA 真正开始运行前检查：

- Reference Tape 是否存在；
- `COMPLETE` 是否存在；
- case 是否合法；
- Start <= Handoff <= End；
- route/config 路径是否存在；
- Recovery 关键配置是否完整。

---

# 20. Step 12 — Run Expert Recovery

点击：

```text
Run Expert Recovery
```

内部流程：

```text
Route start
     │
     ▼
原 E2E control replay
+
PDM shadow
     │
     ▼
Record Start
     │
     ├── Base Collector ON
     ▼
Handoff
     │
     ├── E2E replay OFF
     ├── PDM shadow OFF
     └── PDM Expert control ON
     ▼
Record End
     │
     └── Base Collector OFF
     ▼
Route continue / finish
```

---

# 21. Step 13 — Validate Dataset

点击：

```text
Validate latest dataset
```

GUI 会展示：

```text
frame count
first timestamp
last timestamp
control_source count
source transitions
max dt error
```

例如 Golden 1711：

```text
frames = 120
time   = 8.500 .. 20.400

e2e_replay = 50
pdm_expert = 70

transition:
8.500  -> e2e_replay
13.500 -> pdm_expert
```

并同时运行：

```bash
python -m b2d_collector.validate_dataset <clip>
```

---

# 22. Step 14 — Export

Recovery 本身已经把数据写到：

```text
outputs_recovery_data/
```

所以：

```text
Export latest clip
```

只是把完整 clip 再复制到：

- NAS；
- 训练盘；
- 数据服务器；
- 另一个实验目录。

不是再次生成数据。

---

# 23. 当前必须冻结的 Replay 参数

已经经过实机 zero-error 验证，不建议随意改动：

```text
Clock:
CARLA world elapsed_seconds

case:
time_offset_seconds = 0.0

Replay:
control causal shift = +1 tape tick

Handoff:
case.handoff.sim_time_s
```

如果未来换 E2E 后 Replay 不再精确，应先重新做 alignment 验证，而不是无依据修改这些参数。

---

# 24. 换一个 E2E 算法，哪些东西需要改？

这是整个架构最重要的可扩展性原则。

先区分两个问题：

## A. 换“原始失败算法”

例如：

```text
SparseDriveV2
→ VAD
→ DiffusionDrive
→ TransFuser
```

此时你只是换：

> 谁在第一次运行里驾驶 hero。

**Probe 不需要跟着模型改变。**

---

## B. 换“最终要训练的数据格式”

例如 correction dataset 最终要拿去训练：

```text
VAD
DiffusionDrive
TransFuser
```

此时可能要改变：

- Base Collector sensor profile；
- dataset exporter；
- index builder；
- future trajectory generation；
- annotation transformation。

**这仍然不应该修改 Behavior Tape / Replay 核心。**

---

# 25. 为什么源 E2E 更换时 Behavior Tape 不需要变？

无论 E2E 内部输出的是：

```text
waypoints
trajectory
occupancy
vector queries
diffusion samples
direct control
```

最终 CARLA 真正接受的还是：

```python
carla.VehicleControl(
    throttle=...,
    steer=...,
    brake=...
)
```

所以 Replay 所需要的不是模型的 hidden feature，而是：

```text
该 E2E 最后真正让车执行了什么
```

Probe 从 CARLA hero 上读取已经施加的 control。

因此：

```text
E2E 内部 representation
        │
        │ 与 Replay 无关
        ▼
最终 VehicleControl
        │
        ▼
Probe
```

这就是整个工具链能够跨算法的原因。

---

# 26. 当前 GUI 为什么看起来有 SparseDrive 专用字段？

因为 GUI v1 目前把“启动 E2E 的 launcher”先按已经验证的 SparseDriveV2 命令封装了。

这只是：

```text
launcher adapter
```

不是：

```text
replay core
```

也不是：

```text
probe core
```

当前最简单的跨算法使用方法甚至不需要改 GUI 核心：

```text
GUI:
Start CARLA
Start Probe Listener

另一个 terminal / 对应算法环境:
运行 VAD / DiffusionDrive / TransFuser

GUI:
Refresh Probe
Validate
Load Review
...
```

也就是说：

**只要另一个 E2E 在同一个 CARLA server 中创建/控制 `hero`，Probe 仍然工作。**

---

# 27. 更完整的跨 E2E GUI 适配方式

后续推荐把 E2E 启动部分做成 profile：

```text
configs/e2e_runners/
├── sparsedrive.yaml
├── vad.yaml
├── diffusiondrive.yaml
└── transfuser.yaml
```

一个 profile 只负责：

```yaml
name: VAD
python: /path/to/vad-env/bin/python
cwd: /path/to/VAD-Bench2Drive
command:
  - "{python}"
  - "leaderboard/leaderboard/leaderboard_evaluator.py"
  - "--port={carla_port}"
  - "--routes={route_xml}"
  - "--routes-subset={route_id}"
  - "--agent={agent}"
  - "--agent-config={agent_config}"
environment:
  IS_BENCH2DRIVE: "True"
```

那么 GUI 只负责模板替换和 subprocess。

Probe / Replay / Collector 不需要知道这个算法叫 VAD 还是 TransFuser。

---

# 28. 换成 VAD 作为源 E2E

如果已有能够在 CARLA / Bench2Drive 中运行的 VAD Leaderboard Agent：

需要配置：

```text
VAD Python environment
VAD working directory
Leaderboard evaluator
VAD agent path
VAD config
VAD checkpoint
Route XML
```

运行流程仍然：

```text
Start CARLA
Start Probe
Run VAD
Validate Probe
Review
Case
Recovery
```

不需要：

```text
修改 VAD forward
修改 VAD planner
在 VAD 内加 Behavior Tape
```

VAD 只是第一次 E2E run 的 driver。

---

# 29. 换成 DiffusionDrive 作为源 E2E

同理。

如果 DiffusionDrive 已有 CARLA/Bench2Drive Agent：

```text
Start CARLA
Start Probe
Run DiffusionDrive Agent
```

Probe 仍然只读取 hero 最终 control。

Diffusion model 内部：

- denoising steps；
- trajectory modes；
- anchors；
- planning representation

都不需要写进 Behavior Tape。

---

# 30. 换成 TransFuser 作为源 E2E

同理。

TransFuser 可以内部使用：

```text
RGB
+
LiDAR
+
sensor fusion
```

Probe 不需要读取网络内部 RGB/LiDAR feature。

它只需要：

```text
hero state
hero applied VehicleControl
CARLA world time
```

因此源 E2E 输入模态不同不会改变 Replay contract。

---

# 31. 如果最终 correction data 要训练 VAD

这时才需要考虑 VAD 的训练数据需求。

VAD 是 vectorized end-to-end driving paradigm；CARLA/Bench2Drive 版本的配置通常需要多视角图像以及 ego/agent/map/trajectory 类监督。

推荐 Base Dataset 至少保留：

```text
6-view RGB
sensor calibration
ego pose / velocity / acceleration
ego history trajectory
ego future Expert trajectory
navigation command / target point
nearby agent 3D boxes
agent velocity
agent future trajectory（可由连续帧构建）
traffic signal / stop sign information
map/vector information 或后处理所需原始地图引用
```

然后单独建立：

```text
build_vad_index.py
```

负责将 Base raw clip 转成 VAD loader 需要的：

```text
PKL / annotation index
```

不要为了 VAD 修改 Probe。

---

# 32. 如果最终 correction data 要训练 DiffusionDrive

当前项目已经有一个明确的 DiffusionDrive-CARLA profile：

```text
configs/dataset_profiles/diffusiondrive_b2d_v1.json
```

当前 v1 profile 定义：

```text
RGB:
CAM_FRONT_LEFT
CAM_FRONT
CAM_FRONT_RIGHT

LiDAR:
LIDAR_TOP

Ego:
GPS
IMU
SPEED
```

trajectory target：

```text
time_horizon    = 4.0 s
interval_length = 0.5 s
num_poses       = 8
```

features：

```text
driving_command
ego_velocity
ego_acceleration
```

targets：

```text
future trajectory
agent boxes
```

当前 v1：

```text
bev_semantic_map = false
```

已有：

```text
src/b2d_collector/build_diffusiondrive_index.py
```

可以从 Base clip 构建：

```text
diffusiondrive_samples.jsonl
```

因此 DiffusionDrive 是目前最典型的：

> raw superset collection -> model-specific export

例子。

---

# 33. 如果最终 correction data 要训练 TransFuser

TransFuser 官方 CARLA training dataset 典型包含：

```text
rgb
depth
semantics
lidar
topdown
label_raw / vehicle boxes
measurements
```

其训练阶段核心输入/监督还涉及：

```text
RGB
LiDAR
ego speed
target point
ego waypoint/future route target
BEV
vehicle labels
```

以及可选：

```text
depth
semantic
```

我们当前 Base full collector 已经覆盖其中大量 raw modality：

```text
RGB
Depth
Semantic
LiDAR
Ego measurement
3D actor boxes
Navigation
```

如果要求与 TransFuser 官方 dataloader 目录/字段完全兼容，应增加：

```text
build_transfuser_dataset.py
```

负责：

- 图像命名/目录转换；
- LiDAR `.laz` -> 目标格式；
- measurements schema 转换；
- target point 生成；
- future ego waypoint 生成；
- topdown/BEV target 构建；
- vehicle label schema 转换。

同样：

**不要修改 Probe。**

---

# 34. 三种模型的数据适配对比

| 层 | VAD | DiffusionDrive | TransFuser |
|---|---|---|---|
| Behavior Tape | 相同 | 相同 | 相同 |
| World timestamp | 必须 | 必须 | 必须 |
| Ego applied control | 必须 | 必须 | 必须 |
| Ego state reference | 必须 | 必须 | 必须 |
| Source E2E internal feature | 不需要 | 不需要 | 不需要 |
| Multi-view RGB training data | 需要 | profile v1 需要 3 views | 需要 |
| LiDAR training data | 视实现 | profile v1 需要 | 核心输入 |
| 3D actor annotation | 建议/需要 | target agent boxes | label/监督需要 |
| Ego future Expert trajectory | 需要 | 需要 | waypoint/planning supervision |
| Map/vector annotation | 重要 | 视实现 | 可通过 BEV/topdown 表达 |
| Model-specific exporter | `build_vad_index.py` | 已有 `build_diffusiondrive_index.py` | `build_transfuser_dataset.py` |

这里最重要的是第一行：

```text
Behavior Tape = 相同
```

---

# 35. 如果换 Expert，而不是换源 E2E

当前 Expert 是：

```text
PDM-Lite
```

如果未来换另一个 Expert，最好包装成标准 Leaderboard `AutonomousAgent`：

```python
setup()
sensors()
set_global_plan()
run_step()
destroy()
```

当前：

```text
src/b2d_collector/controller.py
```

通过 ExpertAdapter 管理 Expert。

如果新 Expert 也需要在 Handoff 前 warm-up，可以再提供：

```python
set_shadow_mode(True / False)
```

如果它不需要内部历史，可以不实现 Shadow。

---

# 36. 模型无关接口应该保持什么不变？

未来不管换多少 E2E，建议始终冻结以下抽象：

## Probe Contract

```text
输入：
CARLA server

输出：
Behavior Tape
```

## Replay Contract

```text
输入：
Behavior Tape
Intervention Case

输出：
原 E2E ego 行为直到 Handoff
```

## Expert Contract

```text
输入：
Leaderboard-compatible sensor data

输出：
VehicleControl
```

## Dataset Contract

```text
输入：
CARLA world + final VehicleControl

输出：
Base raw superset clip
```

## Export Contract

```text
输入：
Base raw clip

输出：
VAD / DiffusionDrive / TransFuser specific index/dataset
```

只要这五层分开，项目就不会因为换一个算法重新推倒重来。

---

# 37. 推荐未来目录扩展

```text
configs/
├── e2e_runners/
│   ├── sparsedrive.yaml
│   ├── vad.yaml
│   ├── diffusiondrive.yaml
│   └── transfuser.yaml
│
└── dataset_profiles/
    ├── vad_b2d_v1.json
    ├── diffusiondrive_b2d_v1.json
    └── transfuser_b2d_v1.json

src/b2d_collector/exporters/
├── vad.py
├── diffusiondrive.py
└── transfuser.py
```

这会让 GUI 最终变成：

```text
Source E2E:
[VAD ▼]

Training Export:
[DiffusionDrive ▼]
```

两者彼此独立。

例如完全可以：

```text
Source E2E = SparseDriveV2
Expert     = PDM-Lite
Export     = VAD training format
```

这正是 raw-superset 设计最大的价值。

---

# 38. 当前已验证的 Golden Case

Route 1711 已验证：

```text
Replay:
max position error = 0.000 m
max yaw error      = 0.000 deg

Record Start = 8.5 s
Handoff      = 13.5 s
Record End   = 20.4 s

Frames:
E2E Replay = 50
PDM Expert = 70
Total      = 120

Timestamp:
8.500 .. 20.400

Transition:
frame 0  -> e2e_replay
frame 50 -> pdm_expert
```

这条 case 应作为今后修改 Replay / GUI / Collector 后的回归基准。

---

# 39. 当前开发边界

当前已经完成：

```text
E2E 外部监听                 DONE
Behavior Tape               DONE
人工轨迹 Review              DONE
人工 Failure Window          DONE
World-time Replay            DONE
+1 tick causal alignment     DONE
PDM Shadow                  DONE
Precise Handoff             DONE
Base Dataset Collection      DONE
Dataset Validation           DONE
GUI workflow                DONE
```

当前刻意不做：

```text
自动 failure mining
自动选择 Handoff
自动批量 recovery
full-world NPC forced replay
模型 hidden feature 录制
```

先人工积累多个高质量 failure/correction case，再决定哪些自动化真正值得做。

---

# 40. External Model References

- VAD: https://github.com/hustvl/VAD
- DiffusionDrive: https://github.com/hustvl/DiffusionDrive
- TransFuser (2022/PAMI implementation): https://github.com/autonomousvision/transfuser/tree/2022
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- CARLA Garage / PDM family: https://github.com/autonomousvision/carla_garage

使用第三方代码时应保留各自 LICENSE，并分别遵守其许可证。
