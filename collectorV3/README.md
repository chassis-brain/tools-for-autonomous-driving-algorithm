# CollectorV3

## 1. 设计思路

CollectorV3 是一个面向自动驾驶算法的失败复现与数据采集框架。

核心流程：

    E2E Algorithm
          |
          | 运行并记录真实行为
          v
    Behavior Tape
          |
          | 选择问题片段
          v
    Record Start + Handoff
          |
          | 控制权切换
          |
          +----------------+
          |                |
          v                v
     PDM Expert      Manual PID Expert
          |                |
          +----------------+
                  |
                  v
            Spatial End
                  |
                  v
          Dataset Export

设计原则：

-   Record Start 和 Handoff 完全复现原 E2E 时空状态。
-   End 为自由空间目标点，不绑定原 E2E 时间。
-   支持专家接管和人工规划两种恢复方式。
-   采集过程与具体 E2E 算法解耦。

------------------------------------------------------------------------

## 2. 基本使用方法

启动：

``` bash
conda activate b2d-collector37

cd collectorV3

bash scripts/run_failure_replay_gui.sh
```

GUI流程：

1.  启动 CARLA。
2.  运行 E2E 算法。
3.  自动记录 Behavior Tape。
4.  在轨迹页面选择：
    -   Record Start
    -   Handoff
    -   End
5.  选择恢复方式：
    -   PDM Expert
    -   Manual PID
6.  执行 Recovery。
7.  自动生成采集数据。

------------------------------------------------------------------------

## 3. 更换自动驾驶算法

CollectorV3 不绑定具体 E2E 模型。

替换算法时：

1.  保持 CARLA vehicle 控制接口不变。
2.  保证算法输出：

```{=html}
<!-- -->
```
    VehicleControl
        |
        + throttle
        + brake
        + steer

3.  修改对应启动脚本：

```{=html}
<!-- -->
```
    scripts/

4.  修改配置：

```{=html}
<!-- -->
```
    configs/

5.  重新运行 Probe。

新的算法运行结果会自动生成新的 Behavior Tape。

------------------------------------------------------------------------

## 4. 修改采集数据格式

采集数据由 Writer 管理：

    src/b2d_collector/writer.py

如果需要增加字段：

例如：

-   新传感器
-   新状态量
-   新标注

修改：

    writer.py

对应：

    sensor collection
    annotation generation
    dataset export

如果需要适配新的训练框架：

新增：

    configs/dataset_profiles/

定义新的数据格式。

例如：

    dataset_profiles/
        custom_model.json

然后导出对应格式。

------------------------------------------------------------------------

## 5. Expert 扩展

新增恢复算法：

目录：

    src/b2d_collector/experts/

实现：

    Expert Interface
            |
            + run_step()
            + control output

即可接入：

-   新 Planner
-   新 Controller
-   新轨迹跟踪算法

------------------------------------------------------------------------

## 6. 数据目录

运行数据：

    probe_runs/
        Behavior Tape

    cases/
        intervention case

    outputs_recovery_data/
        final dataset

    logs/
        runtime logs

正式交付时这些目录可以为空。

------------------------------------------------------------------------

## 7. 环境要求

主要依赖：

-   CARLA
-   Python
-   PyTorch
-   OpenCV
-   NumPy

具体环境见：

    requirements.txt
    environment.yml
