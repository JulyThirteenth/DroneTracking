# DroneTracking FSM

`fsm` 是 DroneTracking 的状态机与控制执行层。整体按四层组织：

```text
FSM -> Behavior -> Controller -> Px4Bridge
```

## 四层架构

### 1. FSM

FSM 负责 ROS 侧的系统编排：

- 接收 ROS 话题信息，例如 FSM 命令、车辆状态、路径、参考轨迹、yaw command、障碍物扫描数据。
- 维护和发布 FSM 状态。
- 根据命令触发状态转移。
- 在每个控制周期调用 behavior。

对应实现主要在：

- `fsm_node.py`
- `fsm_core.py`
- `fsm_spec.py`
- `fsm_main.py`

### 2. Behavior

Behavior 负责把 FSM 状态和 ROS 话题输入翻译成 controller 的输入：

- 根据当前 FSM 状态执行起飞、悬停、跟踪、返航、降落等行为。
- 管理 hover target、tracking reference、yaw command、vehicle state。
- 在 MPC 和 MPCC 模式下组织不同的 controller 输入。
- 将 controller 输出转换成 PX4 body-rate/thrust command。

对应实现主要在：

- `fsm_mpc.py`
- `fsm_wrap.py`

### 3. Controller

Controller 负责根据 behavior 提供的输入计算具体控制量：

- MPC：基于参考轨迹计算控制指令。
- MPCC：基于路径点和进度变量计算控制指令。
- 输出 body rate 和 thrust 等低层控制量。

Controller 实现在 `tracking` 模块中，FSM 侧通过 `tracking.tracking_cnt.PathTrackerCtbr` 调用。

### 4. Px4Bridge

Px4Bridge 负责和 PX4/无人机平台通信：

- 发布 `OffboardControlMode`。
- 发布 `VehicleRatesSetpoint`。
- 发布 `VehicleCommand`，例如 arm、offboard、land。
- 通过 `px4_msgs` 与 PX4 ROS2 bridge 通信。

对应实现：

- `fsm_ros.py`

## 文件说明

- `fsm_main.py`
  - FSM node 启动入口。
  - 创建 `DroneFSMNode`。
  - 创建 logger、behavior、tracker。
  - 负责 `rclpy.init()`、`spin()`、shutdown。

- `fsm_node.py`
  - ROS2 FSM node 主体。
  - 订阅命令、车辆状态、路径、参考轨迹、yaw command、障碍物扫描。
  - 发布 FSM state 和 info。
  - 驱动 `FiniteStateMachine` 和 behavior tick。
  - 处理 auto land 判断。

- `fsm_core.py`
  - FSM 核心实现。
  - 定义 `Event` 和 `FiniteStateMachine`。
  - 根据 transition 表完成状态转移。

- `fsm_spec.py`
  - FSM 状态、事件和转移表定义。
  - 定义命令别名到事件的映射。
  - 是状态机行为的静态规格。

- `fsm_wrap.py`
  - 抽象基类和基础封装。
  - 定义 `FSMLoggerBase`、`FSMBehaviorBase`、`FSMNodeBase`。
  - `FSMBehaviorBase` 持有 logger、tracker、Px4Bridge。

- `fsm_mpc.py`
  - MPC/MPCC behavior 实现。
  - `MPCBehavior` 面向参考轨迹。
  - `MPCCBehavior` 面向路径跟踪。
  - 将 FSM 状态和输入数据组织成 controller 调用。

- `fsm_ros.py`
  - ROS/PX4 辅助层。
  - 定义 `VehicleState`。
  - 提供 ROS message 到 ENU 数据的转换函数。
  - 提供 `Px4Bridge`，通过 `px4_msgs` 发布控制指令。

- `fsm_log.py`
  - FSM 日志记录。
  - 记录 event、tick、meta。
  - 输出 CSV，供后续分析和绘图使用。

- `fsm_plot.py`
  - FSM 日志绘图工具。
  - 读取 `fsm_log.py` 生成的日志并生成曲线图。

- `fsm_interface.py`
  - 交互式 FSM 命令行界面。
  - 发布 `/fsm/cmd`。
  - 订阅 `/fsm/state` 和 `/fsm/info`。
  - 根据当前状态提示可用命令。

## 启动方式

FSM node：

```bash
python -m fsm.fsm_main
```

交互式命令界面：

```bash
python -m fsm.fsm_interface
```

通常由 `tools/run_code.sh` 或 `tools/run_oa_code.sh` 启动。
