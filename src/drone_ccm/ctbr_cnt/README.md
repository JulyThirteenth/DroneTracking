# CTBR controller learning

本目录只维护一套训练、仿真和对比基线：标准 CCM、Ego-CCM、SO3-CTBR、
SO3-Full 使用相同轨迹、初值、飞行器、扰动和控制边界。

## 目录

```text
cfg/benchmark.yaml       唯一 benchmark 配置
cfg/iris.yaml            Pegasus Iris 动力学参数
test/                    动力学、certificate 和 benchmark 回归测试
uav_ccm.py               world-frame CCM 训练与加载
uav_ego_ccm.py           ego-centric CCM 训练与加载
uav_sim.py               四旋翼、分配器和 CTBR 内环
uav_so3.py               SO3 baseline
track_benchmark.py       四控制器统一评测入口
```

`fig/` 是可再生输出并已被 Git 忽略。旧 fixed-yaw、observer 和 incremental
实验不属于当前 baseline，已从维护路径移除。

## 模型与边界

| Checkpoint | 用途 | ROS 支持 |
|---|---|---|
| `neu_ccm_practical.pt` | 旧 v9 Pegasus 部署回归模型 | 是 |
| `neu_ccm_linear.pt` | 当前 world-frame CCM baseline | 否 |
| `neu_ego_ccm_active.pt` | 当前 Ego-CCM baseline | 是 |

控制量均为物理 FLU CTBR：`u=(c,p,q,r)`。统一边界为：

```text
lower = [1.691379, -3.839724, -3.839724, -1.570000]
upper = [15.222414, 3.839724, 3.839724, 1.570000]
normalized_thrust = 0.58 * c / 9.81 in [0.10, 0.90]
```

共同训练域：参考速度每轴 `[-2.5,2.5] m/s`，速度误差每轴
`[-1.5,1.5] m/s`，参考倾角 `<=1.0 rad`，姿态误差 `<=0.7 rad`，参考
collective `[6.81,12.81] m/s²`，参考 body-rate `±[1.0,1.0,0.5] rad/s`。

标准 CCM 状态为 `(v,R)`，Ego-CCM 状态为
`(gamma,delta_v_body,delta_psi)`。Ego certificate 只在合法的六维切空间
`T_gamma S2 x R3 x S1` 上计算。两种控制器均采用可训练线性支路加神经
residual，并联合学习对偶度量 `W`：

```text
L = L_closed + L_C1 + L_C2 + L_W_upper
```

## 训练

```bash
cd ~/devspace/agile_flight/DroneTracking/src/drone_ccm/ctbr_cnt

# World-frame CCM；默认输出 neu_ccm_linear.pt
pixi run python -u uav_ccm.py \
  --epochs 30 --batch-size 1024 \
  --training-size 131072 --validation-size 32768 \
  --hidden 64 --controller-hidden 64 --rate 0.5 \
  --lr 1e-3 --lr-step 5 --lr-gamma 0.3 --device cuda

# Ego-CCM
pixi run python -u uav_ego_ccm.py \
  --epochs 30 --batch-size 1024 \
  --training-size 131072 --validation-size 32768 \
  --hidden 64 --rate 0.5 --lr 1e-3 --device cuda \
  --checkpoint neu_ego_ccm_active.pt
```

当前独立验证集结果：

| 模型 | contraction | 最大 `eig(C)` | C1 | C2 loss |
|---|---:|---:|---:|---:|
| World-frame CCM | 99.96643% | 0.94519 | 100% | 0.00302 |
| Ego-CCM | 99.97253% | 0.82002 | 100% | 0.01115 |

这是采样验证，不是连续域的 100% 证明。

## Benchmark

默认命令即完整基线：4 个控制器、4 条轨迹、`1/1.5/2x`、3 个固定 seed、
动态 yaw。位置不参与反馈，仅作为速度误差积分漂移指标。

```bash
MPLCONFIGDIR=/tmp/ctbr_mpl pixi run python -u track_benchmark.py --no-show
```

快速回归可限制速度或控制器：

```bash
MPLCONFIGDIR=/tmp/ctbr_mpl pixi run python -u track_benchmark.py \
  --controller ego-ccm --speed-scale 1.0 --output-dir fig/smoke --no-show
```

输出包含逐轨迹 PNG、`track_benchmark_metrics.csv`、总表
`benchmark_summary.csv` 和带源码/checkpoint 哈希的 `manifest.json`。

最近一次完整基线共 `144/144` 稳定：

| 控制器 | 速度 RMSE | yaw RMSE | 漂移 RMSE | torque RMS | 分配饱和 |
|---|---:|---:|---:|---:|---:|
| SO3-CTBR | 0.06979 m/s | 0.45945 deg | 0.18148 m | 0.05767 N·m | 0% |
| SO3-Full | 0.06875 m/s | **0.27933 deg** | 0.18396 m | 0.13001 N·m | 0.271% |
| World-frame CCM | 0.10100 m/s | 0.84956 deg | 0.25417 m | 0.04620 N·m | 0% |
| Ego-CCM | **0.06791 m/s** | 0.91337 deg | 0.21559 m | **0.04565 N·m** | 0% |

## 验证

```bash
python -m py_compile uav_*.py track_benchmark.py test/*.py
MPLCONFIGDIR=/tmp/ctbr_mpl pixi run python test/test_benchmark.py
MPLCONFIGDIR=/tmp/ctbr_mpl pixi run python test/test_ego_ccm.py
pixi run python test/verify_contraction.py
```

ROS 部署接口、`hover_thrust` 和启动命令见上级 [`README.md`](../README.md)。
