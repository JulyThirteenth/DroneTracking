# drone_racing

The project is developed for SAMFC Drone Racing Competition.


## Videos:

### Chaser racing
![Chaser racing](./assets/chaser_racing.gif)

### MPC demo
![MPC demo](./assets/mpc.gif)




## IsaacSim simulating

Pegasus Simulator v5.1.0 is released for Isaac 5.1.0. \
Px4 version is v1.14.3.

```
isaac_run ./sim/sim.py
```

## Hover thrust testing

As the ctbr based control need hover thrust estimation, we develop the PID based Hover testing to get hover thrust.

```
python ./cnt/hover_cnt.py
```

## MPCC running

There are two controllers: chaser and tracker and two solvers: ipopt and osqp.

The most robust controller and solver in Pegasus Simuation is as below:


```
python main.py --controller tracker --solver osqp
```
