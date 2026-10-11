# Learning Sorting Robots (Webots + Reinforcement Learning)

Four cells, one theme: **nobody is told the sorting rule — every robot
earns it from rewards.**

| # | Cell | World | Controller | Learning |
|---|------|-------|------------|----------|
| 1 | e-puck arena | `worlds/sorting_world.wbt` | `sorting_robot` | tabular Q, 72 states (3 colours), logs, federated sharing, competitive flag |
| 2 | e-puck arena (neural) | same world, set `controller "sorting_dqn"` | `sorting_dqn` | DQN, hue-encoded perception (generalizes to unseen colours), logs |
| 3 | belt + custom arm | `worlds/conveyor_sorting.wbt` | `sorter_arm` | 2x2 Q-table, closed-form 2-link IK |
| 4 | UR5e industrial cell | `worlds/ur5e_cell.wbt` | `ur5e_sorter` + `quality_mast` | 5 learners: sort, throttle, trust, maintain (Q-lambda), opportunistic; per-gauge calibration; adversarial drift; JSON policy card |

## Requirements
- Webots R2022a or newer (UR5e/Robotiq3fGripper/E-puck PROTOs)
- Python 3.8+ configured in Webots (Tools > Preferences > Python)
- numpy (DQN + UR5e controller); matplotlib optional (plotter)

## Running
Open a world in Webots and press **Run** (or **Fast** to train quickly).

- **sorting_world.wbt**: two e-pucks learn red/blue/green -> matching
  zones from scratch. Q-tables + episode logs auto-save to
  `controllers/sorting_robot/`. After `TRAINING_EPISODES` a greedy
  evaluation prints a success rate. Training is resumable.
  - `COMPETITIVE = True` in `sorting_robot.py`: only the robot touching a
    ball is credited (emergent task division).
  - `ORANGE_BALL_1` in the world is the unseen-colour probe (no zone).
- **sorting_dqn**: edit both E-puck nodes' `controller` field to
  `"sorting_dqn"`. Delete `dqn_*.npz` to retrain. Hue encoding lets the
  trained policy route orange to the nearest learned zone zero-shot.
- **conveyor_sorting.wbt**: 3-DOF arm learns colour -> pad; first ~8
  picks are random, then accuracy locks at 100%.
- **ur5e_cell.wbt**: the full plant. Route by colour, reject defects to
  scrap, learn belt speed from the alarm beacon, learn which gauge to
  trust, budget per-gauge calibrations (Q-lambda), seize contamination
  holds for cheap calibrations, and persist everything to
  `controllers/ur5e_sorter/policy_card.json` (warm-starts next run).

### The adversarial-drift experiment
In `controllers/quality_mast/quality_mast.py`: `ADVERSARY = True` makes
the *trusted* gauge wear 3.5x faster (the learner is never told).
Compare trust behaviour, calibrations per 100 picks, and contamination
rate against `ADVERSARY = False`. Warm-starting the adversarial run from
a baseline policy card shows the transferred policy being unlearned.

## Plotting / dashboard
    cd controllers/sorting_robot && python plot_learning.py
    python dashboard.py          # then open http://localhost:8000

Both read `learning_log_*.csv` (written by the tabular e-pucks and the
DQN).

## Resetting experiments
    bash ../create_project.sh --clean    # or delete by hand:
    # controllers/sorting_robot/{qtable,share,learning_log}_*.csv
    # controllers/sorting_dqn/{dqn_*.npz,learning_log_*.csv}
    # controllers/ur5e_sorter/policy_card.json

## Key hyperparameters
| Where | Parameter | Default | Meaning |
|-------|-----------|---------|---------|
| sorting_robot | TRAINING_EPISODES | 180 | before greedy eval |
| sorting_robot | COMPETITIVE / SHARE | False / True | reward mode / federated blending |
| sorting_dqn | EPISODES, EPS_DECAY | 200, 0.97 | DQN schedule |
| ur5e_sorter | CAL_G_COST / CAL_BOTH_COST | -0.8 / -1.5 | per-gauge / both calibration |
| ur5e_sorter | GAMMA_META, LAMBDA | 0.95, 0.75 | Q(lambda) credit |
| ur5e_sorter | WARM_START, EPS_WARM | True, 0.15 | policy-card warm start |
| quality_mast | ADVERSARY, ADV_MULT | True, 3.5 | adversarial drift on trusted gauge |
| quality_mast | DRIFT_K, NOISE_K | 1e-3, 0.19 | wear rate, vibration noise |

## Project layout
    create_project.sh                         this generator
    dashboard.py                              live training dashboard
    README.md
    worlds/
      sorting_world.wbt                       e-puck arena (3 colours + orange probe)
      conveyor_sorting.wbt                    custom 3-DOF arm cell
      ur5e_cell.wbt                           UR5e + dual-gauge mast cell
    controllers/
      sorting_robot/sorting_robot.py          tabular Q-learning e-pucks
      sorting_robot/plot_learning.py          learning-curve plots
      sorting_dqn/sorting_dqn.py              DQN e-pucks (hue perception)
      sorter_arm/sorter_arm.py                3-DOF belt arm
      ur5e_sorter/ur5e_sorter.py              5-learner plant controller (v6)
      quality_mast/quality_mast.py            dual-gauge PLC + alarm (v5)

## Troubleshooting
- **Controller not found**: folder name must exactly match the
  controller string in the world file.
- **Camera errors (older Webots)**: remove `width/height` from
  `E-puckCamera` slots.
- **Gripper never locks (UR5e)**: nudge `TCP` (0.14-0.18) in
  `ur5e_sorter.py` to match your Webots version's gripper geometry.
- **IK residuals**: raise `iters` to 200 or lower `lam` to 0.03 in
  `solve_ik`.
