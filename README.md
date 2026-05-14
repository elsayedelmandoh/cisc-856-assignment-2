# cisc 856 assignment 2 — monte carlo methods in rl

elsayed elmandouh - 20596379 - reinforcement learning - queen's university

[![github](https://img.shields.io/badge/GitHub-elsayedelmandoh-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/elsayedelmandoh/cisc-856-assignment-2)
[![x](https://img.shields.io/badge/X-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/aangpy)
[![linkedin](https://img.shields.io/badge/elsayed-linkedin-0077b5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/elsayed-elmandoh-b5849a1b8/)

---

## table of contents

- [overview](#overview)
- [gridworld environment](#gridworld-environment)
- [algorithms](#algorithms)
- [setup](#setup)
- [usage](#usage)
- [project structure](#project-structure)
- [results](#results)
- [report](#report)
- [author](#author)

---

## overview

this project applies four monte carlo reinforcement learning methods to a 3x4 gridworld environment. the agent starts at state 8 (bottom-left) and must navigate around a wall and past a danger cell to reach the goal at state 3 (top-right) for a +1 reward

### algorithms implemented

| # | algorithm | type | points |
|---|-----------|------|--------|
| i | on-policy mc with exploring starts | control (on-policy) | 10 |
| ii | on-policy mc without exploring starts | control (on-policy) | 10 |
| iii | off-policy mc prediction (weighted is) | prediction (off-policy) | 10 |
| iv | off-policy mc control (weighted is) | control (off-policy) | 10 |

**code total: 40/40**

---

## gridworld environment

```
 0   1   2   3(g)     ← goal (green, +1, terminal)
 4   5(w) 6   7(d)    ← wall (blocked), danger (red, -1, non-terminal)
 8(s) 9  10  11       ← start (bottom-left)
```

| parameter | value |
|-----------|-------|
| grid size | 3x4 (12 states) |
| actions | up, right, left, down |
| start | state 8 (bottom-left) |
| wall | state 5 (bounce back) |
| goal | state 3 (+1, terminal) |
| danger | state 7 (-1, non-terminal) |
| step reward | -0.1 |
| max steps | 30 |
| discount factor | 0.95 |

---

## algorithms

### i. on-policy mc with exploring starts
each episode starts at a random (state, action) pair, then follows the greedy policy. guarantees full coverage of the state-action space from episode 1

### ii. on-policy mc without exploring starts
always starts from state 8. picks random actions 10% of the time, greedy 90%. true on-policy learning without exploring starts

### iii. off-policy mc prediction
evaluates a target policy (the greedy policy from algorithm i) using episodes collected by a uniform random behaviour policy. uses weighted importance sampling to correct for the distribution mismatch

### iv. off-policy mc control
learns a greedy target policy while following a uniform random behaviour policy. the most general setup — full exploration while converging to the optimal policy

all four use **first-visit mc** for lower variance

---

## setup

```bash
# clone repository
git clone https://github.com/elsayedelmandoh/cisc-856-assignment-2
cd cisc-856-assignment-2

# create environment
conda create -n cisc856 python=3.12 -y
conda activate cisc856

# install dependencies
pip install -r requirements.txt
```

---

## usage

```bash
# run main program (generates figures + prints q-tables)
python main.py

# or open the notebook
jupyter notebook notebooks/01-main.ipynb
```

### generated outputs

| file | description |
|------|-------------|
| `docs/02-results/fig01_mc_policies.png` | learned q-value heatmaps + policy arrows for all 3 control methods |
| `docs/02-results/fig02_mc_prediction.png` | off-policy state-value prediction v(s) |
| `docs/02-results/fig03_mc_learning_curves.png` | episode reward curves for all methods |

---

## project structure

```
cisc-856-assignment-2/
├── main.py                    # main execution & plotting
├── README.md
├── requirements.txt           # dependencies
├── .env                       # environment config
├── .env.example               # config template
├── .gitignore
├── src/
│   ├── config/
│   │   └── settings.py        # gridworld parameters
│   └── utils/
│       └── helpers.py         # mc algorithms + env dynamics + plotting
├── notebooks/
│   └── 01-main.ipynb          # notebook version of main.py
└── docs/
    ├── 01-assignment/          # assignment spec (pdf + md)
    ├── 02-results/             # generated figures
    └── 03-deliverables/
        └── 01-report.md        # full report
```

---

## results

### learned q-table (exploring starts)

```
 state      up   right    left    down   best
     0   0.466   0.616   0.476   0.342  right
     1   0.606   0.755   0.476   0.617  right
     2   0.743   0.900   0.617   0.617  right
  GOAL   0.000   0.000   0.000   0.000     up
     4   0.485   0.353   0.351   0.234     up
     6   0.754  -0.245   0.605   0.461     up
  FIRE   0.900  -0.245   0.607   0.345     up
 START   0.360   0.328   0.235   0.235     up
     9   0.325   0.482   0.226   0.319  right
    10   0.616   0.323   0.333   0.475     up
    11  -0.245   0.334   0.483   0.329   left
```

the learned policy routes from start (8) → up → 4 → up → 0 → right → 1 → right → 2 → right → goal (3)

### sample output

```
=== final state values: off-policy prediction ===
  V(    0) = 0.5770
  V(    1) = 0.7247
  V(    2) = 0.9000
  V( GOAL) = 0.0000
  V(    4) = 0.4450
  V(    6) = 0.7287
  V( FIRE) = 0.9000
  V(START) = 0.3098
  V(    9) = 0.4478
  V(   10) = 0.5776
  V(   11) = 0.4188
```

all four algorithms converge successfully. on-policy methods converge faster (~8k episodes) while off-policy methods require ~16k episodes due to importance sampling variance

### figures

see `docs/02-results/` for the generated visualizations

---

## report

full report at [`docs/03-deliverables/01-report.md`](docs/03-deliverables/01-report.md).

covers methodology, results, exploration strategy breakdown, and challenges encountered.

---

## author

elsayed elmandoh - nlp engineer - [linktree](https://linktr.ee/elsayedelmandoh)

