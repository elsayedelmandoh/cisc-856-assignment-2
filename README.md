# cisc 856 assignment 2 — monte carlo methods in rl

elsayed elmandouh - 20596379 - reinforcement learning - queen's university

[![github](https://img.shields.io/badge/GitHub-sentiment__sleuth-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/elsayedelmandoh/cisc-856-assignment-1)
[![Twitter](https://img.shields.io/badge/X-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/aangpy)
[![linkedin](https://img.shields.io/badge/elsayed-linkedin-0077b5?style=for-the-badge&logo=linkedin&logocolor=white)](https://www.linkedin.com/in/elsayed-elmandoh-b5849a1b8/)

## table of contents

- [overview](#overview)
- [setup](#setup)
- [usage](#usage)
- [project structure](#project-structure)
- [results](#results)
- [author](#author)

---

## overview

this project implements a 5x5 gridworld markov decision process (mdp) solved using three approaches:

- **linear system solver**: closed-form solution for deterministic case
- **value iteration**: iterative bellman optimality algorithm
- **policy iteration**: alternating evaluation and improvement

### problem parameters

| parameter | value |
|-----------|-------|
| grid size | 5x5 |
| goal | cell 14 (+5.0) |
| dangers | cells 2, 18, 21 (-5.0) |
| blocked | cells 6, 7, 11, 12 |
| discount factors | 0.75, 0.95 |
| noise levels | 0.0, 0.2 |

---

## setup

```bash
# clone repository
git clone https://github.com/elsayedelmandoh/cisc-856-assignment-1
cd cisc-856-assignment-1

# create environment
conda create -n cisc856 python=3.12 -y
conda activate cisc856

# install dependencies
pip install -r requirements.txt
```

---

## usage

```bash
# run main program
python main.py
```

---

## project structure

```
cisc-856-assignment-1/
├── main.py                # main execution
├── README.md              
├── requirements.txt       # dependencies
├── .env.example           # configuration
├── .gitignore             
├── src/
│   ├── utils/
│   │   ├── gridworld.py   # mdp class
│   │   ├── helpers.py     # vi/pi algorithms
│   │   └── actions.py     # action definitions
│   └── config/
│       └── settings.py    # config loader
├── notebooks/
│   └── 01-assignment-1-code-snippet.ipynb
└── docs/
    ├── 01-assignment/     # assignment
    ├── 02-results/        # visualizations
    └── 03-deliverables/
        └── 01-report.md
```

---

## results

### convergence summary

| algorithm | configuration | iterations |
|-----------|---------------|------------|
| linear solver | γ=0.95, noise=0.0 | 1 |
| linear solver | γ=0.75, noise=0.0 | 1 |
| value iteration | γ=0.95, noise=0.0 | 12 |
| value iteration | γ=0.75, noise=0.0 | 12 |
| value iteration | γ=0.95, noise=0.2 | 15 |
| value iteration | γ=0.75, noise=0.2 | 7 |
| policy iteration | γ=0.95, noise=0.0 | 10 |

### value iteration output (γ=0.95, noise=0.0)

```
  3.15   2.99  -5.00   4.51   4.75
  3.32   ----   ----   4.75   5.00
  3.49   ----   ----   5.00   goal
  3.68   3.87   4.07  -5.00   5.00
  3.49  -5.00   4.29   4.51   4.75
```

## author

elsayed elmandoh - nlp engineer

* connect on linkedin and x - [linktree](https://linktr.ee/elsayedelmandoh)