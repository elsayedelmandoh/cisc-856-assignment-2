"""
cisc 856 assignment 2: apply monte carlo methods to a gridworld environment
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.helpers import (
    mc_exploring_starts,
    mc_epsilon_greedy,
    mc_offpolicy_prediction,
    mc_offpolicy_control,
    draw_grid,
    draw_V,
    smooth,
)
from src.config.settings import (
    N_STATES,
    GOAL, DANGER, WALL, START,
    ACTION_NAMES, OUTPUT_DIR, N_EP, GAMMA,
)


def main(): 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print("running all 4 mc algorithms...")
 
    # ── i. exploring starts ──────────────────────────
    print("[1/4] on-policy mc with exploring starts...")
    Q_es, policy_es, rews_es = mc_exploring_starts(N_EP, GAMMA)
 
    # ── ii. epsilon-greedy ───────────────────────────
    print("[2/4] on-policy mc epsilon-greedy (eps=0.1)...")
    Q_eg, policy_eg, soft_eg, rews_eg = mc_epsilon_greedy(N_EP, GAMMA, epsilon=0.1)
 
    # ── iii. off-policy prediction ───────────────────
    # use the policy from ES as the target to evaluate
    print("[3/4] off-policy mc prediction (target=ES policy)...")
    V_pred = mc_offpolicy_prediction(policy_es, n_episodes=N_EP*2, gamma=GAMMA)
 
    # ── iv. off-policy control ───────────────────────
    print("[4/4] off-policy mc control...")
    Q_opc, policy_opc, rews_opc = mc_offpolicy_control(N_EP*2, GAMMA)
 
    # ─── figure 1: Q/policy grids side by side ───────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("monte carlo control: learned policies & value estimates", fontsize=13)
 
    draw_grid(axes[0], Q_es,  policy_es,  "i. on-policy ES\n(exploring starts)")
    draw_grid(axes[1], Q_eg,  policy_eg,  "ii. on-policy eps-greedy\n(eps=0.1)")
    draw_grid(axes[2], Q_opc, policy_opc, "iv. off-policy control\n(uniform behavior)")
 
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig01_mc_policies.png", dpi=150, bbox_inches='tight')
    plt.close()
 
    # ─── figure 2: off-policy prediction V ───────────
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_V(ax, V_pred, "iii. off-policy prediction V(s)\n(weighted IS, target=ES policy)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig02_mc_prediction.png", dpi=150, bbox_inches='tight')
    plt.close()
 
    # ─── figure 3: learning curves ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("episode reward curves (smoothed window=200)", fontsize=12)
 
    for ax, rews, label, color in zip(
        axes,
        [rews_es, rews_eg, rews_opc],
        ["on-policy ES", "on-policy eps-greedy", "off-policy control"],
        ['steelblue', 'coral', 'seagreen']
    ):
        sm = smooth(rews)
        ax.plot(sm, color=color, lw=1.5)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig03_mc_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
 
    # ─── print final Q values ─────────────────────────
    print("\n=== final Q-table: on-policy ES ===")
    print(f"{'state':>6} {'up':>7} {'right':>7} {'left':>7} {'down':>7} {'best':>6}")
    for s in range(N_STATES):
        if s == WALL: continue
        label = "GOAL" if s==GOAL else ("FIRE" if s==DANGER else ("START" if s==START else str(s)))
        best  = ACTION_NAMES[np.argmax(Q_es[s])]
        print(f"{label:>6} {Q_es[s,0]:>7.3f} {Q_es[s,1]:>7.3f} {Q_es[s,2]:>7.3f} {Q_es[s,3]:>7.3f} {best:>6}")
 
    print("\n=== final state values: off-policy prediction ===")
    for s in range(N_STATES):
        if s == WALL: continue
        label = "GOAL" if s==GOAL else ("FIRE" if s==DANGER else ("START" if s==START else str(s)))
        print(f"  V({label:>5}) = {V_pred[s]:.4f}")
 
    print(f"\ndone. outputs saved to {OUTPUT_DIR}/")
 

if __name__ == "__main__":
    main()
    