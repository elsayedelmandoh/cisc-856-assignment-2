""" 
utils/helpers.py: helper functions for gridworld environment.
"""

from collections import defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config.settings import (
    ROWS, COLS, N_STATES, N_ACTIONS,
    GOAL, DANGER, WALL, START,
    GOAL_REWARD, DANGER_REWARD, STEP_REWARD, MAX_STEPS,
    ACTIONS, ARROW
)

def state_to_rc(s):
    return s // COLS, s % COLS
 
def rc_to_state(r, c):
    return r * COLS + c
 
def step(state, action):
    """
    transition function.
    returns (next_state, reward, done)
    - wall / out-of-bounds: bounce back, step reward only
    - goal: +1 + step reward, terminal
    - danger: -1 + step reward, NOT terminal
    - else: step reward only
    """
    r, c = state_to_rc(state)
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
 
    # boundary or wall check
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c           # bounce: stay
    next_state = rc_to_state(nr, nc)
    if next_state == WALL:
        next_state = state      # wall: bounce back
 
    reward = STEP_REWARD
    done   = False
 
    if next_state == GOAL:
        reward += GOAL_REWARD
        done = True
    elif next_state == DANGER:
        reward += DANGER_REWARD
        # danger is NOT terminal per spec
 
    return next_state, reward, done
 
 
def generate_episode(policy_fn, start_override=None):
    """
    rolls out one episode using policy_fn(state) -> action.
    policy_fn receives the current state and returns an action.
    returns list of (state, action, reward) tuples.
    """
    state = start_override if start_override is not None else START
    episode = []
    for _ in range(MAX_STEPS):
        action = policy_fn(state)
        next_state, reward, done = step(state, action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode
 
 
def compute_returns(episode, gamma=1.0):
    """
    backward pass: G_t = r_t + gamma * G_{t+1}
    returns list of G_t for each timestep t
    """
    G = 0.0
    returns = []
    for _, _, r in reversed(episode):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns
 

#  i. on-policy mc with exploring starts
def mc_exploring_starts(n_episodes=5000, gamma=0.95, first_visit=True):
    """
    exploring starts: each episode begins at a random (state, action) pair.
    guarantees all (s,a) pairs get explored regardless of policy.
 
    policy: deterministic greedy w.r.t. Q
    update: first-visit averaging of returns
    """
    Q       = np.zeros((N_STATES, N_ACTIONS))
    returns = defaultdict(list)   # (s,a) -> list of G values
    policy  = np.zeros(N_STATES, dtype=int)  # greedy policy
 
    # non-terminal, non-wall states for random starts
    valid_starts = [s for s in range(N_STATES)
                    if s != GOAL and s != WALL]
 
    reward_per_ep = []
 
    for ep in range(n_episodes):
        # exploring starts: random (state, action) for first step
        start_state  = random.choice(valid_starts)
        start_action = random.randint(0, N_ACTIONS - 1)
 
        # run first step manually, then follow policy
        next_state, reward, done = step(start_state, start_action)
        episode = [(start_state, start_action, reward)]
 
        if not done:
            state = next_state
            for _ in range(MAX_STEPS - 1):
                action = policy[state]
                ns, r, done = step(state, action)
                episode.append((state, action, r))
                state = ns
                if done:
                    break
 
        reward_per_ep.append(sum(r for _, _, r in episode))
 
        Gs     = compute_returns(episode, gamma)
        visited = set()
 
        for t, ((s, a, _), G) in enumerate(zip(episode, Gs)):
            if first_visit and (s, a) in visited:
                continue
            visited.add((s, a))
            returns[(s, a)].append(G)
            Q[s, a] = np.mean(returns[(s, a)])
 
        # policy improvement: greedy
        for s in range(N_STATES):
            if s != GOAL and s != WALL:
                policy[s] = np.argmax(Q[s])
 
    return Q, policy, reward_per_ep


#  ii. on-policy mc control without exploring starts (epsilon-greedy)
def mc_epsilon_greedy(n_episodes=5000, gamma=0.95, epsilon=0.1, first_visit=True):
    """
    no exploring starts: epsilon-greedy policy handles exploration.
    same policy both generates episodes and gets updated.
 
    epsilon-greedy: prob epsilon -> random, prob 1-epsilon -> greedy
    """
    Q       = np.zeros((N_STATES, N_ACTIONS))
    returns = defaultdict(list)
    policy  = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS  # init uniform
 
    reward_per_ep = []
 
    def eps_greedy(state):
        if random.random() < epsilon:
            return random.randint(0, N_ACTIONS - 1)
        return np.argmax(Q[state])
 
    for ep in range(n_episodes):
        episode = generate_episode(eps_greedy)
        reward_per_ep.append(sum(r for _, _, r in episode))
 
        Gs      = compute_returns(episode, gamma)
        visited = set()
 
        for (s, a, _), G in zip(episode, Gs):
            if first_visit and (s, a) in visited:
                continue
            visited.add((s, a))
            returns[(s, a)].append(G)
            Q[s, a] = np.mean(returns[(s, a)])
 
            # epsilon-greedy policy update
            best_a = np.argmax(Q[s])
            for act in range(N_ACTIONS):
                if act == best_a:
                    policy[s, act] = 1 - epsilon + epsilon / N_ACTIONS
                else:
                    policy[s, act] = epsilon / N_ACTIONS
 
    # extract deterministic greedy policy
    greedy_policy = np.argmax(Q, axis=1)
    return Q, greedy_policy, policy, reward_per_ep
 

#  iii. off-policy mc prediction (weighted importance sampling)
def mc_offpolicy_prediction(target_policy, n_episodes=10000, gamma=0.95):
    """
    off-policy prediction: estimate V^pi using episodes from behavior policy b.
    behavior b: uniform random (ensures full coverage)
    target pi: fixed deterministic policy passed in
 
    uses weighted importance sampling (lower variance than ordinary IS):
      V(s) = sum(W_i * G_i) / sum(W_i)
    incremental form:
      C(s) += W
      V(s) += (W / C(s)) * (G - V(s))
    """
    V = np.zeros(N_STATES)
    C = np.zeros(N_STATES)   # cumulative IS weights per state
 
    # behavior policy: uniform random
    def behavior(state):
        return random.randint(0, N_ACTIONS - 1)
 
    b_prob = 1.0 / N_ACTIONS  # prob of each action under uniform b
 
    for ep in range(n_episodes):
        episode = generate_episode(behavior)
 
        G = 0.0
        W = 1.0  # importance weight, updated backward
 
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
 
            # only update states where target policy matches (or use W)
            pi_a = target_policy[s]   # deterministic target: prob 1 for pi_a, 0 else
            if a != pi_a:
                # target policy would never take this action -> W goes to 0
                # for prediction we still accumulate but W=0 after this point
                # so we break: all earlier steps get W=0 too (nothing to update)
                break
 
            # pi(a|s) = 1 (deterministic), b(a|s) = 1/4
            W = W * (1.0 / b_prob)   # = W * 4 (for each matching step)
 
            C[s] += W
            if C[s] > 0:
                V[s] += (W / C[s]) * (G - V[s])
 
    return V

#  iv. off-policy mc control (weighted IS, greedy target)
def mc_offpolicy_control(n_episodes=10000, gamma=0.95):
    """
    off-policy control: behavior policy b is soft (epsilon-greedy or uniform),
    target policy pi is deterministic greedy updated every episode.
 
    incremental weighted IS update for Q:
      C(s,a) += W
      Q(s,a) += (W / C(s,a)) * (G - Q(s,a))
      pi(s) = argmax_a Q(s,a)
 
    key: if action taken != target policy action, W -> 0 for earlier steps
    so we break out of the backward loop early.
    """
    Q      = np.zeros((N_STATES, N_ACTIONS))
    C      = np.zeros((N_STATES, N_ACTIONS))
    # random init to avoid all-zero argmax bias toward action 0
    target = np.random.randint(0, N_ACTIONS, size=N_STATES)
 
    b_prob = 1.0 / N_ACTIONS  # uniform behavior
 
    def behavior(state):
        return random.randint(0, N_ACTIONS - 1)
 
    reward_per_ep = []
 
    for ep in range(n_episodes):
        episode = generate_episode(behavior)
        reward_per_ep.append(sum(r for _, _, r in episode))
 
        G = 0.0
        W = 1.0
 
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
 
            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
 
            # update target policy
            target[s] = np.argmax(Q[s])
 
            # if behavior didn't match target, IS weight from here back = 0
            if a != target[s]:
                break
 
            # update importance weight
            # pi(a|s)=1 (deterministic greedy), b(a|s)=1/4
            W = W * (1.0 / b_prob)
 
    return Q, target, reward_per_ep
 

#  visualization helpers
GOAL_COLOR = '#27ae60'
DANGER_COLOR = '#e74c3c'
WALL_COLOR = '#1a1a2e'


def draw_grid(ax, Q, policy, title):
    """heatmap of max Q values + policy arrows. row 0 = top (matches assignment grid)."""
    V = np.max(Q, axis=1).reshape(ROWS, COLS)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(V, cmap=cmap, vmin=-1.5, vmax=1.5, aspect='equal', origin='upper')

    for s in range(N_STATES):
        r, c = state_to_rc(s)
        if s == WALL:
            ax.add_patch(mpatches.FancyBboxPatch(
                (c-0.5, r-0.5), 1, 1, boxstyle="square,pad=0",
                facecolor=WALL_COLOR, zorder=2))
            ax.text(c, r, "WALL", ha='center', va='center',
                    color='white', fontsize=7, zorder=3)
        elif s == GOAL:
            ax.add_patch(mpatches.FancyBboxPatch(
                (c-0.5, r-0.5), 1, 1, boxstyle="square,pad=0",
                facecolor=GOAL_COLOR, zorder=2))
            ax.text(c, r, f"GOAL\n+1", ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
        elif s == DANGER:
            ax.add_patch(mpatches.FancyBboxPatch(
                (c-0.5, r-0.5), 1, 1, boxstyle="square,pad=0",
                facecolor=DANGER_COLOR, zorder=2))
            ax.text(c, r, f"FIRE\n-1", ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
        elif s == START:
            a = policy[s]
            dx, dy = ARROW[a]
            ax.annotate("", xy=(c+dx, r+dy), xytext=(c, r),
                        arrowprops=dict(arrowstyle="->", color='navy', lw=2), zorder=3)
            ax.text(c-0.35, r-0.35, "S", fontsize=7, color='navy', zorder=3)
            ax.text(c, r+0.3, f"{V[r,c]:.2f}", ha='center', va='center',
                    fontsize=7, color='white', zorder=3)
        else:
            a = policy[s]
            dx, dy = ARROW[a]
            ax.annotate("", xy=(c+dx, r+dy), xytext=(c, r),
                        arrowprops=dict(arrowstyle="->", color='navy', lw=2), zorder=3)
            ax.text(c, r+0.3, f"{V[r,c]:.2f}", ha='center', va='center',
                    fontsize=7, color='black', zorder=3)

    ax.set_xlim(-0.5, COLS-0.5)
    ax.set_ylim(ROWS-0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_title(title, fontsize=10, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.03)


def draw_V(ax, V, title):
    """heatmap for state value function (prediction). row 0 = top."""
    Vg = V.reshape(ROWS, COLS)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(Vg, cmap=cmap, vmin=-1.5, vmax=1.5, aspect='equal', origin='upper')

    for s in range(N_STATES):
        r, c = state_to_rc(s)
        if s == WALL:
            ax.add_patch(mpatches.FancyBboxPatch(
                (c-0.5, r-0.5), 1, 1, boxstyle="square,pad=0",
                facecolor=WALL_COLOR, zorder=2))
            ax.text(c, r, "WALL", ha='center', va='center',
                    color='white', fontsize=7, zorder=3)
        elif s == GOAL:
            ax.add_patch(mpatches.FancyBboxPatch(
                (c-0.5, r-0.5), 1, 1, boxstyle="square,pad=0",
                facecolor=GOAL_COLOR, zorder=2))
            ax.text(c, r, f"GOAL\n{V[s]:.2f}", ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
        elif s == DANGER:
            ax.add_patch(mpatches.FancyBboxPatch(
                (c-0.5, r-0.5), 1, 1, boxstyle="square,pad=0",
                facecolor=DANGER_COLOR, zorder=2))
            ax.text(c, r, f"FIRE\n{V[s]:.2f}", ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')
        else:
            ax.text(c, r, f"{V[s]:.2f}", ha='center', va='center',
                    fontsize=9, color='black')

    ax.set_xlim(-0.5, COLS-0.5)
    ax.set_ylim(ROWS-0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_title(title, fontsize=10, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.03)
    
 
def smooth(x, w=200):
    return np.convolve(x, np.ones(w)/w, mode='valid')
 

