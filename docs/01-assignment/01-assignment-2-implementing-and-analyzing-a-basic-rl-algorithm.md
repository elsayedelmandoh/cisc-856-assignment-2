# Assignment 2: Monte Carlo Methods in RL    
Due: Thursday May 21, before midnight EST 
 
## Objective: 
Apply Monte Carlo methods to a Gridworld environment 


## Environment: 
The environment is a Gridworld where the agent spawns in a starting position and needs to find a goal state on a discrete stationary grid. The agent has four different actions he can perform, these are: [Up, Right, Left, Down]. The following image shows the environment: 

The starting position is state 8 in the bottom left-hand corner. For each step, the agent can move one state in one direction. If the agent is at the edge of the environment, it goes back to its original position. 

State number 5 is a wall and cannot be entered. the agent moves back to its position. The goal state is the green state where the agent receives a reward of +1. The red state is the danger state. If the agent enters the danger state it will receive a negative reward of -1. For each step the agent takes it receives a reward of -0.1. One episode ends if the agent reaches the goal state or after 30 interactions. 

> Now the grid with GOAL=3, DANGER=7, WALL=5, START=8:
0   1   2   3(G)
4   5(W) 6   7(D)
8(S) 9   10  11


## Tasks: 
Implement the following algorithms to solve the above Gridworld. You can choose either first-visit or every-visit method. 

i. On-policy MC with exploring starts (10 points) 
ii. On-policy MC control without exploring starts (10 points) 
iii. Off-policy MC prediction (10 points) 
iv. Off-policy MC control (10 points) 
 

## Submission: 
 
- Code: Well-documented code implementing the tasks. (40 points) 
- Report: A detailed report discussing the methodology, findings, and insights from applying different Monte Carlo methods to this Gridworld example. Include discussions on the effectiveness of the chosen exploration strategy and any challenges encountered. (60 points) 
 
