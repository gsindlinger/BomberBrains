## BomberBrains

## Reinforcement Learning in Bomberman Game

### Johannes Sindlinger, Daniel Knorr 

This is a repository for submission of the final project of lecture Machine Learning Essentials at Heidelberg University. The goal was to develop a reinforcement learning algorithm for Bomberman: Bomberman is a classic game in which an agent can perform moves, strategically use explosives, and intelligently navigate through obstacles. The primary goal is to clear boxes, collect coins, and most importantly, eliminate enemy agents without destroying themselves.
We developed two approaches that were trained based on Q-learning. 
- [Q-Table](agent_code/qtable_agent)
- [Q-Network](agent_code/dqn_agent)

The code of the agents, results and analyses are in the respective subfolders. The framework for this project can be 
found at [https://github.com/ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl).

### Getting Started
To run a demo of our best performing agent (Q-Table) clone the repository and run ```python main.py play --my-agent qtable_agent```
