# CSCI6660_AI
# Implemented By Manasa Sukavasi, Indhu Allala, Sai Chandan Regati 

There are 2 files associated with this project, Agent.py and Puzzle.py agent_mod.py: An Agent class is created with multiple method implementations. It keeps track of visited states and re-visited states. It has variables like cube, QV which are the cube itself(solved/unsolved - input) and the Q-values respective to the states accordingly.

QV are initially not provided such that, it takes some random set of Qvalues initially then, based on the state it gives Qvalues to the states.

The current state is given as the start state initially and the start state can be provided in two different ways. The user can either provide the cube state values as an array of arrays or as a dictionary in which each array represent a side and each side contains 12 values (4 per row, 4 columns per side and 6 sides per cube)

The previous and the second previous are stored accordingly as the present state depends on them.

The given initial pattern of states are registered in the database and they are assigned with the QValues using the hash function.

We have used a Powerful Reinforcement learning algorithm known as Feature-based Q-learning is used for the implementation. To check the quality of the cube nearing the solved state, we have used the utilization of a pattern database.

The Qlearn method is defined by initiating with the learning rate(alpha), discounting factor (to reduce the reward), epsilon with the given number of episodes. The Q-learning algorithm uses epsilon-greedy approach which selects the next state based on the max reward.

This works on Epsilon-greedy Policy, The policy selects some random value using the random uniform library between 0 and 1 and if that value is greater than the selected value epsilon, thenm it follows the policy and updates Q-Value for current state and action chosen based on the current policy, by taking original Q-value, and adding alpha times the reward value of the new state plus the discounted max_reward of executing every possible action on the new state, minus the original Q-Value. If that value is less than epsilon value, then it updatse Q-Value for current state and randomly chosen action, by taking original Q-value, and adding alpha times the reward value of the new state plus the discounted max_reward of executing every possible action on the new state, minus the original Q-Value.

Based on the explained policy, it performs the action in given possible episodes and the rewards, and checks if it reaches the goal state in any of the given episode.

The reward associated with the goal in the next state is 100 and if not it is discounted by 0.1 by each step and it is also discounted by checking the number of solved sides and number of correct sides and if the the if the number of correct sides in the previous state is greater than the next state then, it is discounted again and the max reward for each next state is calculated and the state with the max reward is chosen as the next move and so on.

This process is continued in the given number of episodes and the program quits after the agent is reached the goal state which is the convergence point.

All the supporting methods to shuffle the cube if the initial input is not provided by the user, checking if the cube reached it's goal state or not, All the states and their moves and their actions are appropriately defined in the puzzle_mod.py file.

The variable n is given to change the number of rotations to shuffle the cube provided initially. and it's trained and assumed that it can do 180 degrees rotation. This helps in handling the branching factor of the state space of the cube.
