import random, time
from Puzzle import State, move, num_solved_sides, num_pieces_correct_side, shuffle, n_move_state, one_move_state


class AgentClass:
    def __init__(self, QValues=None, cube=None):
        self.visited = []
        self.visit_count = {}
        self.revisits = 0
        self.QV = QValues if QValues is not None else {}
        self.R = {}
        self.start_state = cube if cube is not None else n_move_state(n=20) 
        print(self.start_state)
        self.curr_state = self.start_state
        self.prev_state = None
        self.second_last_action = None
        self.actions = self.start_state.actions
        self.one_away = []
        self.two_away = []
        self.three_away = []
        self.four_away = []
        self.five_away = []
        self.six_away = []
        self.last_action = None
        self.move = {"front": 0, "back": 0, "left": 0, "right": 0, "top": 0, "bottom": 0}

    def register_patternsforCube(self):
        s = State()
        for action in self.actions:
            s_ = move(s, action)
            self.one_away.append(s_)
            for action_ in self.actions:
                self.QV[(s_.__hash__(), action_)] = -10 if action_ != action else 10
        for s in self.one_away:
            for action in self.actions:
                s_ = move(s, action)
                self.two_away.append(s_)
                for action_ in self.actions:
                    self.QV[(s_.__hash__(), action_)] = -6 if action_ != action else 6
        for s in self.two_away:
            for action in self.actions:
                s_ = move(s, action)
                self.three_away.append(s_)
                for action_ in self.actions:
                    self.QV[(s_.__hash__(), action_)] = -5 if action_ != action else 5
        
        for s in self.three_away:
            for action in self.actions:
                s_ = move(s, action)
                self.four_away.append(s_)
                for action_ in self.actions:
                    self.QV[(s_.__hash__(), action_)] = -4 if action_ != action else 4

        for s in self.four_away:
            for action in self.actions:
                s_ = move(s, action)
                self.five_away.append(s_)
                for action_ in self.actions:
                    self.QV[(s_.__hash__(), action_)] = -3 if action_ != action else 3
        
        for s in self.five_away:
            for action in self.actions:
                s_ = move(s, action)
                self.six_away.append(s_)
                for action_ in self.actions:
                    self.QV[(s_.__hash__(), action_)] = -1 if action_ != action else 1
            
    # explore
    def QLearn(self, discount=0.99, episodes=15, epsilon=0.1):
        LEARNING_RATE = 0.6
        self.curr_state = self.curr_state
        for i in range(episodes):
            print("=====EPISODE "+str(i)+"=====")
            print("====CURR STATE========")
            print("======================")
            saved_rewards = self.curr_state.__hash__() in self.R.keys()
            if not saved_rewards:
                self.R[self.curr_state.__hash__()] = []
            if not self.curr_state.__hash__ in self.visit_count:
                self.visit_count[self.curr_state.__hash__()] = 1
            else:
                self.visit_count[self.curr_state.__hash__()] += 1
            vc = self.visit_count[self.curr_state.__hash__()]
            for action in self.actions:
                if not (self.curr_state.__hash__(), action) in self.QV.keys():
                    self.QV[(self.curr_state.__hash__(), action)] = 0
                else:
                    self.revisits += 1
                    break
                if not saved_rewards:
                    self.R[self.curr_state.__hash__()].append(self.reward(self.curr_state, action))
            if 100 in self.R[self.curr_state.__hash__()]:
                print("REACHED GOAL, END QLEARN ITERATION")
                return
            follow_policy = random.uniform(0,1.0)
            print("Random value generated is " + str(follow_policy))
            if follow_policy > epsilon:
                print("FOLLOWING POLICY")
                for action in self.actions:
                    print("q value for action " + action + " from curr state is " + str(self.QV[(self.curr_state.__hash__(), action)]))
                best_action = None
                best_QV = -100000000
                for action in self.actions:
                    if self.QV[(self.curr_state.__hash__(), action)] > best_QV and action != self.last_action and action != self.second_last_action:
                        best_action = action
                        best_QV = self.QV[(self.curr_state.__hash__(), action)]
                if best_QV == 0:
                    best_action = random.choice(self.actions)
                    while best_action == self.last_action:
                        best_action = random.choice(self.actions)
                print("Actions chosen = " + best_action)
                self.move[best_action] = self.move[best_action] + 1
                for action in self.actions:
                    curr_QV = self.QV[(self.curr_state.__hash__(), action)]
                    reward = self.reward(self.curr_state, action)
                    max_reward = self.max_reward(self.curr_state, action)
                    self.QV[(self.curr_state.__hash__(), action)] = curr_QV + LEARNING_RATE*(reward +\
                                                         (discount**vc)*max_reward - curr_QV)
                print("New q value for " + best_action + " action is " + str(self.QV[(self.curr_state.__hash__(), best_action)]))
                self.curr_state.move(best_action)
                self.curr_state = self.curr_state.copy()
                if self.curr_state.isGoalState():
                    print("Reached goal state while in Q-learning epsiode " + str(i))
                    time.sleep(2)
                    return
                self.second_last_action = self.last_action  
                self.last_action = best_action
            else:
                action = random.choice(self.actions)
                self.move[action] = self.move[action] + 1
                while action == self.last_action or action == self.second_last_action:
                    action = random.choice(self.actions)
                reward = 0
                for action in self.actions:
                    curr_QV = self.QV[(self.curr_state.__hash__(), action)]
                    reward = self.reward(self.curr_state, action)
                    max_reward = self.max_reward(self.curr_state, action)
                    self.QV[(self.curr_state.__hash__(), action)] = curr_QV + LEARNING_RATE*(reward +\
                                                         (discount**vc)*max_reward - curr_QV)
                self.curr_state.move(action)
                self.curr_state = self.curr_state.copy()
                self.second_last_action = self.last_action  
                self.last_action = action
                if self.curr_state.isGoalState():
                    print("Reached goal state while in Q-learning epsiode " + str(i))
                    time.sleep(2)
                    return

    def Play(self):
        self.second_last_action = None
        self.last_action = None
        self.curr_state = self.start_state
        print(self.curr_state)
        while True:
            best_action = None
            best_QV = -100000000
            if not (self.curr_state.__hash__(), self.actions[0]) in self.QV.keys():
                best_action = random.choice(self.actions)
                while best_action == self.second_last_action or best_action == self.last_action:
                    best_action = random.choice(self.actions)
                for action in self.actions:
                    self.QV[(self.curr_state.__hash__(), action)] = 0
                best_QV = 0
            else:
                for action in self.actions:
                    if self.QV[(self.curr_state.__hash__(), action)] > best_QV \
                    and (action != self.last_action and action != self.second_last_action):
                        best_action = action
                        best_QV = self.QV[(self.curr_state.__hash__(), action)]
            print("Actions chosen = " + best_action)
            print("Last action = " + (self.last_action if self.last_action is not None  else "None"))
            print("Q value is " + str(self.QV[(self.curr_state.__hash__(), best_action)]))
            self.curr_state.move(best_action)
            self.second_last_action = self.last_action
            self.last_action = best_action
            print(self.curr_state)
            if self.curr_state.isGoalState():
                print("AGENT REACHED A GOAL STATE!!!")
                time.sleep(5)
                return


    def print_(self):
        print("=============")
        x = 0
        y = 0
        for key in self.QV.keys():
            if self.QV[key] != 0:
                x += 1
            else:
                y += 1
        print("NO.OF q values in dictionary is " + str(x + y))
        print("NO.OF q values with zero value is " + str(y))
        print("NO.OF q value with non zero value is " + str(x))
        print("NO.OF re-visited states = " + str(self.revisits))
        print(self.move)
            
    def reward(self, state, action):
        next_state = move(state, action)
        if next_state.isGoalState():
            print(state)
            print(next_state)
            print("REWARD IS GOAL")
            return 100
        reward = -0.1
        solved_sides = 2 * (num_solved_sides(next_state) < num_solved_sides(state))
        solved_pieces = 0.5 * (num_pieces_correct_side(next_state) < num_pieces_correct_side(state))
        if (next_state.__hash__(), action) in self.QV.keys():
            reward -= 0.2
        reward -= solved_sides
        reward -= solved_pieces
        return reward

    def max_reward(self, state, action):
        new_state = move(state, action)
        if not new_state in self.R.keys():
            self.R[new_state] = []
            for action in self.actions:
                self.R[new_state].append(self.reward(new_state, action))
        return max(self.R[new_state])

agent = AgentClass()
print("REGISTERING PATTERN DATABASE, THIS WILL TAKE A LITTLE WHILE")
agent.register_patternsforCube()
print(agent.QV)
Epsilons = [i/ 50 for i in range(50)]
Epsilons.reverse()
for i in range(2):
    for j, e in enumerate(Epsilons):
        print("======= ROUND " + str(j) + "=========")
        agent.QLearn(epsilon=e)
print("There are " + str(len(agent.QV)) + " keys in Q Table")
agent.Play()
agent.print_()