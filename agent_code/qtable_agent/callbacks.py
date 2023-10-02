import os
import pickle
import random

from .utils import *
from collections import Counter

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.6
EPSILON = 0.6
EPSILON_MIN = 0.2
EPSILON_DECAY = 0.0005

EXPLORATION_MODE = ["EPS", "UCB", "BOLTZ"] # Choose from 'EPS', 'UCB', 'BOLTZ'
USE_REWARDS_FOR_TRAINING = True


def setup(self):

    index = 0
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.q_file_name = f"./data/{EXPLORATION_MODE[index]}/q_table.npy"
    self.q_file_counts_name = f"./data/{EXPLORATION_MODE[index]}/q_table_counts" \
                              f"{'_test' if not USE_REWARDS_FOR_TRAINING else ''}.npy"
    self.score_file = f"./data/{EXPLORATION_MODE[index]}/scores{'_test' if not USE_REWARDS_FOR_TRAINING else ''}.p"
    self.state_list = deque(maxlen=4)
    self.action_list = deque(maxlen=4)
    self.exploration_mode = EXPLORATION_MODE[index]
    self.eps = EPSILON
    self.eps_min = EPSILON_MIN
    self.eps_decay = EPSILON_DECAY

    if os.path.isfile(self.score_file) and USE_REWARDS_FOR_TRAINING:
        with open(self.score_file, 'rb') as file:
            self.past_scores = pickle.load(file)
    else:
        self.past_scores = {
            'Sum_Rewards': [],
            'Mean_Rewards': [],
            'Reward_Dict': []
        }

    if self.train and USE_REWARDS_FOR_TRAINING:
        if os.path.isfile(self.q_file_name) and os.path.isfile(self.q_file_name):
            # if training and file is present continue training old file
            self.Q_table = np.load(self.q_file_name)
            self.Q_table_counts = np.load(self.q_file_name)
        else:
            self.logger.info("Setting up model from scratch.")
            # if no file is present initialize Q_matrix
            shape = tuple(FEATURE_SHAPE + [len(REDUCED_ACTIONS)])
            self.Q_table = np.zeros(shape)
            self.Q_table_counts = np.zeros(shape)
    else:
        # only load file no training
        self.logger.info("Loading model from saved state.")
        self.Q_table = np.load(self.q_file_name)
        self.Q_table_counts = np.load(self.q_file_name)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if game_state['step'] == 1:
        self.state_list.clear()
        self.action_list.clear()

    state = game_state_to_features(self, game_state)
    symmetrical_states = get_symmetrical_states(state)

    if self.train and USE_REWARDS_FOR_TRAINING:
        if self.exploration_mode == "EPS":
            if random.random() < self.eps:
                self.logger.debug("Choosing action purely at random.")
                # 80%: walk in any direction. 10% wait. 10% bomb.
                return_action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
            else:
                return_action = query_max_action(self, state, symmetrical_states)

        elif self.exploration_mode == "UCB":
            c = 1.0
            total_time_steps = np.sum(self.Q_table_counts)
            ucb_scores = []

            for i, temp_state in enumerate([state] + symmetrical_states + [state, state]):
                if i < 4:
                    action_value = self.Q_table[temp_state][0]
                    action_count = self.Q_table_counts[temp_state][0]
                else:
                    action_value = self.Q_table[temp_state][i-3]
                    action_count = self.Q_table_counts[temp_state][i-3]

                exploration_bonus = c * np.sqrt(
                    np.log(total_time_steps) / (action_count + 1e-6))  # Add a small constant to avoid division by zero
                ucb_score = action_value + exploration_bonus
                ucb_scores.append(ucb_score)


            # Choose the action with the highest UCB score
            return_action = ACTIONS[np.argmax(ucb_scores)]

        elif self.exploration_mode == "BOLTZ":
            temperature = 1
            boltzmann_values = np.zeros(len(ACTIONS))
            for i, temp_state in enumerate([state] + symmetrical_states + [state, state]):
                if i < 4:
                    boltzmann_values[i] = self.Q_table[temp_state][0]
                    boltzmann_values[i] = np.exp(boltzmann_values[i] / temperature)
                else:
                    boltzmann_values[i] = self.Q_table[temp_state][i-3]
                    boltzmann_values[i] = np.exp(boltzmann_values[i] / temperature)

            boltzmann_values /= np.sum(boltzmann_values)
            return_action = ACTIONS[np.random.choice(len(boltzmann_values), p=boltzmann_values)]

    else:
        return_action = query_max_action(self, state, symmetrical_states)

    base_return_action, shift = convert_to_base_action(ACTIONS.index(return_action))
    initial_state = get_symmetrical_state_single(state, shift)
    self.Q_table_counts[initial_state][base_return_action] += 1
    return return_action

def query_max_action(self, state, symmetrical_states):
    max_values = []
    state_values = self.Q_table[state]

    action = np.argmax(self.Q_table[state])
    max_value = self.Q_table[state][action]
    if any(self.Q_table[state] == 0):
        self.logger.debug(f"At least one state / action not trained yet: {state}")
    max_action = convert_symmetric_action(action, 0)
    max_values.append((max_value, ACTIONS[max_action]))
    for i, temp_state in enumerate(symmetrical_states):

        action = np.argmax(self.Q_table[temp_state])
        if any(self.Q_table[temp_state] == 0):
            self.logger.debug(f"At least one state / action not trained yet: {temp_state}")
        max_value_temp = self.Q_table[temp_state][action]
        max_values.append((max_value_temp, ACTIONS[convert_symmetric_action(action, -(i + 1))]))
        # only consider symmetric states when their maximum action is a movement
        if max_value_temp > max_value and action == 0:
            max_action = convert_symmetric_action(action, -(i + 1))
            max_value = max_value_temp

    return_action = ACTIONS[max_action]

    # prevent model to repeat action
    if not (self.train and USE_REWARDS_FOR_TRAINING) and len(self.state_list) == 4 and \
            ((self.state_list[0] == self.state_list[2] and self.state_list[1] == self.state_list[3] and
             self.action_list[0] == self.action_list[2] and self.action_list[1] == self.action_list[3]) or \
             (self.state_list[0] == self.state_list[1] == self.state_list[2] == self.state_list[3] and
             self.action_list[0] == self.action_list[2] == self.action_list[1] == self.action_list[3])):
        sorted_max_values = sorted(max_values, key=lambda x: x[0], reverse=True)

        for i, value in enumerate(sorted_max_values):
            if sorted_max_values[i][1] != return_action:
                max_value = sorted_max_values[i][0]
                return_action = sorted_max_values[i][1]
                self.logger.debug(f"Repeated action: Action List: {self.action_list}, State List: {self.state_list}")
                break
    #else:
    #    sorted_max_values_differences = sorted_max_values[0][0] - [value for value, _ in sorted_max_values]
    #    sorted_max_values_differences = [difference for difference in sorted_max_values_differences if difference < 3]

    #    if(all(value > 0 for value, _ in sorted_max_values[:len(sorted_max_values_differences)])):
    #        temperatures = [10 ** (0.1 + value / 30 * (np.log(10) - 0.1)) for value in [value for value,
    #        #        _ in sorted_max_values]]
    #        sorted_max_values_scaled = [value ** (1 / temperature) for value, temperature
    #                                         in zip([value for value, _ in sorted_max_values], temperatures[:len(
            #                                         sorted_max_values_differences)])]
    #        total_scaled_value = sum(sorted_max_values_scaled)

            # Normalize the scaled values
    #        normalized_values = [scaled_value / total_scaled_value for scaled_value in sorted_max_values_scaled]
    #        selected_index = random.choices(range(len(sorted_max_values_differences)), weights=normalized_values)[0]
    #        temp_return_action = sorted_max_values[selected_index][1]
    #        if(temp_return_action != return_action):
    #            print("Return Action changed")

    #        return_action = temp_return_action

    # We could find out that, whenever at most three symmetrical states supposed to throw a bomb it seemed to be
    # a good choice (since there is a should drop bomb flag / feature). So treating this as special.
    max_actions = [action for _, action in max_values]
    counter_helper = Counter(max_actions)
    for cnt_action, count in counter_helper.items():
        if count > 2 and cnt_action == 'BOMB':
            return_action = cnt_action
            break


    self.logger.debug(f"Current state: {state}, Symmetric states: {symmetrical_states}")
    self.logger.debug(f"State values for actions: {state_values}")
    self.logger.debug(f"Max Actions for Symmetric States: {max_values}")
    self.logger.debug(f"Querying model for action. Choosing '{return_action}'")
    self.state_list.appendleft(state)
    self.action_list.appendleft(return_action)

    return return_action


