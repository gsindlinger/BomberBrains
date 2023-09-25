from collections import deque, namedtuple
from typing import List
import events as e
from .DQN import ReplayMemory, extract_tensors
from .callbacks import state_to_features, SAVED_FILE
from .customized_rewards import stayed_in_blast, prevent_long_wait, escape_bomb, get_position_feature, invalid_action_for_state, closer_to_coin, get_closest_coin_feature, get_crates_field, running_into_explosion, dropped_corner_bomb, WAITED_TOO_LONG, ESCAPING_BOMB_FULLY, ESCAPING_BOMB_PARTLY, RUNNING_INTO_BOMB, RUNNING_INTO_BOMB_PARTLY, INVALID_MOVE_MANUAL, VALID_MOVE_MANUAL, DIRECTIONS, COIN_CLOSER, COIN_FURTHER, NOT_LEAVING_BLAST, RUNNING_INTO_EXPLOSION, DROPPED_CORNER_BOMB
import numpy as np
import matplotlib.pyplot as plt

import time
import shutil

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_AS_NUMBER = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
POSITION_HISTORY_SIZE = 6

# Hyper parameters
BATCH_SIZE = 64
GAMMA = 0.999
TARGET_UPDATE = 10
MEMORY_SIZE = 100000

###############################
########## SETUP ##############
###############################

def setup_training(self):
    self.memory = ReplayMemory(MEMORY_SIZE)
    self.eval_rewards_counter = 10
    self.eval_rewards_dic  = {10: 0}
    self.positions = deque(maxlen=POSITION_HISTORY_SIZE)
    self.waited_times = 0

##############################################
########## game_events_occurred ##############
##############################################

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.old_state = new_game_state
    action = ACTIONS.index(self_action)

    field = old_game_state['field']
    _, _, bomb_possible, (self_x, self_y) = old_game_state['self']  # Agent's current position
    others = old_game_state['others']
    bombs = old_game_state['bombs']
    explosion_map = old_game_state['explosion_map']
    coins = old_game_state['coins']

    position_feature = get_position_feature(bombs, explosion_map, field, others, self_x, self_y)
    closest_item_feature = get_closest_coin_feature(self_x, self_y, field, coins)

    if closest_item_feature == DIRECTIONS.index("NO_ITEM"):
        temp_field_remove_crates, crates = get_crates_field(field)
        closest_item_feature = get_closest_coin_feature(self_x, self_y, temp_field_remove_crates, crates)

    # append position to position history
    self.positions.append(new_game_state['self'][-1])

    # add custom events
    prevent_long_wait(self, events)
    escape_bomb(self, old_game_state, new_game_state, events)
    closer_to_coin(self, old_game_state, closest_item_feature, new_game_state, _, events, action)
    invalid_action_for_state(self, position_feature, action, events)
    stayed_in_blast(old_game_state, new_game_state, events)
    # running_into_explosion(old_game_state, events, self_action)
    # dropped_corner_bomb(old_game_state, events)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # add reward to reward list
    self.eval_rewards_dic[self.eval_rewards_counter] += reward_from_events(self, events)
    
    # store Transition in memory
    self.memory.push(Transition(state_to_features(old_game_state), ACTION_AS_NUMBER[self_action], state_to_features(new_game_state), reward_from_events(self, events)))
    
    # train only if there are enough Transitions
    if self.memory.can_provide_sample(BATCH_SIZE):
        train_agent(self)

######################################
########## end_of_round ##############
######################################

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'end_of_round Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # add reward to reward list
    self.eval_rewards_dic[self.eval_rewards_counter] += reward_from_events(self, events)

    # store Transition in memory
    self.memory.push(Transition(state_to_features(self.old_state), ACTION_AS_NUMBER[last_action], state_to_features(last_game_state), reward_from_events(self, events)))
    
    # update target model every TARGET_UPDATE round
    if last_game_state['round'] % TARGET_UPDATE == 0:
        self.target_net.set_weights(self.policy_net.get_weights())
        self.eval_rewards_counter = last_game_state['round']
        self.eval_rewards_dic[self.eval_rewards_counter] = 0
    
    # plotting
    if last_game_state['round'] == 2010:
        self.eval_rewards_dic.popitem()
        
        # Extract keys and values
        keys = list(self.eval_rewards_dic.keys())
        values = list(self.eval_rewards_dic.values())

        # Create a line graph
        plt.plot(keys, values, marker='o', linestyle='-', label='Data Points')

        # Fit a polynomial trend line (adjust the degree as needed)
        degree = 2
        coefficients = np.polyfit(keys, values, degree)
        trend_values = np.polyval(coefficients, keys)

        # Plot the trend line
        plt.plot(keys, trend_values, linestyle='--', label=f'Trend (Degree {degree})', color='red')

        # Add labels and title
        plt.xlabel('Round')
        plt.ylabel('Reward')
        plt.title('Reward by Round with Trend Line')

        # Customize x-axis tick labels
        plt.xticks(keys)

        # Add a legend
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.locator_params(axis='both', nbins=7)
        #plt.show()
        localtime = time.localtime()
        time_string = time.strftime("%Y%m%d-%H:%M:%S", localtime)
        plt.savefig(time_string+".png")
 
        # save logs
        source = "./logs/dqn_agent.log"
        destination = time_string+"_agent.log"
        shutil.copyfile(source, destination)
        source = "../../logs/game.log"
        destination = time_string+"_game.log"
        shutil.copyfile(source, destination)
 
        # save model incrementally
        shutil.copytree(SAVED_FILE, time_string+"_model")        

    # overwrite model with updated model
    self.policy_net.save(SAVED_FILE)

############################################
########## reward_from_events ##############
############################################

def reward_from_events(self, events: List[str]) -> int:

    game_rewards = {
        e.INVALID_ACTION: -10,
        e.CRATE_DESTROYED: 10,
        e.COIN_COLLECTED: 15,
        e.KILLED_OPPONENT: 20,
        e.KILLED_SELF: -30,
        e.GOT_KILLED: -27,
        e.SURVIVED_ROUND: 2,
        WAITED_TOO_LONG: -15,
        NOT_LEAVING_BLAST: -30,
        ESCAPING_BOMB_FULLY: 20,
        ESCAPING_BOMB_PARTLY: 15,
        RUNNING_INTO_BOMB: -25,
        RUNNING_INTO_BOMB_PARTLY: -17,
        COIN_CLOSER: 9,
        COIN_FURTHER: -12,
        INVALID_MOVE_MANUAL: -20,
        VALID_MOVE_MANUAL: 2,
        # RUNNING_INTO_EXPLOSION: -32,
        # DROPPED_CORNER_BOMB: -22
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

#####################################
########## train_agent ##############
#####################################

def train_agent(self):
    batch_transitions = self.memory.sample(BATCH_SIZE)

    #states and next_states shape = (256, 1, 196)
    states, actions, next_states, rewards = extract_tensors(batch_transitions)

    # for keras network, we have to cut out one dimension which is on axis 1
    states = np.squeeze(states, axis=1)
    next_states = np.squeeze(next_states, axis=1)

    next_q_values_policy_net = self.policy_net.predict(next_states)
    next_q_values_target_net = self.target_net.predict(next_states)
    target_q_values = self.policy_net.predict(states)

    index = np.arange(BATCH_SIZE)
    target_q_values[index, actions] = rewards + GAMMA * next_q_values_target_net[index, np.argmax(next_q_values_policy_net, axis=1)]
    
    # we use keras .fit() method to train our dqn-agent
    self.policy_net.fit(states, target_q_values, verbose=0)