from collections import namedtuple, deque, defaultdict

import pickle
from typing import List
from collections import Counter

import events as e
import numpy as np

from .callbacks import *
from .utils import *

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
POSITION_HISTORY_SIZE = 6  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Custom events
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
UNESCAPABLE_BOMB = "UNESCAPABLE_BOMB"
BOMB_DESTROYS_CRATE = "BOMB_DESTROYS_CRATE"
WAITED_TOO_LONG = "WAITED_TOO_LONG"
WAITED_TOO_LONG_JUMPING = "WAITED_TOO_LONG_JUMPING"
IN_BLAST_RADIUS = "IN_BLAST_RADIUS"
COIN_CLOSER = "COIN_CLOSER"
COIN_FURTHER = "COIN_FURTHER"
ESCAPING_BOMB_FULLY = "ESCAPING_BOMB_FULLY"
ESCAPING_BOMB_PARTLY = "ESCAPING_BOMB_PARTLY"
RUNNING_INTO_BOMB = "RUNNING_INTO_BOMB"
RUNNING_INTO_BOMB_PARTLY = "RUNNING_INTO_BOMB_PARTLY"
INVALID_MOVE_MANUAL = "INVALID_MOVE_MANUAL"
VALID_MOVE_MANUAL = "VALID_MOVE_MANUAL"
WAITED_INTO_BOMB = "WAITED_INTO_BOMB"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.waited_times = 0
    self.rewards = defaultdict(int)
    self.positions = deque(maxlen=POSITION_HISTORY_SIZE)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    if game_state_to_features(self, old_game_state) is not None:
        action = ACTIONS.index(self_action)
        old_state = game_state_to_features(self, old_game_state)
        new_state = game_state_to_features(self, new_game_state)

        #append position to position history
        self.positions.append(new_game_state['self'][-1])

        add_custom_events_to_rewards(action, events, new_game_state, new_state, old_game_state, old_state, self)

        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

        # get reward
        reward, reward_list = reward_from_events(self, events)
        for reward_temp, value in reward_list.items():
            self.rewards[reward_temp] += 1

        if USE_REWARDS_FOR_TRAINING:
            # update q matrix with one item for all symmetric states (choose ACTION 'UP' as reference)
            temp_action, shift_index = convert_to_base_action(action)
            old_state_temp = get_symmetrical_state_single(old_state, shift_index)

            old_value = self.Q_table[old_state_temp][temp_action]
            next_max = np.max(self.Q_table[new_state])
            for i, temp_state in enumerate(get_symmetrical_states(new_state)):
                next_val = np.max(self.Q_table[temp_state])
                if next_val > next_max:
                    next_max = next_val

            new_value = (
                    (1 - LEARNING_RATE) * old_value
                    + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
            )
            self.Q_table[old_state_temp][temp_action] = new_value

            for i, transition in enumerate(self.transitions):
                transition_state = transition.state
                if transition_state is None:
                    continue
                transition_action = transition.action
                if transition_action is None:
                    continue
                transition_action = ACTIONS.index(transition_action)

                transition_action, shift_index = convert_to_base_action(action)
                transition_state = get_symmetrical_state_single(transition_state, shift_index)
                transition_action_value = self.Q_table[transition_state][transition_action]
                transition_alpha = LEARNING_RATE ** (i + 2)
                new_value = (
                        (1 - transition_alpha) * transition_action_value
                        + transition_alpha * (reward + DISCOUNT_FACTOR * next_max)
                )
                self.Q_table[transition_state][transition_action] = new_value

        self.transitions.appendleft(Transition(game_state_to_features(self, old_game_state), self_action,
                                           game_state_to_features(self, new_game_state), reward))

def add_custom_events_to_rewards(action, events, new_game_state, new_state, old_game_state, old_state, self,
                                 final_game=False):
    # add custom events
    prevent_long_wait(self, events)
    invalid_action_for_state(self, old_state, action, events)

    if not final_game:
        escape_bomb(self, old_game_state, new_game_state, events)
        closer_to_coin(self, old_game_state, old_state, new_game_state, new_state, events, action)
        dropped_escabable_bomb(self, old_state, new_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    action = ACTIONS.index(last_action)
    if game_state_to_features(self, last_game_state) is not None:
        old_state = game_state_to_features(self, last_game_state)

        add_custom_events_to_rewards(action, events, None, None, last_game_state, old_state, self, True)

        # get reward
        reward, reward_list = reward_from_events(self, events)
        for temp_reward, value in reward_list.items():
            self.rewards[temp_reward] += 1

        if USE_REWARDS_FOR_TRAINING:
            # update q matrix with one item for all symmetric states (choose ACTION 'UP' as reference)
            temp_action, shift_index = convert_to_base_action(action)
            old_state_temp = get_symmetrical_state_single(old_state, shift_index)
            old_value = self.Q_table[old_state_temp][temp_action]
            new_value = (
                    (1 - LEARNING_RATE) * old_value
                    + LEARNING_RATE * reward
            )
            self.Q_table[old_state_temp][temp_action] = new_value

            for i, transition in enumerate(self.transitions):
                transition_state = transition.state
                if transition_state is None:
                    continue
                transition_action = transition.action
                if transition_action is None:
                    continue
                transition_action = ACTIONS.index(transition_action)

                transition_action, shift_index = convert_to_base_action(action)
                transition_state = get_symmetrical_state_single(transition_state, shift_index)
                transition_action_value = self.Q_table[transition_state][transition_action]
                transition_alpha = LEARNING_RATE if 'GOT_KILLED' in events else LEARNING_RATE ** (i + 2)
                new_value = (
                        (1 - transition_alpha) * transition_action_value
                        + transition_alpha * reward
                )
                self.Q_table[transition_state][transition_action] = new_value

        self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
        self.transitions.appendleft(Transition(game_state_to_features(self, last_game_state), last_action, None,
                                            reward_from_events(self, events)))

        if self.eps != None and self.eps > self.eps_min:
            self.eps -= self.eps_decay
            if self.eps < self.eps_min:
                eps = self.eps_min

    mean_reward = sum(self.rewards.values()) / len(self.rewards)
    reward_sum = sum(self.rewards.values())
    self.past_scores['Sum_Rewards'].append(reward_sum)
    self.past_scores['Mean_Rewards'].append(mean_reward)
    self.past_scores['Reward_Dict'].append(self.rewards)

    with open(self.score_file, 'wb') as file:
        pickle.dump(self.past_scores, file)
    np.save(self.q_file_name, self.Q_table)
    np.save(self.q_file_counts_name, self.Q_table_counts)

    self.positions.clear()
    self.transitions.clear()
    self.rewards = defaultdict(int)

def invalid_action_for_state(self, old_state, action, events):
    if e.INVALID_ACTION not in events:
        correct_move = action < 4 and old_state[action] in [1,2,3] # check whether a movement was made into a free field
        missed_move = action > 3 and any(state == 0 for state in old_state) and old_state[4] != 1
        if correct_move or missed_move:
            events.append(INVALID_MOVE_MANUAL)
    elif action < 4 and old_state[action] == POSITION_MAPPING.index('FREE'):
        events.append(VALID_MOVE_MANUAL)


def prevent_long_wait(self, events):
    """
    Method trying tro prevent the agent to wait for longer times. Adds the WAITED_TOO_LONG event, if the agent
    waited more than wait_limit steps.
    Args:
        self:
        events:

    Returns:

    """
    wait_limit = 3
    if e.WAITED in events:
        self.waited_times += 1
    else:
        self.waited_times = 0
    if self.waited_times > wait_limit:
        events.append(WAITED_TOO_LONG)

    # also if distance didn't change too much punish agent (so prevent from jumping always into the back again always)
    count_positions = Counter(self.positions)
    count_greater_than_2 = 0

    for _, count in count_positions.items():
        if count >= 3:
            count_greater_than_2 += 1

    if count_greater_than_2 >= 2:
        events.append(WAITED_TOO_LONG_JUMPING)
def escape_bomb(self, old_game_state, new_game_state, events):
    old_bomb_blasts_coord = [coords for coords, _, _, _ in get_bomb_blasts(old_game_state['bombs'], old_game_state[
        'field'])]
    new_bomb_blasts_coord = [coords for coords, _, _, _ in get_bomb_blasts(new_game_state['bombs'], new_game_state[
        'field'])]

    old_bombs = [coords for coords, _ in old_game_state['bombs']]
    new_bombs = [coords for coords, _ in new_game_state['bombs']]

    player_coord_new = new_game_state['self'][-1]
    player_coord_old = old_game_state['self'][-1]

    if(
            player_coord_old in old_bomb_blasts_coord and
            player_coord_new not in new_bomb_blasts_coord
    ):
        events.append(ESCAPING_BOMB_FULLY)
    elif(
            player_coord_old not in old_bomb_blasts_coord and
            player_coord_new in new_bomb_blasts_coord and
            e.BOMB_DROPPED not in events
    ):
        events.append(RUNNING_INTO_BOMB)
    elif(player_coord_old in old_bomb_blasts_coord and player_coord_new in new_bomb_blasts_coord):
        closest_bomb_old, _ = get_closest_item_bfs(player_coord_old[0],
                                                   player_coord_old[1],
                                                   old_bombs, old_game_state['field'])
        closest_bomb_new, _ = get_closest_item_bfs(player_coord_new[0],
                                                player_coord_new[1],
                                                new_bombs, new_game_state['field'])
        if not closest_bomb_old == None and not closest_bomb_new == None:
            if get_manhatten_distance(closest_bomb_new, player_coord_new) > \
                    get_manhatten_distance(closest_bomb_old, player_coord_old):
                events.append(ESCAPING_BOMB_PARTLY)
            elif get_manhatten_distance(closest_bomb_new, player_coord_new) < \
                    get_manhatten_distance(closest_bomb_old, player_coord_old) and \
                    not e.BOMB_DROPPED in events:
                events.append(RUNNING_INTO_BOMB_PARTLY)


def closer_to_coin(self, old_game_state, old_state_features, new_game_state, new_state_features, events, action):
    player_coord_new = new_game_state['self'][-1]
    player_coord_old = old_game_state['self'][-1]

    old_closest_coin_coord, _ = get_closest_item_bfs(player_coord_old[0], player_coord_old[1], old_game_state['coins'],
                                                  old_game_state['field'])
    new_closest_coin_coord, _ = get_closest_item_bfs(player_coord_new[0], player_coord_new[1], new_game_state['coins'],
                                                  new_game_state['field'])

    # Check whether the way to the closest coin was blocked (only for up, left, right, down). If so, don't punish
    # movement which steps away the player.
    coin_was_blocked = False
    if (old_state_features[4] == DIRECTIONS.index('UP') and action == ACTIONS.index('UP')) or \
            (old_state_features[4] == DIRECTIONS.index('RIGHT') and action == ACTIONS.index('RIGHT')) or \
            (old_state_features[4] == DIRECTIONS.index('LEFT') and action == ACTIONS.index('LEFT')) or \
            (old_state_features[4] == DIRECTIONS.index('DOWN') and action == ACTIONS.index('DOWN')):
        coin_was_blocked = True

    if new_closest_coin_coord is None:
        modified_field_old, crates_old = get_crates_field(old_game_state['field'])
        modified_field_new, crates_new = get_crates_field(new_game_state['field'])

        old_closest_coin_coord, _ = get_closest_item_bfs(player_coord_old[0], player_coord_old[1],
                                                         crates_old,
                                                         modified_field_old)
        new_closest_coin_coord, _ = get_closest_item_bfs(player_coord_new[0], player_coord_new[1],
                                                         crates_new,
                                                         modified_field_new)

    if old_closest_coin_coord is None or new_closest_coin_coord is None:
        return

    if(get_manhatten_distance(player_coord_old, old_closest_coin_coord)
       < get_manhatten_distance(player_coord_new, new_closest_coin_coord)) and \
            not e.COIN_COLLECTED in events and \
            not coin_was_blocked:
        events.append(COIN_FURTHER)
    elif(get_manhatten_distance(player_coord_old, old_closest_coin_coord)
         > get_manhatten_distance(player_coord_new, new_closest_coin_coord)):
        events.append(COIN_CLOSER)

def dropped_escabable_bomb(self, old_game_state, new_game_state, events):
    if(e.BOMB_DROPPED in events):
        if(old_game_state[4] == 1):
            events.append(ESCAPABLE_BOMB)
        else:
            events.append(UNESCAPABLE_BOMB)

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent gets based on custom and predefined events.

    :param events: List of events that occurred during the game step.
    :return: Total reward based on events.
    """
    game_rewards = {
        e.INVALID_ACTION: -20,
        e.CRATE_DESTROYED: 10,
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 30,
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -30,
        WAITED_TOO_LONG: -15,
        WAITED_TOO_LONG_JUMPING: -12,
        ESCAPING_BOMB_FULLY: 15,
        ESCAPING_BOMB_PARTLY: 11,
        RUNNING_INTO_BOMB: -16,
        RUNNING_INTO_BOMB_PARTLY: -12,
        COIN_CLOSER: 9,
        COIN_FURTHER: -10,
        ESCAPABLE_BOMB: 8,
        INVALID_MOVE_MANUAL: -7,
        VALID_MOVE_MANUAL: 5,
    }

    reward_sum = 0
    reward_list = defaultdict(int)
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            reward_list[event] += game_rewards[event]


    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum, reward_list

