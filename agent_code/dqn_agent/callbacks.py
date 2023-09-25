import os
import random
import numpy as np
from tensorflow import keras
from .DQN import create_network, EpsilonGreedyStrategy
import settings

# Hyper parameters
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.0001

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
SAVED_FILE = "dqn_agent.model"
VIEW_SIZE: int = 7

###############################
########## SETUP ##############
###############################

def setup(self):
    print("setup")
    self.strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)
    self.exploration_rate_step = 0
    if os.path.exists(SAVED_FILE):
        self.logger.info("Loading existing model")
        self.policy_net = keras.models.load_model(SAVED_FILE)
        if self.train:
            self.target_net = create_network()
            # Copy the weights and biases from the policy_net to the target_net
            self.target_net.set_weights(self.policy_net.get_weights())
    elif self.train:
        self.logger.info("New model")
        self.policy_net = create_network()
        self.target_net = create_network()
        # Copy the weights and biases from the policy_net to the target_net
        self.target_net.set_weights(self.policy_net.get_weights())
    else:
        self.logger.info("No trained model available")
        exit()

#############################
########## ACT ##############
#############################

def act(self, game_state: dict) -> str:
    self.dqn_input_features = state_to_features(game_state)
    exploration_rate = self.strategy.get_exploration_rate(self.exploration_rate_step)
    
    #Exploration vs. Exploitation
    if self.train and exploration_rate > random.random():
        self.exploration_rate_step += 1
        action_index = random.randrange(len(ACTIONS)) #Explore
        self.logger.info(f"Action RANDOM: {ACTIONS[action_index]}")
        return ACTIONS[action_index]
    else:
        self.exploration_rate_step += 1
        actions = self.policy_net.predict(self.dqn_input_features) #Exploit
        action_index_array = np.argmax(actions, axis=1)
        action_index = action_index_array[0]
        self.logger.info(f"Action via Model: {ACTIONS[action_index]}")
        return ACTIONS[action_index]

###########################################
########## state_to_features ##############
###########################################

def state_to_features(game_state: dict) -> np.ndarray:

    # create the view of the self_player (x_range and y-range) --> whole game matrix is way too large!
    _, _, _, (player_x, player_y) = game_state["self"]

    if (player_x - VIEW_SIZE // 2) < 0:
        player_x = VIEW_SIZE // 2
    if (player_y - VIEW_SIZE // 2) < 0:
        player_y = VIEW_SIZE // 2

    if (player_x + VIEW_SIZE // 2) >= settings.COLS:
        player_x = settings.COLS - VIEW_SIZE // 2 - 1
    if (player_y + VIEW_SIZE // 2) >= settings.ROWS:
        player_y = settings.ROWS - VIEW_SIZE // 2 - 1

    x_range = range(player_x - VIEW_SIZE // 2, player_x + VIEW_SIZE // 2 + 1)
    y_range = range(player_y - VIEW_SIZE // 2, player_y + VIEW_SIZE // 2 + 1)

    # for each observation category create a channel
    channels = []

    # 0.5 = True, -0.5 = False in the following 7x7 Matrices

    # Fill the 7x7 view of the player with the crates/obstacles tiles
    field_view_section = game_state["field"][np.ix_(y_range, x_range)]

    crates_features = np.where(field_view_section == 1, 0.5, -0.5)
    wall_features = np.where(field_view_section == -1, 0.5, -0.5)

    # Iterate through the coin positions and set "0.5" in the submatrix if the coin is within the view, the rest 
    # of the tiles have the value -0.5
    coins = game_state["coins"]
    coins_features = np.full((VIEW_SIZE, VIEW_SIZE), -0.5)

    for coin in coins:
        x, y = coin
        if x in x_range and y in y_range:
            coins_features[y - y_range.start, x - x_range.start] = 0.5

    # Iterate through the bomb positions and set the counter value in the submatrix if the bomb is within the view
    bombs = game_state["bombs"]
    bombs_features = np.full((VIEW_SIZE, VIEW_SIZE), -0.5)

    for bomb in bombs:
        (x, y), t = bomb
        if x in x_range and y in y_range:
            bombs_features[y - y_range.start, x - x_range.start] = (settings.BOMB_TIMER - t) / settings.BOMB_TIMER
    
    # Iterate through the player positions and set the desired value for other players to 0.5 if the specific player is within the view
    other_players_features = np.full((VIEW_SIZE, VIEW_SIZE), -0.5)
    others = game_state['others']

    for (_, _, _, (x, y)) in others:
        if x in x_range and y in y_range:
            other_players_features[y - y_range.start, x - x_range.start] = 0.5

    # Set 0.5 for the tile, where the own player is positioned 
    self_player_features = np.full((VIEW_SIZE, VIEW_SIZE), -0.5)
    self = game_state["self"]
    (_, _, _, (x, y)) = self
    if x in x_range and y in y_range:
        self_player_features[y - y_range.start, x - x_range.start] = 0.5

    # Now we have to find the free tiles, this depends on our above observations
    matrices = [crates_features, wall_features, bombs_features, other_players_features]

    # Initialize the new matrix with zeros
    freeTiles_features = np.zeros_like(crates_features)

    # Iterate through the rows and columns of the matrices
    for row in range(freeTiles_features.shape[0]):
        for col in range(freeTiles_features.shape[1]):
            # Initialize a flag to check if any matrix has a non-zero value
            any_positive_value = False
            
            # Iterate through the list of matrices
            for matrix in matrices:
                # Check if any matrix has a 0.5 value at the current tile
                if matrix[row, col] == 0.5:
                    any_positive_value = True
                    break  # No need to check other matrices once we find a 0.5 value
            
            # Set the corresponding value in the freeTiles_features matrix based on the flag
            if any_positive_value:
                freeTiles_features[row, col] = -0.5
            else:
                freeTiles_features[row, col] = 0.5

    # features shape = (7, 7)
    channels.append(freeTiles_features)
    channels.append(crates_features)
    channels.append(wall_features)
    channels.append(coins_features)
    channels.append(bombs_features)
    channels.append(other_players_features)
    channels.append(self_player_features)

    # concatenate the channels as a feature tensor (they must have the same shape)
    # stacked_channels shape = (7, 7, 7)
    stacked_channels = np.stack(channels).astype(float)

    # return stacked_channels as a vector
    # stacked_channels shape = (343,)
    stacked_channels = stacked_channels.reshape(-1)
    
    #add batch dimension for keras network
    # stacked_channels shape = (1, 343)
    stacked_channels = np.expand_dims(stacked_channels, axis=0)

    return stacked_channels


def more_features():
    pass