import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# Hyper parameters
LR = 0.001
FEATURE_SHAPE = (343,)
N_ACTIONS = 6
N_NEURONS = 64

####################################
######### Network creation #########
####################################

def create_network():

    # Create a sequential model
    model = tf.keras.Sequential()
    # Add layers to the model
    model.add(layers.Dense(N_NEURONS, input_shape=FEATURE_SHAPE, kernel_initializer="random_normal", activation="relu"))
    model.add(layers.Dense(N_ACTIONS, activation='softmax'))  # Output layer

    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=LR))

    return model


#####################################
########## ReplayMemory #############
#####################################

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    # Add Transition to memory array
    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.push_count % self.capacity] = transition
        self.push_count += 1

    # Get a sample of Transitions
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    # Check if there are enough Transitions
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


####################################
####### EpsilonGreedyStrategy ######
####################################

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    # at the beginning, we have 100% exploration, step by step the % gets lower and we start to exploit 
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


####################################
####### Extract Tensors ############
####################################

def extract_tensors(transitions):
    batch = Transition(*zip(*transitions))
    states = batch.state
    actions = batch.action
    next_states = batch.next_state
    rewards = batch.reward
    return(np.array(states, dtype=np.float32), np.array(actions, dtype=np.integer), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32))