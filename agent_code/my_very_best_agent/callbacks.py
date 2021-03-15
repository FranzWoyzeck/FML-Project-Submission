import os
import pickle
import random

import numpy as np
from itertools import permutations

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

CALLED = False
FEATURES = None
TMP = None
DIRECTION = np.unique(np.array(list(permutations([-1, -1, 0, 0, 1, 1], 2))), axis=0)
DISCRETE_OS_SIZE = [len(DIRECTION)]


def get_closest_coin_idx(state):
    vec_neu = np.abs(np.array(state.get('coins') - np.array(state.get('self')[3])))
    return np.argmin(vec_neu.T[0] + vec_neu.T[1])


def get_feature_space(state: dict):
    environment = np.array(state.get('field'))
    corner_a = environment[0:6, 0:6]
    corner_b = environment[11:17, 0:6]
    corner_c = environment[11:17, 11:17]
    corner_d = environment[0:6, 11:17]
    features = np.empty([31, 3, 3])

    for i in range(4):
        features[i] = corner_a[0:3, 3 - i:6 - i]
        features[i + 1 * 4] = corner_b[3:6, 3 - i:6 - i]
        features[i + 2 * 4] = corner_c[3:6, 3 - i:6 - i]
        features[i + 3 * 4] = corner_d[0:3, 3 - i:6 - i]

    for i in range(3):
        features[16 + i] = corner_a[3 - i:6 - i, 0:3]
        features[16 + i + 1 * 3] = corner_b[3 - i:6 - i, 0:3]
        features[16 + i + 2 * 3] = corner_c[3 - i:6 - i, 3:6]
        features[16 + i + 3 * 3] = corner_d[3 - i:6 - i, 3:6]

    features[27 + 1] = corner_a[2:5, 1:4]
    features[27 + 2] = corner_a[2:5, 2:5]
    features[27 + 3] = corner_a[3:6, 2:5]

    return np.unique(features, axis=0)


def get_relative_min_coin_direction(state: dict):
    coins_relative = np.array(state.get('coins'))-np.array(state.get('self')[3])
    min_index = np.argmin(np.abs(coins_relative.T[0])+np.abs(coins_relative.T[1]))
    return coins_relative[min_index]


def get_environment_data(state: dict):
    environment = np.array(state.get('field'))
    index = np.array(state.get('self')[3])
    kernel = np.empty([3, 3])
    for i in range(-1, 2):
        for j in range(-1, 2):
            tmp_index = np.array([index[0]+i, index[1]+j])
            kernel[i+1, j+1] = environment[tmp_index[0], tmp_index[1]]
    return kernel


def get_current_state(state: dict):
    global CALLED
    global FEATURES
    if not CALLED:
        FEATURES = get_feature_space(state)
        CALLED = True
    index = 0
    position_area = get_environment_data(state)
    for i, area in enumerate(FEATURES):
        if np.array_equal(area, position_area):
            index = i
            break
    return index


def check_for_index(array, element):
    for i, arr in enumerate(array):
        if np.array_equal(arr, element):
            return i
    return 0


def get_direction_index(state: dict):
    coin_distance = get_relative_min_coin_direction(state)
    # if player is on the coin
    if not np.any(coin_distance):
        return check_for_index(DIRECTION, np.array([0, 0]))
    else:
        # check if coin is not reachable by just one coordinate
        if np.all(coin_distance):
            # check if coin is on the left-lower and right-upper corner
            if coin_distance[0] * coin_distance[1] < 0:
                # check if coin is left-lower
                if coin_distance[0] < 0:
                    return check_for_index(DIRECTION, np.array([-1, 1]))
                else:
                    return check_for_index(DIRECTION, np.array([1, -1]))
            else:
                # check if coin is left-upper
                if coin_distance[0] < 0:
                    return check_for_index(DIRECTION, np.array([-1, -1]))
                else:
                    return check_for_index(DIRECTION, np.array([1, 1]))
        else:
            # check if coin left
            if coin_distance[0] < 0:
                return check_for_index(DIRECTION, np.array([-1, 0]))
            # check if coin right
            elif coin_distance[0] > 0:
                return check_for_index(DIRECTION, np.array([1, 0]))
            # it can only be up or down
            else:
                # check if up
                if coin_distance[1] < 0:
                    return check_for_index(DIRECTION, np.array([0, -1]))
                else:
                    return check_for_index(DIRECTION, np.array([0, 1]))


def get_discrete_state(state):
    get_direction_index(state)
    global CALLED
    global FEATURES
    if not CALLED:
        FEATURES = get_feature_space(state)
        CALLED = True
    return tuple([get_direction_index(state), get_current_state(state)])


def setup(self):
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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.epsilon = 1
        self.start_epsilon_decaying = 1
        self.end_epsilon_decaying = 100000
        if not os.path.isfile("my-saved-model.pt"):
            self.epsilon_decay_value = self.epsilon / (self.end_epsilon_decaying - self.start_epsilon_decaying)
            self.logger.info("Setting up model from scratch.")
            q_table = np.full((DISCRETE_OS_SIZE + [15] + [len(ACTIONS)]), -0.5)
            print(q_table)
            self.model = q_table
        else:
            self.epsilon_decay_value = self.epsilon / (self.end_epsilon_decaying - self.start_epsilon_decaying)
            self.logger.info("Loading model from saved state.")
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
            print(self.model)
    else:
        self.epsilon = 0.9
        self.start_epsilon_decaying = 1
        self.end_epsilon_decaying = 100000
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        print(self.model)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train and random.uniform(0, 1) < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])
    self.logger.debug("Querying model for action.")
    # print(ACTIONS[np.argmax(self.model[get_discrete_state(game_state)])])
    # print(self.model[get_discrete_state(game_state)])
    return ACTIONS[np.argmax(self.model[get_discrete_state(game_state)])]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)