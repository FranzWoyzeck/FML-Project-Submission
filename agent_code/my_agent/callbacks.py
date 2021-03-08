import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_q_table(self):
        # only create new q_table, if the is no according file
        if not os.path.isfile("my-saved-q-table.pt"):
            self.logger.info("Setting up q_table from scratch.")
            self.q_table: dict = dict()

        else:
            self.logger.info("Loading q-table from saved state.")
            with open("my-saved-q-table.pt", "rb") as file:
                self.q_table:dict = dict(pickle.load(file))

def setup_training_vectors(self):
    # create new dictionary for action_vectors, if there is no according file.
    if not os.path.isfile("action_vectors.pt"):
        self.logger.info("setting up new action vectors")
        self.action_vectors = dict()
    else:
        self.logger.info("loading action vectors from saved state.")
        with open("action_vectors.pt", "rb") as file:
            self.action_vectors:dict = dict(pickle.load(file))


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    setup_q_table(self)
    setup_training_vectors(self)

def get_action_via_lin_val_approx(self, game_state):
        features = state_to_features(game_state)
        Q = np.zeros(len(ACTIONS))
        for i, a in enumerate(ACTIONS):
            # calculate Q value for action a via 'linear value approximation'
            if a not in self.action_vectors:
                # if there is no action vector yet, make a (random) guess
                l = len(features)
                self.action_vectors[a] = np.random.rand(l)
            Q[i] = np.dot(features, self.action_vectors[a])
        return ACTIONS[Q.argmax()]

def get_action_via_q_table(self, game_state):
    S_string = str(state_to_features(game_state))
    if S_string in self.q_table:
        self.logger.debug("State: "+ S_string+ ", choosing move: " + ACTIONS[self.q_table[S_string].argmax()])
        # self.logger.debug("q-values for cur state: "+str(self.q_table[S_string]))
        return ACTIONS[self.q_table[S_string].argmax()]
    self.logger.debug("No idea what to do. choosing randomly")
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

def get_action_via_q_table_else_via_lin_val_approx(self, game_state):
    S_string = str(state_to_features(game_state))
    if S_string in self.q_table:
        self.logger.debug("State: "+ S_string+ ", choosing move: " + ACTIONS[self.q_table[S_string].argmax()])
        # self.logger.debug("q-values for cur state: "+str(self.q_table[S_string]))
        return ACTIONS[self.q_table[S_string].argmax()]
    self.logger.debug("No q-table entry. Use linear value estimation: ")
    return get_action_via_lin_val_approx(self, game_state)

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return get_action_via_q_table_else_via_lin_val_approx(self, game_state)
    # return get_action_via_lin_val_approx(self, game_state)


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

    n = 1   # number of fields the agent can see in each direction

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []

    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    # bombs = game_state['bombs']
    # bomb_xys = [xy for (xy, t) in bombs]
    # others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']

    nearest_coin = get_nearest_coin(game_state)
    radar = create_coin_radar(coins, (x,y), max_dist=5)

    # Add selected features
    
    # channels.append(arena[x-n:x+n+1, y-n:y+n+1]) # field of view of short-sighted agent
    # channels.append(radar)  # shows in wich direction are coins
    
    channels.append( get_position_case( (x,y) ) )   # add position case (4 possible cases)
    channels.append( dist_to_proximity( get_direction_from_to( (x,y), nearest_coin, max_c=2 ), 3))
    channels.append( (1, -1) )   # two constants that might help the linear value approximation
    # channels.append( get_longest_direction(dir) )   # add direction in which agent has to go furthest


                               

    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)

def dist_to_proximity(pos:tuple, max_dist:int = 15)->tuple:
    # converts relative position to proximity values (high values mean that the position is near)
    a = max_dist - np.abs(pos[0])
    b = max_dist - np.abs(pos[1])
    if pos[0] < 0:
        a *= -1
    if pos[1] < 0:
        b *= -1
    return (a, b)

def get_direction_from_to(p1:tuple, p2:tuple, max_c:int=2)->tuple:
    # compute coordinates of vector from p1 to p2 and limit coordinates by max_c
    a = max(min( p2[0]-p1[0], max_c), -max_c)
    b = max(min( p2[1]-p1[1], max_c), -max_c)
    return (a, b)

def get_nearest_coin(state:dict):
    # TODO catch case in which there are no more coins left
    coins = state['coins'] 
    _, _, _, (x,y) = state['self']
    minDistToCoin = np.inf
    nearestCoin = (0, 0)
    for coin in coins:
        dist = manhattenDist((x,y), coin)
        if dist < minDistToCoin:
            minDistToCoin = dist
            nearestCoin = coin
    return nearestCoin
    
def create_coin_radar(coins, pos, max_dist=np.inf):
    rev_sorted_coins = coins.copy()
    rev_sorted_coins.sort(reverse=True, key=lambda a: manhattenDist(a, pos))
    radar = np.zeros((3,3))
    for coin in rev_sorted_coins:
        v = get_direction_from_to(pos, coin, 1)
        d = min(manhattenDist(coin, pos), max_dist)
        radar[v[0]+1,v[1]+1] = d
    return radar

def get_vector_from_to(p1, p2)->tuple:
    try:
        return (p2[0]-p1[0], p2[1]-p1[1])
    except:
        return (None,None)

def get_longest_direction(v):
    if (v[0] == None) or (v[1] == None):
        return (None, None)
    if np.abs(v[0]) > np.abs(v[1]):
        return (1, 0)
    else:
        return (0, 1)

def get_position_case(position):
    a:bool = position[0] % 2
    b:bool = position[1] % 2
    return (a, b)

def manhattenDist(p1, p2):
    try:
        return np.abs( p1[0]-p2[0] ) + np.abs( p1[1]-p2[1] )
    except:
        return None