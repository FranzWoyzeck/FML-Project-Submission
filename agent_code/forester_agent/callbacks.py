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

def setup_regressors(self):
    # only create new regressors, if the is no according file
    if not os.path.isfile("action_regressors.pt"):
        self.logger.info("Setting up action_regressors from scratch.")
        self.regressors: dict = dict()
    else:
        self.logger.info("Loading regressors from saved state.")
        with open("action_regressors.pt", "rb") as file:
            self.regressors:dict = dict(pickle.load(file))

def setup_stats(self):
    # only create new stats, if the is no according file
    if not os.path.isfile("stats.pt"):
        self.logger.info("Setting up stats from scratch.")
        self.stats: dict = dict()
        self.stats["n_games"] = 1
    else:
        self.logger.info("Loading stats from saved state.")
        with open("stats.pt", "rb") as file:
            self.stats:dict = dict(pickle.load(file))

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
    setup_regressors(self)
    setup_stats(self)

def get_action_via_regressors(self, game_state):
    if not self.regressors:
        return get_action_via_q_table(self, game_state)
    self.logger.debug("choosing action via regressors.")
    state = state_to_features(game_state)
    Q = np.zeros(len(ACTIONS))
    for i, a in enumerate(ACTIONS):
        Q[i] = self.regressors[a].predict( [state] )
    return ACTIONS[Q.argmax()]

def get_action_via_q_table_else_via_regressors(self, game_state):
    state_tuple = tuple(state_to_features(game_state).tolist())
    self.logger.debug("State: "+ str(state_tuple))
    if state_tuple in self.q_table:
        # find best action via q_table
        self.logger.debug("Choosing move via q-table: " + ACTIONS[self.q_table[state_tuple].argmax()])
        return ACTIONS[self.q_table[state_tuple].argmax()]

    if self.regressors:
        # find best action via regressors
        self.logger.debug("choosing action via regressors.")
        state = state_to_features(game_state)
        Q = np.zeros(len(ACTIONS))
        for i, a in enumerate(ACTIONS):
            Q[i] = self.regressors[a].predict( [state] )
        return ACTIONS[Q.argmax()]

    # guess best action (random)
    self.logger.debug("No regressors, choosing randomly")
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])


def get_action_via_q_table(self, game_state):
    state = tuple(state_to_features(game_state).tolist())
    if state in self.q_table:
        self.logger.debug("State: "+ str(state)+ ", choosing move via q-table: " + ACTIONS[self.q_table[state].argmax()])
        # self.logger.debug("q-values for cur state: "+str(self.q_table[S_string]))
        return ACTIONS[self.q_table[state].argmax()]
    self.logger.debug("No idea what to do. choosing randomly")
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])



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
        self.logger.debug("Choosing action purely at random (epsilon-greedy).")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    # return get_action_via_q_table(self, game_state)
    # return get_action_via_regressors(self, game_state)
    return get_action_via_q_table_else_via_regressors(self, game_state)


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
    bombs = game_state['bombs']
    # bomb_xys = [xy for (xy, t) in bombs]
    # others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']

    nearest_coin = get_nearest_coin(game_state)
    print(arena.size)
    a = np.reshape(arena[x-n:x+n+1, y-n:y+n+1], 9)
    c = np.asarray(get_direction_from_to( (x,y), nearest_coin, max_c=2 ))
    d = get_position_case((x, y))
    if bombs != []:
        #print(bombs[0][0])
        e = get_direction_from_to( (x,y), bombs[0][0], max_c=2 )
    else: 
        e = np.array([100, 100])

    channel = np.concatenate((a, c, d, e))

    # Add selected features
    channels.append(channel)
    # channels.append(arena[x-n:x+n+1, y-n:y+n+1]) # field of view of short-sighted agent
    # channels.append(radar)  # shows in wich direction are coins
    
    # channels.append( get_position_case( (x,y) ) )   # add position case (4 possible cases)
    # channels.append( get_direction_from_to( (x,y), nearest_coin, max_c=2 ))
    # channels.append( get_longest_direction(dir) )   # add direction in which agent has to go furthest


                               

    stacked_channels = np.stack(channels)
    return stacked_channels.reshape(-1)

def get_danger_zone(bombs):
    # TODO: sort bombs by countdown (descending)
    field_len = 17
    range:int = 4   #3 tiles in each direction
    danger_zone = np.zeros((field_len, field_len))
    for p, t in sorted (bombs, key = lambda x: x[1], reverse=True):
        danger_zone[ max(p[0]-range,0):min(p[0]+range, field_len) , p[1]] = t
        danger_zone[ p[0], max(p[1]-range,0):min(p[1]+range, field_len)] = t
    return danger_zone(2, 2), 4

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