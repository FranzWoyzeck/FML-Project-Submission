import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
old_pos = None

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
    # only create new model, if the is no according file
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model: dict = dict()

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model:dict = dict(pickle.load(file))


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

    #self.logger.debug("Querying model for action.")
    S_string = str(state_to_features(game_state))
    if S_string in self.model:
        self.logger.debug("State: "+ S_string+ ", choosing move: " + ACTIONS[self.model[S_string].argmax()])
        self.logger.debug("q-values for cur state: "+str(self.model[S_string]))
        return ACTIONS[self.model[S_string].argmax()]
    self.logger.debug("No idea what to do. choosing randomly")
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])


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

    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']

    nearest_coin = get_nearest_coin(game_state)
    dir = get_direction_from_to( (x,y), nearest_coin )

    # Add selected featuress
    channels.append( get_position_case( (x,y) ) )   # add position case (4 possible cases)
    channels.append( dir )                          # add directions in which the agent has to go (just direction, no length)
    channels.append( get_longest_direction(dir) )   # add direction in which agent has to go furthest
    #channels.append( getVectorFromTo( (x,y), nearestCoin) )    
    #channels.append(nearestCoin)
    if old_pos:
        channels.append( get_direction_from_to( (old_x, old_y), (x,y) ) )     # where did the agent come from? (this might prevent moving to previous state)
    (old_x, old_y) = (x,y)
                               
    
    

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)

def get_direction_from_to(p1, p2):
    v = get_vector_from_to(p1, p2)
    if (v[0] == None) or (v[1] == None):
        return (None, None)
    if v[0] < 0:
        a = -1
    elif v[0] > 0: 
        a = 1
    else: 
        a = 0
    if v[1] < 0:
        b = -1
    elif v[1] > 0: 
        b = 1
    else: 
        b = 0
    return (a,b)


def get_nearest_coin(state:dict):
    # TODO catch case in which there are no more coins left
    coins = state['coins'] 
    _, _, _, (x,y) = state['self']
    minDistToCoin = np.inf
    nearestCoin = (None, None)
    for coin in coins:
        dist = manhattenDist((x,y), coin)
        if dist < minDistToCoin:
            minDistToCoin = dist
            nearestCoin = coin
    return nearestCoin

def get_vector_from_to(p1, p2):
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