import numpy as np
import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, get_nearest_coin, manhattenDist, ACTIONS


ACTION_TO_INT = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE: int = 30        # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS: float = 1.0   # record enemy transitions with probability ...
ALPHA: float = 0.05                     # learning rate
GAMMA:float = 0.9                       # Discount

# model stats
N_GAMES = "N_GAMES"
N_MOVES = "N_MOVES"

# Events
DECREASE_COIN_DISTANCE = "DECREASE_COIN_DISTANCE"
INCREASE_COIN_DISTANCE = "INCREASE_COIN_DISTANCE"
DETECTED_LOOP = "DETECTED_LOOP"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # init model stats
    if N_GAMES not in self.model:
        self.model[N_GAMES] = 0
    if N_MOVES not in self.model:
        self.model[N_MOVES] = 0


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.model[N_MOVES] += 1    # count up number of moves
    
    if new_game_state and old_game_state:
        # reward if distance to nearest coin decreased
        old_nearest_coin = get_nearest_coin(old_game_state)
        new_nearest_coin = get_nearest_coin(new_game_state)
        _, _, _, old_pos = old_game_state['self']
        _, _, _, new_pos = new_game_state['self']
        
        old_dist_to_coin = manhattenDist(old_pos, old_nearest_coin)
        new_dist_to_coin = manhattenDist(new_pos, new_nearest_coin)
        if old_dist_to_coin and new_dist_to_coin:   #makes sure the distance is not None, due to missing coins
            if old_dist_to_coin > new_dist_to_coin:
                events.append(DECREASE_COIN_DISTANCE)
            else:
                events.append(INCREASE_COIN_DISTANCE)

        # check for loops in the agents behaviour, i.e. agent reaches a state twice
        next_state_features = state_to_features(new_game_state)
        if detect_loop(self, next_state_features):
            events.append(DETECTED_LOOP)


        ### append new transition. EVENTS HAVE TO BE ADDED BEFOREHAND ###
        self.transitions.append(Transition(state_to_features(old_game_state), self_action, next_state_features, reward_from_events(self, events)))
        q_val = q_learning(self)    # updates Q-value for correspronding game_state-action-pair



        # TODO: update Q_value for all analogue game_state - action-pairs
        """
        state_action_pairs =  get_analogue_game_states_action_pairs(old_game_state, self_action, self)
        for s_a_pair in state_action_pairs:
            state = s_a_pair[0]
            action = s_a_pair[1]
            #self.logger.debug("state: "+str(state))
            #self.logger.debug("action: "+str(action))
            s_string = str(state_to_features(s_a_pair[0]))
            self.logger.debug("analogue state: " + s_string + str(ACTIONS[s_a_pair[1]]))
            if s_string not in self.model:
                self.model[s_string] = np.ones(6) +  np.random.rand(6)/10 # TODO: guess init values
            self.model[s_string][action] = q_val
        """


        
        
# test punish_loops function
def detect_loop(self, next_state_features):
    for t in self.transitions:
        if (next_state_features == t.next_state).all():
            return True
    return False

# TODO implement n-step q-learning
def q_learning(self):
    if len(self.transitions) < 1:
        return
    t = self.transitions[-1]    # contains last ('state', 'action', 'next_state', 'reward')
    s_cur = str(t.state)
    s_next = str(t.next_state)
    a = ACTION_TO_INT[t.action]
    r = t.reward
    # set initial q-values for not yet visited states
    if s_cur not in self.model:
        self.model[s_cur] = np.ones(6) +  np.random.rand(6)/10# TODO: guess init values
    if s_next not in self.model:
        self.model[s_next] = np.ones(6) + np.random.rand(6)/10# TODO: guess init values
    # calculate new q-value
    q_s_cur_a = self.model[s_cur][a]
    q_s_next_a_best = self.model[s_next].argmax()
    q_val = q_s_cur_a + ALPHA * ( r + GAMMA * q_s_next_a_best - q_s_cur_a )
    self.model[s_cur][a] = q_val
    return q_val
    


# TODO DEBUG: overwriting new_state does not work
def get_analogue_game_states_action_pairs(game_state, action:str, self, N=15):
    state_action_pairs = []
    new_state = game_state.copy()
    new_action = action

    for i in range(2): # mirrorings
        state_action_pairs.append((new_state.copy(), ACTION_TO_INT[new_action]))

        for j in range(3): # rotations
            # rotate field
            new_state['field'] = np.rot90(new_state['field'])
            # rotate own position
            name, score, bombs_left, (x, y) = new_state['self']
            new_state['self'] = name, score, bombs_left, rot90_pos((x, y))
            # TODO: rotate bomb positions
            # TODO: rotate position of other players
            # rotate coin positions
            coins = new_state['coins']
            new_state['coins'] = [rot90_pos(c) for c in coins]
            # rotate action
            new_action = rot90_action_string(new_action)
            # add state and action to resp. arrays
            state_action_pairs.append( (new_state.copy(), ACTION_TO_INT[new_action]) )

        # mirror field
        new_state['field'] = np.flipud(new_state['field'])
        # mirror own position
        name, score, bombs_left, (x, y) = new_state['self']
        new_state['self'] = name, score, bombs_left, mirror_pos((x,y))
        # TODO: mirror bomb positions
        # TODO: mirror positions of other players
        # mirror coin positions
        coins = new_state['coins']
        coins = [mirror_pos(c) for c in coins]
        new_state['coins'] = coins
        # mirror action
        new_action = mirror_action_string(new_action)

    return state_action_pairs    

# tested and works fine
def mirror_pos(pos:tuple, N=15):
    return (N-pos[0]+1, pos[1])

# tested and works fine
def rot90_pos(pos:tuple, N=15):
    return (N-pos[1]+1 , pos[0])

# Tested and works fine
def mirror_action_string(action_string:str)->str:
    action_int = ACTION_TO_INT[action_string]
    if action_int < 4:
        return ACTIONS[(action_int +2)%4]
    else:
        return ACTIONS[action_int]

# tested and works fine
def rot90_action_string(action_string:str)-> str:
    action_int = ACTION_TO_INT[action_string]
    if action_int < 4:
        return ACTIONS[(action_int +1)%4]
    else:
        return ACTIONS[action_int]



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.model[N_GAMES] += 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))



    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.DECREASE_COIN_DISTANCE: 1,
        e.INCREASE_COIN_DISTANCE: -1,
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -20,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.BOMB_DROPPED: -100,
        e.WAITED: -10,
        e.GOT_KILLED: -1000,
        e.DETECTED_LOOP: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
