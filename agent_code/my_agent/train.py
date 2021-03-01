import numpy as np
import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, get_nearest_coin, manhattenDist

ACTION_TO_INT = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE: int = 3        # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS: float = 1.0   # record enemy transitions with probability ...
ALPHA: float = 0.05                     # learning rate
GAMMA:float = 0.9                       # Discount
#Q_INIT = np.array([1., 1., 1., 1., 1. , 1.])    #TODO: currently not in useossenhei

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

    # Idea: Add your own events to hand out rewards
    # TODO: reward if distance to nearest coin decreased
    if new_game_state and old_game_state:
        old_nearest_coin = get_nearest_coin(old_game_state)
        new_nearest_coin = get_nearest_coin(new_game_state)
        _, _, _, old_pos = old_game_state['self']
        _, _, _, new_pos = new_game_state['self']
        
        old_dist_to_coin = manhattenDist(old_pos, old_nearest_coin)
        new_dist_to_coin = manhattenDist(new_pos, new_nearest_coin)
        if old_dist_to_coin and new_dist_to_coin:   #makes sure the distance is not None, due to missing coins
            if old_dist_to_coin > new_dist_to_coin:
                events.append(DECREASE_COIN_DISTANCE)
                self.logger.debug("APPENDED DECREASE_COIN_DISTANCE")
            else:
                events.append(INCREASE_COIN_DISTANCE)
                self.logger.debug("APPENDED INCREASE_COIN_DISTANCE")



    # state_to_features is defined in callbacks.py
    if new_game_state and old_game_state:   # makes sure there is no state_to_feature(state) = None
        # check for loops in the agents behaviour, i.e. agent reaches a state twice
        next_state_features = state_to_features(new_game_state)
        if detect_loop(self, next_state_features):
            self.logger.debug("APPENDED DETECTED_LOOP")
            events.append(DETECTED_LOOP)

        # append new transition
        self.transitions.append(Transition(state_to_features(old_game_state), self_action, next_state_features, reward_from_events(self, events)))
        q_learning(self)

        
        
#TODO: test punish_loops function
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
    self.model[s_cur][a] = q_s_cur_a + ALPHA * ( r + GAMMA * q_s_next_a_best - q_s_cur_a )

    
def sarsa_OLD(q_cur, r_cur, q_next): 
    return q_cur + ALPHA * ( r_cur + GAMMA * q_next  - q_cur)

# TODO
def q_learning_OLD(q_cur, r_cur, q_col_next):
    return q_cur + ALPHA * ( r_cur + GAMMA * q_col_next.argmax()  - q_cur)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
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
