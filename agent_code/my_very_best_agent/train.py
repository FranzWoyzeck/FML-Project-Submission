import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features, get_discrete_state, get_current_state

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
FURTHER_EVENT = "FURTHER"
CLOSER_EVENT = "CLOSER"
CORNER_EVENT = "CORNER"

# Bool
FOUND_COIN = False
CLOSEST_COIN = 0

LEARNING_RATE = 0.1
DISCOUNT = 0.95


COINS_COLLECTED = 0



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def get_environment_coordinates(state: dict):
    kernel = np.empty([3, 3], dtype='i,i')
    for i in range(-1, 2):
        for j in range(-1, 2):
            kernel[i+1, j+1] = (state.get('self')[3][0] + i, state.get('self')[3][0] + j)
    print(kernel)


def get_environment_data(state: dict):
    environment = np.array(state.get('field'))
    index = np.array(state.get('self')[3])
    kernel = np.empty([3, 3])
    for i in range(-1, 2):
        for j in range(-1, 2):
            tmp_index = np.array([index[0]+i, index[1]+j])
            kernel[i+1, j+1] = environment[tmp_index[0], tmp_index[1]]
    return kernel



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
    global FOUND_COIN
    global CLOSEST_COIN
    if self.transitions and old_game_state:
        if True:
            vec_neu = np.abs(np.array(new_game_state.get('coins')-np.array(new_game_state.get('self')[3])))
            CLOSEST_COIN = np.argmin(vec_neu.T[0]+vec_neu.T[1])

            #coins = new_game_state.get('coins')[CLOSEST_COIN]

            discrete_state = get_discrete_state(old_game_state)
            current_q = self.model[discrete_state]

            new_discrete_state = get_discrete_state(new_game_state)
            max_future_q = np.max(self.model[new_discrete_state])

            action = int(np.where(np.array(ACTIONS) == self_action)[0])

            new_model = (1 - LEARNING_RATE) * current_q[action] + LEARNING_RATE * (reward_from_events(self, events) + DISCOUNT
                                                                 * max_future_q)
            self.model[discrete_state][action] = new_model

            vec_alt = np.abs(np.array(old_game_state.get('coins'))[CLOSEST_COIN] - np.array(old_game_state.get('self')[3]))
            vec_neu = np.abs(np.array(new_game_state.get('coins'))[CLOSEST_COIN] - np.array(new_game_state.get('self')[3]))
            vec_alt = vec_alt.T[0] + vec_alt.T[1]
            vec_neu = vec_neu.T[0] + vec_neu.T[1]
            if vec_alt - vec_neu <= 0:
                events.append(FURTHER_EVENT)
            else:
                events.append(CLOSER_EVENT)

        # state_to_features is defined in callbacks.py

    # if the player is at a corner it should trigger a negative reward
    if np.sum(get_environment_data(new_game_state) == -1) >= 6:
        events.append(CORNER_EVENT)

    self.transitions.append(Transition(old_game_state, self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    global COINS_COLLECTED
    COINS_COLLECTED = 0
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))
    # Store the model
    self.epsilon -= self.epsilon_decay_value
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    global COINS_COLLECTED
    game_rewards = {
        e.COIN_COLLECTED: 50,
        #e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -2,
        e.WAITED: -5,
        # idea: the custom event is bad
        FURTHER_EVENT: -1,


    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
