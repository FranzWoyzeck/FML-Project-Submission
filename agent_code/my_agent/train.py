import pickle
import random
from collections import namedtuple, deque
from sklearn.preprocessing import normalize
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features, get_discrete_state, get_relative_min_coin_direction

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'])

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
FURTHER_EVENT = "FURTHER"
CLOSER_EVENT = "CLOSER"
CORNER_EVENT = "CORNER"

# Bool
FOUND_COIN = False
CLOSEST_COIN = 0

COLLECTED_COIN_EVENT = "COLLECTED_COIN"

LOOP_EVENT = "LOOP"

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


def q_learning(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    discrete_state = (get_discrete_state(old_game_state))

    current_q = self.model[discrete_state]
    new_discrete_state = (get_discrete_state(new_game_state))
    max_future_q = np.max(self.model[new_discrete_state])
    action = int(np.where(np.array(ACTIONS) == self_action)[0])

    reward = reward_from_events(self, events)

    new_model = (1 - LEARNING_RATE) * current_q[action] + LEARNING_RATE * \
                (reward + DISCOUNT * max_future_q)

    self.model[discrete_state][action] = new_model
    for i in range(3):
        self.model[i] = normalize(self.model[i], axis=1, norm='l1')

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
    reward = reward_from_events(self, events)
    if old_game_state:
        if True:
            vec_alt = get_relative_min_coin_direction(old_game_state)
            vec_neu = get_relative_min_coin_direction(new_game_state)
            if new_game_state.get('self')[3][0] % 2 == 0 and new_game_state.get('self')[3][1] % 2 != 0:
                if vec_neu[0] == 0 and vec_neu[1] != 0:
                    vec_neu[0] = vec_neu[0] + 2
            elif new_game_state.get('self')[3][1] % 2 == 0 and new_game_state.get('self')[3][0] % 2 != 0:
                if vec_neu[1] == 0 and vec_neu[0] != 0:
                    vec_neu[1] = vec_neu[1] + 2
            vec_alt = np.abs(vec_alt[0]) + np.abs(vec_alt[1])
            vec_neu = np.abs(vec_neu[0]) + np.abs(vec_neu[1])
            if vec_alt == 1 and int(len(old_game_state.get('coins')) > int(len(new_game_state.get('coins')))):
                events.append(COLLECTED_COIN_EVENT)
            if vec_alt - vec_neu <= 0:
                events.append(FURTHER_EVENT)
            else:
                events.append(CLOSER_EVENT)

            if self.transitions[0][0] and self.transitions[2][0] and self.transitions[3][0] and self.transitions[1][0]:
                position_0 = self.transitions[0][0].get('self')[3]
                position_1 = self.transitions[1][0].get('self')[3]
                position_2 = self.transitions[2][0].get('self')[3]
                position_3 = self.transitions[3][0].get('self')[3]
                if position_0 == position_2 and position_1 == position_3:
                    for i in range(4):
                        q_learning(self, self.transitions[i][0], self.transitions[i][1], self.transitions[i][2], [LOOP_EVENT])

            q_learning(self, old_game_state, self_action, new_game_state, events)
            # Events before here


        # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward_from_events(self, events)))


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
    print(self.model)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    global COINS_COLLECTED
    game_rewards = {
        COLLECTED_COIN_EVENT: 50,
        e.INVALID_ACTION: -3,
        e.WAITED: -10,
        FURTHER_EVENT: -1,
        CLOSER_EVENT: 2,
        LOOP_EVENT: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
