import os
import numpy as np
from .DQN import DQN, select_action, device
import torch

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'WAIT']

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
    torch.manual_seed(0)
    self.policy_net = DQN(17, 17, 6).to(device)
    if os.path.isfile("my-saved-model"):
        self.policy_net.load_state_dict(torch.load("my-saved-model"))
        self.policy_net.eval()
        print("Loaded saved model")
    self.target_net = DQN(17, 17, 6).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.count = 0
    self.steps_done = 0
    self.old_gs = None
    self.end = False
    self.verbose = False


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # self.logger.debug("Querying model for action.")
    action = ACTIONS[select_action(self, state_to_features(self, game_state))]
    self.old_gs = game_state
    if self.verbose:
        print(action)
    return action


def state_to_features(self, game_state: dict) -> np.array:
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
    # concatenate them as a feature tensor (they must have the same shape), ...
    coins = game_state["coins"]
    pos = np.array(game_state["self"][3])

    tmp = np.zeros((17, 17))#game_state["field"]
    tmp[pos[0], pos[1]] = -1

    for k in range(len(coins)):
        i,j = coins[k]
        tmp[i,j] = 1

    channels.append(tmp)
    # todo small windows
    #print(channels)
    return channels
