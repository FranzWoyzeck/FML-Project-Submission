import random
from collections import namedtuple, deque
from typing import List
import os
import torch
from torch import optim
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS
from .DQN import DQN, device, optimize_model, TARGET_UPDATE
from settings import MAX_STEPS
from torch.utils.tensorboard import SummaryWriter

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 9  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LOOPS_SIZE  = 6
# Events



class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args, loop=False):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        if loop:
            self.memory[self.position] = args
        else:
            self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # default `log_dir` is "runs" - we'll be more specific here
    print("Start Training")
    self.writer = SummaryWriter('./runs')
    self.memory = ReplayMemory(10000)
    self.loops = ReplayMemory(LOOPS_SIZE)
    self.coordinate_history = deque([], 20)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.1)
    self.count = 0
    self.steps_done = 0
    self.running_loss = 0


def action_to_number(action):
    for i, a in enumerate(ACTIONS):
        if a == action:
            return i


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
    # self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state == None:
        old_game_state = new_game_state

    # Gather information about the game state
    arena = new_game_state['field']
    _, score, bombs_left, (x, y) = new_game_state['self']
    bombs = new_game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in new_game_state['others']]
    coins = new_game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    if self_action is not None:
        # If agent has been in the same location three times recently, it's a loop
        self.loops.push(self_action, loop=True)
        if len(set(self.loops.memory)) < 4:
            events.append(e.REPEAT)
    if self.coordinate_history.count((x, y)) > 2:
        events.append(e.LOOP)
    self.coordinate_history.append((x, y))


        
    for coin in coins:
        coin = np.array(coin)
        position = np.array([x,y])
        dist = np.linalg.norm(coin-position)



    # Idea: Add your own events to hand out rewards
    vec_new = np.array(new_game_state["self"][3])
    vec_old = np.array(old_game_state["self"][3])
    middle = np.array([8,8])
    direction = vec_new-vec_old
    dist_old = np.linalg.norm(vec_old-middle)
    dist_new = np.linalg.norm(vec_new-middle)
    self.nsteps = new_game_state["round"]

    if vec_new[0]==1 or vec_new[0]==15 or vec_new[1]==1 or vec_new[1]==15:
        events.append(e.WALL_AREA)

    if self_action != "WAIT":
        events.append(e.NOT_WAITED)

    if self_action is not None:
        # state_to_features is defined in callbacks.py
        ogs = torch.tensor(np.array([state_to_features(self, old_game_state)]), device=device, dtype=torch.float)
        ngs = torch.tensor(np.array([state_to_features(self, new_game_state)]), device=device, dtype=torch.float)
        reward = torch.tensor(np.array([reward_from_events(self, events)]), device=device, dtype=torch.long)
        action = torch.tensor([action_to_number(self_action)], device=device, dtype=torch.long)

        self.memory.push(ogs, action, ngs, reward)
        optimize_model(self)

        # Update the target network, copying all weights and biases in DQN
        self.count += 1
        if self.count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.end = True
    ogs = torch.tensor(np.array([state_to_features(self, last_game_state)]), device=device, dtype=torch.float)
    reward = torch.tensor(np.array([reward_from_events(self, events)]), device=device, dtype=torch.long)
    action = torch.tensor([action_to_number(last_action)], device=device, dtype=torch.long)

    self.memory.push(ogs, action, None, reward)
    optimize_model(self)
    # Store the model
    torch.save(self.policy_net.state_dict(), "my-saved-model")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    global MAX_STEPS
    game_rewards = {
        e.LOOP: -1,
        e.COIN_COLLECTED: 10,
        e.MOVED_LEFT: 0.1,
        e.MOVED_RIGHT: 0.1,
        e.MOVED_UP: 0.1,
        e.MOVED_DOWN: 0.1,
        e.WAITED: -1,
        e.INVALID_ACTION: -1,
        e.NOT_WAITED: 0,
        e.REPEAT: -1,
        e.WALL_AREA: -1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0
    }
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print(events, reward_sum)
    return reward_sum
