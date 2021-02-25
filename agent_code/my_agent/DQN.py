import math
import random
import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 512
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 1e5
TARGET_UPDATE = 10


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        ks = 3
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=ks, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=ks, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=ks, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=ks, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def select_action(self, state):
    if self.train:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        #print(self.steps_done)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state_ = torch.tensor([state], device=device, dtype=torch.float)
                return self.policy_net(state_).max(1)[1].view(1, 1)
        else:
            if self.verbose:
                print("random chosen")
            return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state_ = torch.tensor([state], device=device, dtype=torch.float)
            return self.policy_net(state_).max(1)[1].view(1, 1)


def optimize_model(self):
    if len(self.memory) < BATCH_SIZE:
        return
    transitions = self.memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).reshape(BATCH_SIZE, 1)
    reward_batch = torch.cat(batch.reward)

    q_values = self.policy_net(state_batch).gather(1, action_batch)

    target_pred = torch.zeros(BATCH_SIZE, device=device)
    target_pred[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    target_values = (target_pred * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))
    #print(loss)
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    self.running_loss += loss.item()
    if self.steps_done % 10 == 9:  # every 1000 mini-batches...
        # ...log the running loss
        self.writer.add_scalar('training loss',
                          self.running_loss,
                         BATCH_SIZE + self.steps_done)

        # ...log a Matplotlib Figure showing the model's predictions on a
        # random mini-batch
        self.running_loss = 0.0


