import numpy as np
from itertools import permutations


class MODEL:
    def __init__(self):
        """
        initialization of the super class model: discrete states, actions, learning_rate, discount, rewards
        change of learning rates can defined here
        """
        self.model = np.array([])
        self.discrete_state_old = 0
        self.discrete_state_new = 0
        self.actions = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])
        self.learning_rate = 0.1
        self.discount = 0.95
        self.rewards = 0

    def q_learning(self, self_action):
        """
        The standard q-learning algorithm to update the model
        Args:
            self_action:

        Returns:
        a new value for the state we came from
        """
        old_q = self.model[self.discrete_state_old]
        max_future_q = np.max(self.model[self.discrete_state_new])
        action = int(np.where(np.array(self.actions) == self_action)[0])
        self.model[self.discrete_state_old][action] = (1 - self.learning_rate) * old_q[action] + self.learning_rate * (self.rewards + self.discount * max_future_q)
        self.end_of_action()

    def end_of_action(self):
        """
        transforms the new discrete state into the old state
        """
        self.discrete_state_old = self.discrete_state_new


class CRATES(MODEL):
    """
    A class of model only able to bomb crates in an intelligent method
    """
    def __init__(self):
        """
        additional variables need to add like:
            size_crates (total number of crates on the field)
            direction (a 4*1 array of 0,1 to indicate the possability of moving)
            called (bool to indicate if a crate was chosen to be bombed)
            arg_min (the index of the chosen crate)
            bomb_triggered (bool to indicated if a bomb was laid)
        """
        super().__init__()
        self.direction = np.unique(np.array(list(permutations([1, 1, 1, 0], 4))), axis=0)
        self.direction = np.concatenate(
            (self.direction, np.unique(np.array(list(permutations([1, 1, 0, 0], 4))), axis=0)), axis=0)
        self.direction = np.concatenate(
            (self.direction, np.unique(np.array(list(permutations([1, 0, 0, 0], 4))), axis=0)), axis=0)
        self.direction = np.concatenate((self.direction, np.array([[0, 0, 0, 0]])), axis=0)
        self.model = np.full(([9] + [8] + [len(self.direction)+1] + [len(self.actions)]), -0.5)
        self.size_crates = 0
        self.crate = False
        self.arg_min = 0
        self.bomb_triggered = False
        self.countdown = 3
        self.bomb_position = 0

    def update(self, state):
        """
        Updates the new discrete state
        """
        index_crates = self.get_crate_state(state)
        index_danger_zone = self.get_danger_zone(state)
        index_directions = self.get_possible_directions(state)
        self.discrete_state_new = (index_crates, index_danger_zone, index_directions)

    def get_danger_zone(self, state):
        """
        gives the featback if the players state is a state of danger,
         and if where the danger (bomb) is relative to player, or not (range of bomb or not)
        Inputs:
            state: game_state, get_relative_bomb(game_state)
        Returns:
        index_danger_zone in update
        """
        if state.get('self')[2]:
            # no bombs laid yet
            return 0
        else:
            relative_bomb = self.get_relative_bomb(state)
            if np.all(relative_bomb):
                # diagonal so no danger zone here
                return 1
            else:
                if np.abs(relative_bomb[0]) + np.abs(relative_bomb[1]) > 3:
                    # enough distance between bomb and player
                    return 2
                else:
                    if relative_bomb[0] == 0:
                        if relative_bomb[1] == 0:
                            # bomb on player
                            return 3
                        elif relative_bomb[1] < 0:
                            # bomb is top
                            return 4
                        else:
                            # bomb bottom
                            return 5
                    else:
                        if relative_bomb[0] > 0:
                            # bomb right
                            return 6
                        else:
                            # bomb left
                            return 7

    @staticmethod
    def get_relative_bomb(state):
        """
        a static_method to calculate the relative player bomb situation
        Inputs:
            game_state
        Returns:
            relative_bomb coordinates of chosen/closest bomb
        """
        position = np.array(state.get('self')[3])
        bombs = state.get('bombs')
        bombs = [bomb[0] for bomb in bombs]
        relative_bombs = bombs - position
        return relative_bombs[np.argmin(np.abs(relative_bombs.T[0]) + np.abs(relative_bombs.T[1]))]

    def get_possible_directions(self, state):
        """
        checking in which possible directions state the player is now
        Inputs:
            game_state, self.direction
        Returns:
            index_index_directions in update
        """
        position = np.array(state.get('self')[3])
        field = np.array(state.get('field'))
        field = field[position[0] - 1:position[0] + 2, position[1] - 1:position[1] + 2]
        field = np.array([field[1, 0], field[0, 1], field[2, 1], field[1, 2]])
        field = np.where(field != 0, 1, 0)
        for i, direction in enumerate(self.direction):
            if np.array_equal(direction, field):
                return i
        return 0

    def get_crate_state(self, state):
        """
        A function to differ all possibilities where a crate could be top right bottom .... and
        espacially if player is directly next to a crate
            game_state, self.get_closest_crate(state)
        Returns:
            index_crates in update
        """
        closest_crate, _ = self.get_closest_crate(state)
        # if not reachable by one coordinate
        if np.all(closest_crate):
            # bottom right
            if closest_crate[0] > 0 and closest_crate[1] > 0:
                return 0
            # bottom left
            elif closest_crate[0] < 0 < closest_crate[1]:
                return 1
            # top right
            elif closest_crate[0] > 0 > closest_crate[1]:
                return 2
            # top left
            else:
                return 3
        else:
            if np.abs(closest_crate[0]) + np.abs(closest_crate[1]) != 1:
                if closest_crate[0] == 0:
                    # top
                    if closest_crate[1] < 0:
                        return 4
                    # bottom
                    else:
                        return 5
                else:
                    # right
                    if closest_crate[0] > 0:
                        return 6
                    # left
                    else:
                        return 7
            else:
                # directly next to crate
                return 8

    def get_closest_crate(self, state, closest_index=-1):
        """
        Player should be only interested in 1 crate a time which is the next one
        Inputs:
            game_state, called, arg_min, size_crates
        Returns:
            coordinates of closest crate
        """
        field = state.get('field')
        player = state.get('self')[3]
        crates = np.argwhere(field == 1)

        relative_crates = crates - player
        manhattan = np.abs(relative_crates.T[0]) + np.abs(relative_crates.T[1])
        if self.size_crates != len(relative_crates):
            if closest_index == -1:
                if player[0] % 2 == 0 and player[1] % 2 != 0:
                    for i, crate in enumerate(relative_crates):
                        if crate[0] == 0 and crate[1] != 0:
                            manhattan[i] += 2
                elif player[1] % 2 == 0 and player[0] % 2 != 0:
                    for i, crate in enumerate(relative_crates):
                        if crate[1] == 0 and crate[0] != 0:
                            manhattan[i] += 2
                self.arg_min = np.argmin(manhattan)
        self.size_crates = crates.size
        return relative_crates[self.arg_min], self.arg_min

    def get_events(self, events, new_game_state, old_game_state, self_action):
        """
        Player should be only interested in 1 crate a time which is the next one
        Inputs:
            game_state, called, arg_min, size_crates
        Returns:
            coordinates of closest crate
        """
        if self.bomb_triggered:
            self.countdown -= 1
        if "BOMB_DROPPED" in events:
            self.bomb_triggered = True
            self.bomb_position = self.get_crate_state(new_game_state)
            if self.bomb_position == 8:
                events.append("PERFECT_MOVE")
        if self.countdown < 0:
            #if "CRATE_DESTROYED" not in events:
            #    events.append("FALSE_BOMB")
            self.countdown = 3
            self.bomb_triggered = False
        if old_game_state:
            if not new_game_state.get('bombs'):
                if self_action == "WAIT":
                    events.append("WAIT_NO_BOMB")
                old_size_crates = len(np.argwhere(old_game_state.get('field') == 1))
                new_size_crates = len(np.argwhere(new_game_state.get('field') == 1))
                if old_size_crates == new_size_crates:
                    old_position = old_game_state.get('self')[3]
                    new_position = new_game_state.get('self')[3]
                    old_crate, crate_index = self.get_closest_crate(old_game_state)
                    new_crate, _ = self.get_closest_crate(new_game_state, crate_index)
                    manhattan_old = np.abs(old_crate.T[0]) + np.abs(old_crate.T[1])
                    manhattan_new = np.abs(new_crate.T[0]) + np.abs(new_crate.T[1])
                    if old_position[0] % 2 == 0 and old_position[1] % 2 != 0:
                        if old_crate[0] == 0 and old_crate[1] != 0:
                            manhattan_old += 2
                    elif old_position[1] % 2 == 0 and old_position[0] % 2 != 0:
                        if old_crate[1] == 0 and old_crate[0] != 0:
                            manhattan_old += 2
                    if new_position[0] % 2 == 0 and new_position[1] % 2 != 0:
                        if new_crate[0] == 0 and new_crate[1] != 0:
                            manhattan_new += 2
                    elif new_position[1] % 2 == 0 and new_position[0] % 2 != 0:
                        if new_crate[1] == 0 and new_crate[0] != 0:
                            manhattan_new += 2
                    dist = manhattan_new - manhattan_old

                    if dist < 0:
                        events.append("CLOSER_CRATE")
                    else:
                        events.append("FURTHER_CRATE")
            else:
                if not old_game_state.get('self')[2]:
                    old_bomb = self.get_relative_bomb(old_game_state)
                    new_bomb = self.get_relative_bomb(new_game_state)
                    manhattan_old = np.abs(old_bomb.T[0]) + np.abs(old_bomb.T[1])
                    manhattan_new = np.abs(new_bomb.T[0]) + np.abs(new_bomb.T[1])
                    dist = manhattan_new - manhattan_old
                    if dist < 0:
                        events.append("CLOSER_BOMB")
                    elif dist > 0:
                        events.append("FURTHER_BOMB")
                    #if self.bomb_position == 8 and "CRATE_DESTROYED" in events:
                    #    events.append("PERFECT_MOVE")
                index_of_danger_zone = self.get_danger_zone(new_game_state)
                if 0 < index_of_danger_zone <= 2:
                    events.append("NO_DANGER")
                else:
                    events.append("DANGER")

        return events


class ENEMY(MODEL):
    """
    A class of model only able to bomb enemies in an intelligent method
    """
    def __init__(self):
        """
        additional variables need to add like:
            size_crates (total number of crates on the field)
            direction (a 4*1 array of 0,1 to indicate the possability of moving)
            called (bool to indicate if a crate was chosen to be bombed)
            arg_min (the index of the chosen crate)
            bomb_triggered (bool to indicated if a bomb was laid)
        """
        super().__init__()
        self.direction = np.unique(np.array(list(permutations([1, 1, 1, 0], 4))), axis=0)
        self.direction = np.concatenate(
            (self.direction, np.unique(np.array(list(permutations([1, 1, 0, 0], 4))), axis=0)), axis=0)
        self.direction = np.concatenate(
            (self.direction, np.unique(np.array(list(permutations([1, 0, 0, 0], 4))), axis=0)), axis=0)
        self.direction = np.concatenate((self.direction, np.array([[0, 0, 0, 0]])), axis=0)
        self.model = np.full(([9] + [8] + [len(self.direction)+1] + [len(self.actions)]), -0.5)
        self.size_crates = 0
        self.crate = False
        self.arg_min = 0
        self.bomb_triggered = False
        self.countdown = 3
        self.bomb_position = 0

    def update(self, state):
        """
        Updates the new discrete state
        """
        index_crates = self.get_crate_state(state)
        index_danger_zone = self.get_danger_zone(state)
        index_directions = self.get_possible_directions(state)
        self.discrete_state_new = (index_crates, index_danger_zone, index_directions)

    def get_danger_zone(self, state):
        """
        gives the featback if the players state is a state of danger,
         and if where the danger (bomb) is relative to player, or not (range of bomb or not)
        Inputs:
            state: game_state, get_relative_bomb(game_state)
        Returns:
        index_danger_zone in update
        """
        if not state.get('bombs'):
            # no bombs laid yet
            return 0
        else:
            relative_bombs = self.get_relative_bombs(state)
            relative_bomb = relative_bombs[np.argmin(np.abs(relative_bombs.T[0]) + np.abs(relative_bombs.T[1]))]
            if np.all(relative_bomb):
                # diagonal so no danger zone here
                return 1
            else:
                if np.abs(relative_bomb[0]) + np.abs(relative_bomb[1]) > 3:
                    # enough distance between bomb and player
                    return 2
                else:
                    if relative_bomb[0] == 0:
                        if relative_bomb[1] == 0:
                            # bomb on player
                            return 3
                        elif relative_bomb[1] < 0:
                            # bomb is top
                            return 4
                        else:
                            # bomb bottom
                            return 5
                    else:
                        if relative_bomb[0] > 0:
                            # bomb right
                            return 6
                        else:
                            # bomb left
                            return 7

    @staticmethod
    def get_relative_bombs(state):
        """
        a static_method to calculate the relative player bomb situation
        Inputs:
            game_state
        Returns:
            relative_bomb coordinates of chosen/closest bomb
        """
        position = np.array(state.get('self')[3])
        bombs = state.get('bombs')
        bombs = [bomb[0] for bomb in bombs]
        relative_bombs = bombs - position
        return relative_bombs

    def get_possible_directions(self, state):
        """
        checking in which possible directions state the player is now
        Inputs:
            game_state, self.direction
        Returns:
            index_index_directions in update
        """
        position = np.array(state.get('self')[3])
        bombs = state.get('bombs')
        bombs = [bomb[0] for bomb in bombs]
        field = np.array(state.get('field'))
        for bomb in bombs:
            field[bomb] = 5
        field = field[position[0] - 1:position[0] + 2, position[1] - 1:position[1] + 2]
        field = np.array([field[1, 0], field[0, 1], field[2, 1], field[1, 2]])
        field = np.where(field != 0, 1, 0)
        for i, direction in enumerate(self.direction):
            if np.array_equal(direction, field):
                return i
        return 0

    def get_crate_state(self, state):
        """
        A function to differ all possibilities where a crate could be top right bottom .... and
        espacially if player is directly next to a crate
            game_state, self.get_closest_crate(state)
        Returns:
            index_crates in update
        """
        closest_crate, _ = self.get_closest_crate(state)
        # if not reachable by one coordinate
        if np.all(closest_crate):
            # bottom right
            if closest_crate[0] > 0 and closest_crate[1] > 0:
                return 0
            # bottom left
            elif closest_crate[0] < 0 < closest_crate[1]:
                return 1
            # top right
            elif closest_crate[0] > 0 > closest_crate[1]:
                return 2
            # top left
            else:
                return 3
        else:
            if np.abs(closest_crate[0]) + np.abs(closest_crate[1]) != 1:
                if closest_crate[0] == 0:
                    # top
                    if closest_crate[1] < 0:
                        return 4
                    # bottom
                    else:
                        return 5
                else:
                    # right
                    if closest_crate[0] > 0:
                        return 6
                    # left
                    else:
                        return 7
            else:
                # directly next to crate
                return 8

    def get_closest_crate(self, state, closest_index=-1):
        """
        Player should be only interested in 1 crate a time which is the next one
        Inputs:
            game_state, called, arg_min, size_crates
        Returns:
            coordinates of closest crate
        """
        enemies = state.get('others')
        enemy = np.array([np.array(e[3]) for e in enemies])
        player = state.get('self')[3]
        relative_enemy = enemy - player
        manhattan = np.abs(relative_enemy.T[0]) + np.abs(relative_enemy.T[1])
        if self.size_crates != len(relative_enemy):
            if closest_index == -1:
                if player[0] % 2 == 0 and player[1] % 2 != 0:
                    for i, crate in enumerate(relative_enemy):
                        if crate[0] == 0 and crate[1] != 0:
                            manhattan[i] += 2
                elif player[1] % 2 == 0 and player[0] % 2 != 0:
                    for i, crate in enumerate(relative_enemy):
                        if crate[1] == 0 and crate[0] != 0:
                            manhattan[i] += 2
                self.arg_min = np.argmin(manhattan)
        self.size_crates = enemy.size
        return relative_enemy[self.arg_min], self.arg_min

    def get_events(self, events, new_game_state, old_game_state, self_action):
        """
        Player should be only interested in 1 crate a time which is the next one
        Inputs:
            game_state, called, arg_min, size_crates
        Returns:
            coordinates of closest crate
        """
        if self.bomb_triggered:
            self.countdown -= 1
        if "BOMB_DROPPED" in events:
            self.bomb_triggered = True
            self.bomb_position = self.get_crate_state(new_game_state)
            if self.bomb_position == 8:
                events.append("PERFECT_MOVE")
        if self.countdown < 0:
            #if "CRATE_DESTROYED" not in events:
            #    events.append("FALSE_BOMB")
            self.countdown = 3
            self.bomb_triggered = False
        if old_game_state:
            if not new_game_state.get('bombs'):
                if self_action == "WAIT":
                    events.append("WAIT_NO_BOMB")
                old_size_crates = len(old_game_state.get('others'))
                new_size_crates = len(new_game_state.get('others'))
                if old_size_crates == new_size_crates:
                    old_position = old_game_state.get('self')[3]
                    new_position = new_game_state.get('self')[3]
                    old_crate, crate_index = self.get_closest_crate(old_game_state)
                    new_crate, _ = self.get_closest_crate(new_game_state, crate_index)
                    manhattan_old = np.abs(old_crate.T[0]) + np.abs(old_crate.T[1])
                    manhattan_new = np.abs(new_crate.T[0]) + np.abs(new_crate.T[1])
                    if old_position[0] % 2 == 0 and old_position[1] % 2 != 0:
                        if old_crate[0] == 0 and old_crate[1] != 0:
                            manhattan_old += 2
                    elif old_position[1] % 2 == 0 and old_position[0] % 2 != 0:
                        if old_crate[1] == 0 and old_crate[0] != 0:
                            manhattan_old += 2
                    if new_position[0] % 2 == 0 and new_position[1] % 2 != 0:
                        if new_crate[0] == 0 and new_crate[1] != 0:
                            manhattan_new += 2
                    elif new_position[1] % 2 == 0 and new_position[0] % 2 != 0:
                        if new_crate[1] == 0 and new_crate[0] != 0:
                            manhattan_new += 2
                    dist = manhattan_new - manhattan_old
                    if dist < -1:
                        events.append("CLOSER_CRATE")
                    else:
                        events.append("FURTHER_CRATE")
            else:
                if not old_game_state.get('self')[2]:
                    old_bombs = self.get_relative_bombs(old_game_state)
                    new_bombs = self.get_relative_bombs(new_game_state)
                    if len(old_bombs) == len(new_bombs):
                        for i in range(len(old_bombs)):
                            manhattan_old = np.abs(old_bombs[i].T[0]) + np.abs(old_bombs[i].T[1])
                            manhattan_new = np.abs(new_bombs[i].T[0]) + np.abs(new_bombs[i].T[1])
                            dist = manhattan_new - manhattan_old
                            if dist < 0:
                                events.append("CLOSER_BOMB")
                            elif dist > 0:
                                events.append("FURTHER_BOMB")
                    #if self.bomb_position == 8 and "CRATE_DESTROYED" in events:
                    #    events.append("PERFECT_MOVE")
                index_of_danger_zone = self.get_danger_zone(new_game_state)
                if 0 < index_of_danger_zone <= 2:
                    events.append("NO_DANGER")
                else:
                    events.append("DANGER")

        return events
