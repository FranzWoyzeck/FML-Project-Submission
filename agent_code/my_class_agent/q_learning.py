import numpy as np
from itertools import permutations
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


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

    @staticmethod
    def get_distance(field, start, end):
        field = np.where(field != 0, -1, 1)
        grid = Grid(matrix=field)
        start = grid.node(start[0], start[1])
        end = grid.node(end[0], end[1])
        finder = AStarFinder()
        path, runs = finder.find_path(start, end, grid)
        return len(path)



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

        self.bombs = np.unique(np.array(list(permutations([1, 1, 1, 1, 0], 5))), axis=0)
        self.bombs = np.concatenate(
            (self.bombs, np.unique(np.array(list(permutations([1, 1, 1, 0, 0], 5))), axis=0)), axis=0)
        self.bombs = np.concatenate(
            (self.bombs, np.unique(np.array(list(permutations([1, 1, 0, 0, 0], 5))), axis=0)), axis=0)
        self.bombs = np.concatenate(
            (self.bombs, np.unique(np.array(list(permutations([1, 0, 0, 0, 0], 5))), axis=0)), axis=0)

        self.bombs = np.concatenate((self.bombs, np.array([[0, 0, 0, 0, 0]])), axis=0)
        self.bombs = np.concatenate((self.bombs, np.array([[1, 1, 1, 1, 1]])), axis=0)

        self.model = np.random.uniform(low=-2, high=-0.0001, size=([2] + [len(self.bombs)+1] + [len(self.bombs)+1] + [9] + [len(self.direction)+1] + [len(self.actions)]))
        self.size_crates = 0
        self.crate = False
        self.arg_min = 0
        self.bomb_triggered = False
        self.countdown = 3
        self.bomb_position = 0

    @staticmethod
    def add_enemy_field(enemy_position, x, y, field):
        if x:
            for i in range(-3, 4):
                field.append([enemy_position[0] + i, enemy_position[1]])
        if y:
            for i in range(-3, 4):
                field.append([enemy_position[0], enemy_position[1] + i])
        return field

    def enemy(self, state):
        enemies = [enemy[3] for enemy in state.get('others')]
        position = state.get('self')[3]
        field = []
        for enemy in enemies:
            if enemy[1] % 2 != 0 and enemy[0] % 2 == 0:
                field = self.add_enemy_field(enemy, True, False, field)
            elif enemy[1] % 2 == 0 and enemy[0] % 2 != 0:
                field = self.add_enemy_field(enemy, False, True, field)
            else:
                field = self.add_enemy_field(enemy, True, True, field)
        if list(position) in field:
            return 1
        else:
            return 0

    def closest_enemy(self, state):
        enemies = [enemy[3] for enemy in state.get('others')]
        field = state.get('field')
        position = state.get('self')[3]
        relative_enemies = []
        for enemy in enemies:
            relative_enemies.append(self.get_distance(field, position, enemy))
        return np.argmin(np.array(relative_enemies))

    @staticmethod
    def relative_direction(a, b):
        relative_direction = np.array(a) - np.array(b)
        if relative_direction[0] == 0:
            if relative_direction[1] > 0:
                return 0
            elif relative_direction[1] < 0:
                return 2
            else:
                return 8
        elif relative_direction[1] == 0:
            if relative_direction[0] > 0:
                return 3
            else:
                return 1
        else:
            if relative_direction[0] < 0 < relative_direction[1]:
                return 4
            elif relative_direction[0] < 0 > relative_direction[1]:
                return 5
            elif relative_direction[0] > 0 > relative_direction[1]:
                return 6
            elif relative_direction[0] > 0 < relative_direction[1]:
                return 7

    def relative_enemy_position(self, enemy, state):
        position = state.get('self')[3]
        return self.relative_direction(position, enemy)

    def update(self, state):
        """
        Updates the new discrete state
        """
        enemies = [enemy[3] for enemy in state.get('others')]
        enemy = enemies[self.closest_enemy(state)]
        index_relative_enemy = self.relative_enemy_position(enemy, state)
        index_enemy_zone = self.enemy(state)
        index_bomb_zone = self.get_index_bombs_zone(state)
        index_direction = self.get_possible_directions(state)
        self.discrete_state_new = (index_enemy_zone, index_bomb_zone[0], index_bomb_zone[1], index_relative_enemy, index_direction)

    @staticmethod
    def bomb_zone(state):
        bombs_array = np.zeros(10)
        position = np.array(state.get('self')[3])
        bombs = np.array([np.array(bomb[0]) for bomb in state.get('bombs')])
        if not bombs.size == 0:
            relative_bombs = bombs - position
            relative_bombs_distance = np.abs(relative_bombs.T[0] + relative_bombs.T[1])
            closest_bomb = relative_bombs[np.argmin(relative_bombs_distance)]
            for bomb in relative_bombs:
                if bomb[0] == 0:
                    # top or bottom or on player
                    if bomb[1] < 0:
                        bombs_array[1] = 1
                    elif bomb[1] > 0:
                        bombs_array[3] = 1
                    else:
                        bombs_array[9] = 1
                elif bomb[1] == 0:
                    # left or right
                    if bomb[0] > 0:
                        bombs_array[2] = 1
                    else:
                        bombs_array[4] = 1
                else:
                    # diagonal
                    if bomb[0] > 0:
                        if bomb[1] < 0:
                            bombs_array[5] = 1
                        else:
                            bombs_array[6] = 1
                    else:
                        if bomb[1] > 0:
                            bombs_array[7] = 1
                        else:
                            bombs_array[8] = 1
            if not (np.all(relative_bombs) or np.all(np.where(relative_bombs_distance <= 4, 0, 1))):
                # no diagonal so  danger zone here and not enough distance between player and bomb
                bombs_array[0] = 1
        return bombs_array

    def get_index_bombs_zone(self, state):
        position = self.bomb_zone(state)
        a = [0, 0]
        for i, direction in enumerate(self.bombs):
            if np.array_equal(direction, position[:5]):
                a[0] = i
            if np.array_equal(direction, position[5:]):
                a[1] = i
        return a

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

    def get_events(self, events, new_game_state, old_game_state, self_action):
        """
        Player should be only interested in 1 crate a time which is the next one
        Inputs:
            game_state, called, arg_min, size_crates
        Returns:
            coordinates of closest crate
        """
        if old_game_state:
            old_position = old_game_state.get('self')[3]
            new_position = new_game_state.get('self')[3]
            old_field = old_game_state.get('field')
            new_field = new_game_state.get('field')
            if not new_game_state.get('bombs'):
                if self_action == "WAIT":
                    events.append("WAIT_BOMB")
            else:
                danger_zone = self.bomb_zone(new_game_state)[0]
                if danger_zone == 0:
                    events.append("NO_DANGER")
                else:
                    events.append("DANGER")
                if old_game_state.get('bombs'):
                    old_bombs = [bomb[0] for bomb in old_game_state.get('bombs')]
                    new_bombs = [bomb[0] for bomb in new_game_state.get('bombs')]
                    old_dist = []
                    for old_bomb in old_bombs:
                        old_dist.append(self.get_distance(old_field, old_position, old_bomb))
                    new_dist = []
                    for new_bomb in new_bombs:
                        new_dist.append(self.get_distance(new_field, new_position, new_bomb))
                    if np.min(np.array(new_dist)) <= np.min(np.array(old_dist)):
                        events.append("CLOSER_BOMB")
                    else:
                        events.append("FURTHER_BOMB")

            if self.enemy(new_game_state) == 0:
                events.append("NO_ENEMY")
                new_enemies = [enemy[3] for enemy in new_game_state.get('others')]
                old_enemies = [enemy[3] for enemy in old_game_state.get('others')]

                old_enemy = old_enemies[self.closest_enemy(old_game_state)]
                new_enemy = new_enemies[self.closest_enemy(new_game_state)]

                new_dist = self.get_distance(new_field, new_position, new_enemy)
                old_dist = self.get_distance(old_field, old_position, old_enemy)
                if new_dist < old_dist:
                    events.append("CLOSER_ENEMY")
                elif new_dist > old_dist:
                    events.append("FURTHER_ENEMY")

            else:
                events.append("ENEMY")

        return events
