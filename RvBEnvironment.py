import numpy as np

from RvB_env import GameBoard, read_config
from ship import Ship
from RvBVisualizer import draw
from RvBFieldSetup import set_field


class RvBEnvironment:

    def __init__(self, stationary_drone=False, config_file=None):

        self.config = read_config(config_file)
        self.actions = ['a', 'w', 'd', 'x']

        self.ddg_number = 0
        self.sag_number = 0

        self.ddg_already_rewarded = False
        self.sag_already_rewarded = False

        self.stationary_drone = stationary_drone

        self.game = None
        self.steps_taken = 0
        self.reset()

    def get_observation(self):

        for ship in self.game.active_ships + self.game.ships_at_dest + self.game.found_ships:
            if ship.number == self.ddg_number:
                _, ddg_dim_observation = ship.create_dimentional_viewport(active_ships=self.game.active_ships, drones=self.game.drones, number_of_ships=2)
                unit_vector = ship.get_unit_dir_to_goal()
                ddg_goal_vector = np.array([unit_vector[0], unit_vector[1]])
                ddg_heading = Ship.dirs[ship.heading_idx]
                # ddg_missle_count = ship.missile_count
            elif ship.number == self.sag_number:
                _, sag_dim_observation = ship.create_dimentional_viewport(active_ships=self.game.active_ships, drones=self.game.drones, number_of_ships=2)
                unit_vector = ship.get_unit_dir_to_goal()
                sag_goal_vector = np.array([unit_vector[0], unit_vector[1]])
                sag_heading = Ship.dirs[ship.heading_idx]
                # sag_missle_count = ship.missile_count

        # observation = (ddg_dim_observation, ddg_goal_vector, ddg_heading, ddg_missle_count, sag_dim_observation, sag_goal_vector, sag_heading, sag_missle_count)
        observation = (ddg_dim_observation, ddg_goal_vector, ddg_heading, sag_dim_observation, sag_goal_vector,sag_heading)

        return observation

    def get_all_ships(self):

        return self.game.active_ships + self.game.found_ships + self.game.ships_at_dest

    def get_individual_reward(self, ship_id, is_ddg):

        # has the ship been found
        for ship in self.game.found_ships:
            if ship.number == ship_id:
                # experiment 3 - changing the reward
                return -100

        # did the ship reach its destination the first time?
        for ship in self.game.ships_at_dest:
            if ship.number == ship_id:
                if is_ddg:
                    return 1000
                else:
                    # return 200
                    # experiment 3 - changing the reward
                    return 100

        return -1

    def step(self, actions):

        self.steps_taken += 1

        # take the actions
        p1_action, p2_action = actions
        self.game.env_step((self.ddg_number, self.actions[p1_action]), (self.sag_number, self.actions[p2_action]))

        ddg_reward = -1
        sag_reward = -1

        if self.ddg_already_rewarded:
            ddg_reward = 0
        else:
            ddg_reward = self.get_individual_reward(self.ddg_number, True)
            if ddg_reward > -1:
                self.ddg_already_rewarded = True

        if self.sag_already_rewarded:
            sag_reward = 0
        else:
            sag_reward = self.get_individual_reward(self.sag_number, False)
            if sag_reward > -1:
                self.sag_already_rewarded = True

        ddg_observation = None
        sag_observation = None

        observation = self.get_observation()
        reward = (ddg_reward, sag_reward)

        max_steps = 50

        # done if max steps taken or either of the ships were caught
        # done = self.steps_taken >= max_steps or len(self.game.found_ships) > 0 or len(self.game.ships_at_dest) == 2

        ddg_found = False
        for ship in self.game.found_ships:
            if ship.number == self.ddg_number:
                ddg_found = True
                break
        done = self.steps_taken >= max_steps or ddg_found or len(self.game.ships_at_dest) == 2

        return observation, reward, done, {}

    def reset(self):

        self.game = GameBoard(**self.config)
        set_field(self.game)

        self.steps_taken = 0
        self.ddg_already_rewarded = False
        self.sag_already_rewarded = False

        if self.stationary_drone:
            self.game.drones[0].x = 6
            self.game.drones[0].y = 4

        self.ddg_number = self.game.active_ships[0].number
        self.sag_number = self.game.active_ships[1].number

        observation = self.get_observation()

        return observation

    def render(self, mode='human', close=False):

        all_ships = self.game.active_ships + self.game.found_ships + self.game.ships_at_dest
        ddg = None
        sag = None
        for ship in all_ships:
            if ship.get_number() == self.ddg_number:
                ddg = ship
            elif ship.get_number() == self.sag_number:
                sag = ship

        ships = [ddg, sag]
        ship_types = ['ddg', 'sag']
        drones = self.game.drones

        img = draw(ships, ship_types, drones, num_grids=self.game.width, size=self.game.width*50)

        return img
