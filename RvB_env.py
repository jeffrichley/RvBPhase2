import configparser
from drone import Drone
import numpy as np
import os
from random import randint, random
from ship import Ship
import shutil


class GameBoard:
    def __init__(self, width, height, num_ships, num_drones, random_start_end, sense_radius, comms_radius,
                 missile_range, missile_count, missile_accuracy, jam_radius, jam_time, detection_radius,
                 speed, pattern, print_type, sequential):
        self.finished = False
        self.width = width
        self.height = height
        self.print_type = print_type
        self.sequential = sequential
        self.num_ships = max(num_ships, 1)
        self.num_drones = max(num_drones, 0)

        self.found_ships = []
        self.ships_at_dest = []
        self.active_ships = self.generate_ships(random_start_end, sense_radius, comms_radius, missile_range,
                                                missile_count, missile_accuracy, jam_radius, jam_time)
        self.drones = self.generate_drones(detection_radius, speed, pattern)

    def generate_ships(self, random_start_end, sense_radius, comms_radius, missile_range,
                       missile_count, missile_accuracy, jam_radius, jam_time):
        ships = []
        goal = self.width-1, self.height-1
        if not random_start_end:
            start = (1, 1)
            heading = (1, 1)
        else:
            start = randint(0, self.width-1), randint(0, self.height-1)
            goal = (self.width-1, self.height-1)
            # start with a heading towards the center of the map in case we're on the edge and might fall off
            hor_third = start[0] * 3 // self.width
            vert_third = start[1] * 3 // self.height
            heading = (hor_third - 1, vert_third - 1)

        for i in range(self.num_ships):
            x = start[0] + i
            y = start[1]
            my_goal = goal[0]-x, goal[1]-y
            new_ship = Ship(x, y, my_goal, sense_radius, comms_radius, missile_range,
                            missile_count, missile_accuracy, jam_radius, jam_time)
            ships.append(new_ship)

        return ships

    def generate_drones(self, drone_det_r, drone_speed, drone_pattern):
        drones = []
        for i in range(self.num_drones):
            x = randint(0, self.width-1)
            y = randint(0, self.height-1)
            drones.append(Drone(x, y, drone_det_r, drone_speed, drone_pattern))

        return drones

    def print_board(self):
        board = np.array([' '] * self.height * self.width, dtype=str).reshape(self.height, self.width)
        for ship in self.active_ships:
            x, y = ship.get_location()
            if 0 <= x < self.width and 0 <= y < self.height:
                board[y, x] = str(ship.get_number())
            x, y = ship.get_goal()
            board[y, x] = '$'
        for drone in self.drones:
            x, y = drone.get_location()
            r = drone.get_detection_radius()

            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    if 0 <= x+i < self.width and 0 <= y+j < self.height:
                        board[y+j, x+i] = '*'

        print('='*self.width)
        for row in board[::-1]:
            print(''.join(row))
        print('='*self.width)

    def display_viewport(self, ship):
        print(ship.get_info())
        viewport = ship.create_viewport(self.active_ships, self.drones)

        print((ship.get_viewsize() + 2) * "~")
        for row in viewport[::-1]:
            print('|' + ''.join(row) + '|')
        print((ship.get_viewsize() + 2) * "~")

    def fire_missile(self, ship):
        # determine closest drone to the given ship
        ship_x, ship_y = ship.get_location()
        distances = [(drone, drone.get_distance(ship_x, ship_y)) for drone in self.drones]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        target_drone, distance = sorted_distances[0]
        roll = random()
        if distance < ship.get_missile_range() and roll < ship.get_missile_accuracy():
            # print("Destroyed drone %d with a roll of %.2f" % (target_drone.get_number(), roll))
            self.drones.remove(target_drone)
            del target_drone
            return True
        else:
            # print("A roll of %.2f wasn't good enough. Drone %d lives!" % (roll, target_drone.get_number()))
            return False

    def jam_drones(self, ship):
        ship_x, ship_y = ship.get_location()
        distances = [(drone, drone.get_distance(ship_x, ship_y)) for drone in self.drones]
        drones_within_range = [drone for drone, dist in distances if dist < ship.get_jam_radius()]
        for drone in drones_within_range:
            drone.affect(ship.get_jam_time())
        # print('Affected %d drones, which will each be jammed for %d time steps' %
        #       (len(drones_within_range), ship.get_jam_time()))

    def move_ships_sequentially(self):
        for ship in self.active_ships:
            if self.print_type == 'ship':
                self.display_viewport(ship)

            # only triggered when user types 'q' to quit application:
            code = ship.prompt_and_handle()
            if code == 'q':
                return -1
            elif code == 'v':
                self.fire_missile(ship)
            elif code == 'x':
                self.jam_drones(ship)

        return 0

    def move_ships_simultaneously(self):
        moves = []
        for ship in self.active_ships:
            if self.print_type == 'ship':
                self.display_viewport(ship)

            # only triggered when user types 'q' to quit application:
            moves.append(ship.prompt_user_for_command())
            if moves[-1] == 'q':
                return -1
        for command, ship in zip(moves, self.active_ships):
            code = ship.handle_command(command)
            if code == 'v':
                self.fire_missile(ship)
            elif code == 'x':
                self.jam_drones(ship)

        return 0

    def check_ship_goals(self):
        for ship in self.active_ships:
            if ship.arrived_at_destination():
                self.ships_at_dest.append(ship)
                # print("Ship %d reached the goal!" % ship.get_number())
        for ship in self.ships_at_dest:
            if ship in self.active_ships:
                self.active_ships.remove(ship)

    def check_ships_caught(self):
        for i, drone in enumerate(self.drones):
            for ship in self.active_ships:
                if drone.within_radius(*ship.get_location(), ship.get_comms_radius()):
                    self.active_ships.remove(ship)
                    self.found_ships.append(ship)
                    # print("Drone %d found ship %d" % (i, ship.get_number()))

    def run_simulation(self):
        while self.active_ships:
            if self.print_type == 'board':
                # print("Active ships: ")
                # [print(ship.get_info()) for ship in self.active_ships]
                self.print_board()

            code = self.move_ships_sequentially() if self.sequential else self.move_ships_sequentially()
            if code == -1:
                self.active_ships = []
                break

            for drone in self.drones:
                drone.move(self.width, self.height)

            self.check_ship_goals()
            self.check_ships_caught()

        # print("ships detected: ", [ship.get_number() for ship in self.found_ships])
        # print("ships at destination: ", [ship.get_number() for ship in self.ships_at_dest])

    # added by Jeff for gym taking actions
    def env_step(self, ddg_info, sag_info):

        # move all units
        for ship in self.active_ships:
            command = None
            if ship.number == ddg_info[0]:
                command = ddg_info[1]
            elif ship.number == sag_info[0]:
                command = sag_info[1]

            # if ship.prompt_and_handle():
            #     self.active_ships = []
            #     break
            if command is not None:
                prev_ship_x, prev_ship_y = ship.get_location()

                ship.handle_command(command=command)

                if command == 'v':
                    self.fire_missile(ship)
                elif command == 'x':
                    self.jam_drones(ship)


                ship_x, ship_y = ship.get_location()
                if ship_x < 0 or ship_y < 0 or ship_x >= self.width or ship_y >= self.height:
                    ship.x = prev_ship_x
                    ship.y = prev_ship_y

                # print(f'{ship.get_number()} {(prev_ship_x, prev_ship_y)} -> {command} -> {ship.get_location()}')


        for drone in self.drones:
            drone.move(self.width, self.height)

        # check if someone found the goal
        for ship in self.active_ships:
            if ship.arrived_at_destination():
                self.ships_at_dest.append(ship)
                # print("Ship %d reached the goal!" % ship.get_number())
        for ship in self.ships_at_dest:
            if ship in self.active_ships:
                self.active_ships.remove(ship)

        # check if drones found ships
        for i, drone in enumerate(self.drones):
            for ship in self.active_ships:
                if ship.number != sag_info[0] and drone.within_radius(*ship.get_location()):
                    self.active_ships.remove(ship)
                    self.found_ships.append(ship)
                    # print("Drone %d found ship %d" % (i, ship.get_number()))


def read_config(filename):
    config = configparser.ConfigParser()
    if not os.path.exists(filename):
        shutil.copy(filename+'.example', filename)
    config.read(filename)

    env_vars = dict()
    env = config['Environment']
    env_vars['width'] = env.getint('width') or 30
    env_vars['height'] = env.getint('height') or 30
    env_vars['random_start_end'] = env.getboolean('random_start_end') or False
    env_vars['print_type'] = env['print_type'] or 'ship'
    env_vars['sequential'] = env.getboolean('sequential') or True

    ship = config['Ship']
    env_vars['num_ships'] = ship.getint('num_ships') or 1
    env_vars['sense_radius'] = ship.getint('sense_radius') or 3
    env_vars['comms_radius'] = ship.getint('comms_radius') or 1
    env_vars['missile_range'] = ship.getint('missile_range') or 2
    env_vars['missile_count'] = ship.getint('missile_count') or 1
    env_vars['missile_accuracy'] = ship.getfloat('missile_accuracy') or 1.0
    env_vars['jam_radius'] = ship.getint('jam_radius') or 1
    env_vars['jam_time'] = ship.getint('jam_time') or 1

    drone = config['Drone']
    env_vars['num_drones'] = drone.getint('num_drones') or 2
    env_vars['detection_radius'] = drone.getint('detection_radius') or 1
    env_vars['speed'] = drone.getint('speed') or 1
    env_vars['pattern'] = drone['pattern'] or 'still'

    # ---- sanity checks ----
    assert env_vars['num_ships'] > 0
    assert env_vars['sense_radius'] > env_vars['comms_radius'] + env_vars['detection_radius'], \
        'the sense radius needs to be larger or it is effectively worthless'
    assert env_vars['sense_radius'] > 0
    assert env_vars['comms_radius'] >= 0
    assert env_vars['detection_radius'] >= 0
    assert env_vars['jam_radius'] > env_vars['comms_radius'] + env_vars['detection_radius'], \
        'the jamming radius needs to be larger or it is effectively worthless'
    assert env_vars['jam_time'] >= 0
    assert env_vars['speed'] >= 0
    assert env_vars['missile_count'] >= 0
    assert env_vars['width'] > 0
    assert env_vars['height'] > 0

    if env.getboolean('print_config'):
        print_config(env_vars)

    return env_vars


def print_config(config):
    for k, v in config.items():
        spaces = 20 - len(k)
        print("%s%s%s" % (k, ' '*spaces, v))


if __name__ == '__main__':
    myconfig = read_config('../config.ini')
    game = GameBoard(**myconfig)
    game.run_simulation()

