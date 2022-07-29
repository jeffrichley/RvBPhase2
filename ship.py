import numpy as np


class Ship:
    dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    commands = {'q': 'quit simulation', 'a': 'turn left', 'w': 'keep straight', 'd': 'turn right',
                'z': 'toggle sensor', 'x': 'enable effector', 'c': 'toggle comms', 'v': 'launch missile'}

    def __init__(self, x, y, goal, sense_radius, comms_radius, missile_range, missile_count,
                 missile_accuracy, jam_radius, jam_time, heading=None):
        self.number = Ship.ship_number
        Ship.ship_number += 1
        self.x = x
        self.y = y
        self.goal = goal
        self.sense_radius = sense_radius
        self.comms_radius = comms_radius
        self.missile_range = missile_range
        self.missile_count = missile_count
        self.missile_accuracy = missile_accuracy
        self.jam_radius = jam_radius
        self.jam_time = jam_time
        self.sensing_on = True
        self.comms_on = True
        self.effector_on = False
        self.heading_idx = self.get_heading_idx(heading)

    def get_number(self):
        return self.number

    def get_location(self):
        return self.x, self.y

    def get_goal(self):
        return self.goal

    def get_sense_radius(self):
        return self.sense_radius

    def get_comms_radius(self):
        return self.comms_radius / (1 if self.comms_on else 2)

    def get_missile_range(self):
        return self.missile_range

    def get_missile_accuracy(self):
        return self.missile_accuracy

    def get_jam_radius(self):
        return self.jam_radius

    def get_jam_time(self):
        return self.jam_time

    """
        Will be used to see if an object is in range of this ship.
        
        r is self.comms_radius when the object is a ship or self.sense_radius when object is another drone
    """
    def in_range(self, x, y, r):
        r = min(r, self.comms_radius)
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5 <= r

    """
        The ML gym needs an input of direction to goal that doesn't give the full magnitude, so this function 
        will calculate and return that unit vector.
    """
    def get_unit_dir_to_goal(self):
        x_dif = self.goal[0] - self.x
        y_dif = self.goal[1] - self.y
        r = (x_dif ** 2 + y_dif ** 2) ** 0.5

        answer = (0, 0)
        if r != 0:
            answer = (x_dif / r, y_dif / r)

        return answer

    """
        Find which ships are in comms range with given ship
    """
    def get_ships_in_range(self, active_ships):
        connected = []
        to_check = [self]
        while to_check:
            current = to_check.pop(0)
            if current in connected:
                continue

            connected.append(current)
            c_x, c_y = current.get_location()
            for s in [s for s in active_ships if s not in connected]:
                if s.in_range(c_x, c_y, self.comms_radius):
                    to_check.append(s)

        return connected

    def get_viewsize(self):
        return 2 * (self.sense_radius + self.comms_radius * (Ship.ship_number - 1)) + 1

    """
        Create a representation of what the ship can "see" based on the sensing radius 
        combined with all views of ships within comms range
    """
    def create_viewport(self, active_ships, drones, unknown_char='?', safe_char='.', danger_char='*', goal_char='$'):
        # get maximum range based on sensing range, comms range, and number of ships
        max_r = self.sense_radius + self.comms_radius * (Ship.ship_number - 1)
        view_w = view_h = 2 * max_r + 1

        viewport = np.array([unknown_char] * view_w * view_h, dtype=str).reshape(view_h, view_w)

        # check which ships are in comms range
        ships_in_range = self.get_ships_in_range(active_ships)

        my_x, my_y = self.get_location()

        # handle drones when in range
        for drone in drones:
            x, y = drone.get_location()

            # if the drone is seen by any of the ships within comms range
            if any([s.in_range(x, y, s.get_sense_radius()) for s in ships_in_range]):
                r = drone.get_detection_radius()

                # make x/y coordinates relative to ship and fill in viewport
                x += max_r - my_x
                y += max_r - my_y
                for i in range(-r, r+1):
                    for j in range(-r, r+1):
                        if 0 <= x+i < view_w and 0 <= y+j < view_h:
                            viewport[y+j, x+i] = danger_char

        # plot ships within comms range
        for ship in ships_in_range:
            relative_x, relative_y = ship.get_location()
            relative_y += max_r - my_y
            relative_x += max_r - my_x
            local_r = ship.get_sense_radius()
            for i in range(-local_r, local_r+1):
                for j in range(-local_r, local_r+1):
                    w = relative_x+i
                    h = relative_y+j
                    if 0 <= w < view_w and 0 <= h < view_h and viewport[h, w] == unknown_char:
                        viewport[h, w] = safe_char
            if 0 <= relative_x < view_w and 0 <= relative_y < view_h:
                viewport[relative_y, relative_x] = str(ship.get_number())

        # handle goal when in range
        x, y = self.get_goal()
        x += max_r - my_x
        y += max_r - my_y
        if 0 <= x < view_w and 0 <= y < view_h:
            viewport[y, x] = goal_char

        viewport[max_r, max_r] = str(self.get_number())

        return viewport

    def get_info(self):
        heading_str = '(%d, %d)' % Ship.dirs[self.heading_idx]
        goal_str = '(%.2f, %.2f)' % (self.get_unit_dir_to_goal())

        status = 'ship %d @ (%d, %d) facing %s with goal of %s' % (self.number, self.x, self.y, heading_str, goal_str)
        status += ', comms on' if self.comms_on else ''
        status += ', sensor on' if self.sensing_on else ''
        status += ', effector on' if self.effector_on else ''

        return status

    def arrived_at_destination(self):
        return (self.x, self.y) == self.goal

    @staticmethod
    def get_heading_idx(heading):
        if heading is None:
            return 0
        check = [i for i, d in enumerate(Ship.dirs) if d == heading]
        return 0 if not check else check[0]

    """
        Directly update x and y coordinates
    """
    def place(self, x, y, heading=None):
        self.x = x
        self.y = y
        self.heading_idx = self.get_heading_idx(heading)

    """
        Move the ship based on the current location, speed, and heading
    """
    def move(self, left=False, right=False):
        if left and right:
            # print("Cannot turn left and right. Check your input")
            return
        self.heading_idx += -1 if left else 1 if right else 0
        self.heading_idx %= len(Ship.dirs)

        movement_direction = Ship.dirs[self.heading_idx]
        self.x += movement_direction[0]
        self.y += movement_direction[1]

    @staticmethod
    def print_command_list():
        [print(letter, ' => ', command) for letter, command in Ship.commands.items()]

    def prompt_user_for_command(self):
        command = input('Give command for ship %d (h to print available commands): ' % self.number)
        if command == '' or command not in 'qawdzxc':
            if command == 'h':
                self.print_command_list()
            elif command == 'v':
                if self.missile_count <= 0:
                    print("Out of missiles -- try giving a different command")
                else:
                    return command
            else:
                print('%s is not a valid command' % command)
            return self.prompt_user_for_command()
        return command

    """
        Change state of the ship based on the command given
        
        Returns -1 if the simulation should end, 0 to keep going
    """
    def handle_command(self, command):
        # ***** always disable effector just in case it's on ******
        self.effector_on = False

        if command == 'q':
            print("Ship %d decided to quit" % self.number)
        elif command == 'w':
            self.move()
        elif command == 'a':
            self.move(left=True)
        elif command == 'd':
            self.move(right=True)
        elif command == 'z':
            self.sensing_on = not self.sensing_on
            self.move()
        elif command == 'x':
            self.sensing_on = self.comms_on = False
            self.effector_on = True
            self.move()
        elif command == 'c':
            self.comms_on = not self.comms_on
            self.move()
        elif command == 'v':
            if self.missile_count:
                # print("Ship %d fired missile!" % self.number)
                self.missile_count -= 1

                self.move()
            # else:
            #     print("Not enough missiles to fire -- try again")
            #     return self.prompt_and_handle()
        else:
            print('Not sure how you gave an invalid command, but here we are trying to do something undefined')
        return command

    def prompt_and_handle(self):
        return self.handle_command(self.prompt_user_for_command())


    # added by Jeff for the observation
    def create_dimentional_viewport(self, active_ships, drones, number_of_ships, unknown_char='?', safe_char='.', danger_char='*', goal_char='$'):

        # get maximum range based on sensing range, comms range, and number of ships
        # max_r = self.sense_radius + self.comms_radius * (Ship.ship_number - 1)
        max_r = self.sense_radius + self.comms_radius * (number_of_ships - 1)
        view_w = view_h = 2 * max_r + 1

        viewport = np.array([unknown_char] * view_w * view_h, dtype=str).reshape(view_h, view_w)
        dim_viewport = np.zeros((view_h, view_w, 5))

        # layers
        layer_self = 0
        layer_friends = 1
        layer_danger = 2
        layer_unknown = 3
        layer_goal = 4

        # assume everything is uknown
        dim_viewport[:, :, layer_unknown] = 1

        # check which ships are in comms range
        ships_in_range = self.get_ships_in_range(active_ships)
        my_x, my_y = self.get_location()

        # handle drones when in range
        for drone in drones:
            x, y = drone.get_location()

            # if the drone is seen by any of the ships within comms range
            if any([s.in_range(x, y, s.get_sense_radius()) for s in ships_in_range]):
                r = drone.get_detection_radius()

                # make x/y coordinates relative to ship and fill in viewport
                x += max_r - my_x
                y += max_r - my_y
                for i in range(-r, r + 1):
                    for j in range(-r, r + 1):
                        if 0 <= x + i < view_w and 0 <= y + j < view_h:
                            viewport[y + j, x + i] = danger_char
                            dim_viewport[y + j, x + i, layer_danger] = 1
                            dim_viewport[y + j, x + i, layer_unknown] = 0

        # plot ships within comms range
        for ship in ships_in_range:
            relative_x, relative_y = ship.get_location()
            relative_y += max_r - my_y
            relative_x += max_r - my_x
            local_r = ship.get_sense_radius()
            for i in range(-local_r, local_r + 1):
                for j in range(-local_r, local_r + 1):
                    w = relative_x + i
                    h = relative_y + j
                    if 0 <= w < view_w and 0 <= h < view_h:
                        if viewport[h, w] == unknown_char:
                            viewport[h, w] = safe_char

                            dim_viewport[h, w, layer_danger] = 0
                            dim_viewport[h, w, layer_unknown] = 0

            if 0 <= relative_x < view_w and 0 <= relative_y < view_h:
                viewport[relative_y, relative_x] = str(ship.get_number())

                if ship.get_number() != self.get_number():
                    dim_viewport[relative_y, relative_x, layer_friends] = 1
                    dim_viewport[relative_y, relative_x, layer_unknown] = 0

        # handle goal when in range
        x, y = self.get_goal()
        x += max_r - my_x
        y += max_r - my_y
        if 0 <= x < view_w and 0 <= y < view_h:
            viewport[y, x] = goal_char
            dim_viewport[y, x, layer_goal] = 1
            dim_viewport[y, x, layer_unknown] = 0

        viewport[max_r, max_r] = str(self.get_number())
        dim_viewport[max_r, max_r, layer_self] = 1
        dim_viewport[max_r, max_r, layer_unknown] = 0

        return viewport, dim_viewport


Ship.ship_number = 0


