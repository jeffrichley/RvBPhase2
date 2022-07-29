import random
import enum


class Patterns(enum.Enum):
    not_implemented = -1
    still = 0
    straight = 1
    patrol_circle = 2
    patrol_square = 3
    grid_search = 4
    

def string_to_pattern(string):
    if string == 'still':
        return Patterns.still
    elif string == 'straight':
        return Patterns.straight
    elif string == 'patrol_circle':
        return Patterns.patrol_circle
    elif string == 'patrol_square':
        return Patterns.patrol_square
    elif string == 'grid_search':
        return Patterns.grid_search
    return Patterns.not_implemented


class Drone:
    circle_dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    square_dirs = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    def __init__(self, x, y, detection_radius, speed, flight_pattern=Patterns.still):
        self.number = Drone.number
        Drone.number += 1
        self.x = x
        self.y = y
        self.detection_radius = detection_radius
        self.speed = speed
        self.flight_pattern = string_to_pattern(flight_pattern)
        self.jammed = 0
        self.clockwise = random.random() < 0.5
        self.heading = (0, 0)
        self.patrol_idx = 0
        self.get_heading()

    def get_number(self):
        return self.number

    def get_location(self):
        return self.x, self.y

    def get_distance(self, x, y):
        return ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5

    def get_detection_radius(self):
        return self.detection_radius // (2 if self.jammed > 0 else 1)

    """
        Query the drone to see if a location is within its detection radius
        
        Inputs (float/int): x and y coordinates of a location, comms radius (larger if on)
        Outputs (bool): whether or not the location is visible to the drone
    """
    def within_radius(self, x, y, comms_radius=0):
        dist = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
        effective_radius = (self.detection_radius + comms_radius) / 2 if self.jammed > 0 else 1
        return dist <= effective_radius

    """
        Put the drone in a location instead of using the movement pattern defined
    """
    def place(self, x, y):
        self.x = x
        self.y = y

    """
        Determine how to move based on flight pattern and take the appropriate number of steps. Each step reduces the 
        time left jammed by 1.
    """
    def move(self, w, h):
        self.jammed = max(0, self.jammed - 1)
        if self.flight_pattern == Patterns.still:
            return 0
        elif self.flight_pattern == Patterns.straight:
            self.x += self.heading[0]
            self.y += self.heading[1]

            if self.x < 0 or self.x >= w:
                self.heading = (-self.heading[0], self.heading[1])
            if self.y < 0 or self.y >= h:
                self.heading = (self.heading[0], -self.heading[1])
        elif self.flight_pattern in [Patterns.patrol_circle, Patterns.patrol_square]:
            self.x += self.heading[0] * self.speed
            self.y += self.heading[1] * self.speed
            self.get_heading()
        else:
            print("Flight pattern not implemented yet")
            return -1

    """
        Placeholder function for setting how the drone will be affected (ship weapons, weather, etc)
    """
    def affect(self, jam_time):
        self.jammed = max(0, jam_time)

    """
        Determine heading based on position, anchor, and pattern
    """
    def get_heading(self):
        if self.flight_pattern == Patterns.still:
            self.heading = (0, 0)
        elif self.flight_pattern == Patterns.straight:
            self.heading = random.randint(-self.speed, self.speed), random.randint(-self.speed, self.speed)
            if self.heading == (0, 0):
                self.get_heading()
        elif self.flight_pattern == Patterns.patrol_circle:
            self.heading = Drone.circle_dirs[self.patrol_idx % len(Drone.circle_dirs)]
            self.patrol_idx += 1 if self.clockwise else -1
        elif self.flight_pattern == Patterns.patrol_square:
            self.heading = Drone.square_dirs[self.patrol_idx % len(Drone.square_dirs)]
            self.patrol_idx += 1 if self.clockwise else -1


Drone.number = 0
