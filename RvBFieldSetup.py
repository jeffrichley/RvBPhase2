from random import randint, random
from math import sqrt

from ship import Ship


def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance_tuples(one, two):
    return distance(one[0], one[1], two[0], two[1])


def set_field(game, goal_distance=4):
    size = game.width

    ship1 = game.active_ships[0]
    ship2 = game.active_ships[1]

    # put the ships around the edges
    val = randint(0, size-2)

    # does this go on the top or the left?
    if random() < 0.5:
        x, y = val, 0
        x2, y2 = val + 1, 0
        # should it be top or bottom
        if random() < 0.5:
            y += size - 1
            y2 += size - 1
    else:
        x, y = 0, val
        x2, y2 = 0, val + 1
        # should it be on the right or left?
        if random() < 0.5:
            x += size - 1
            x2 += size - 1

    # set the ship values
    ship1.x = x
    ship1.y = y
    ship2.x = x2
    ship2.y = y2

    # set the goals
    goal1 = ship1.goal
    goal2 = ship2.goal

    # find a place for ship1 goal
    while goal1 == ship1.goal or distance(goal1[0], goal1[1], ship1.x, ship1.y) < goal_distance:
        goal1 = (randint(1, size - 2), randint(1, size - 2))

    # find a place for ship2 goal
    while goal1 == goal2 or goal2 == ship2.goal or distance(goal2[0], goal2[1], ship2.x, ship2.y) < goal_distance:
        # goal2 = (randint(1, 8), randint(1, 8))
        goal2 = (randint(1, size - 2), randint(1, size - 2))

    # set the goals
    ship1.goal = goal1
    ship2.goal = goal2

    # set the ship's heading pointed toward the goal
    dirx1, diry1 = ship1.get_unit_dir_to_goal()
    dirx1 = round(dirx1)
    diry1 = round(diry1)

    dirx2, diry2 = ship2.get_unit_dir_to_goal()
    dirx2 = round(dirx2)
    diry2 = round(diry2)

    ship1.heading_idx = Ship.dirs.index((dirx1, diry1))
    ship2.heading_idx = Ship.dirs.index((dirx2, diry2))

    # add the drone away from the goal and ships
    # this random selection and trying over again feels clunky.  in 10,000,000 runs, it averages 2 tries
    for drone in game.drones:
        drone_x, drone_y = ship1.x, ship1.y
        while distance(drone_x, drone_y, ship1.x, ship1.y) < 3 or \
                distance(drone_x, drone_y, ship2.x, ship2.y) < 3 or \
                distance(drone_x, drone_y, ship1.goal[0], ship1.goal[1]) < 2 or \
                distance(drone_x, drone_y, ship2.goal[0], ship2.goal[1]) < 2:

            drone_x, drone_y = randint(1, size - 2), randint(1, size - 2)

        # drone = game.drones[0]
        drone.x = drone_x
        drone.y = drone_y


if __name__ == '__main__':
    from RvBEnvironment import RvBEnvironment
    import time

    env = RvBEnvironment()
    env_game = env.game

    start = time.time()
    set_field(env_game, goal_distance=8)
    end = time.time()

    print(f'set_field took {end - start}')

    for ship in env_game.active_ships:
        print(f'ship {ship.number} ({ship.x:2}, {ship.y:2}) with goal {ship.goal} and heading {Ship.dirs[ship.heading_idx]}')

    print(f'drone {env_game.drones[0].x}, {env_game.drones[0].y}')

    for i in range(2):
        ship = env_game.active_ships[i]
        drone = env_game.drones[0]
        print(f'Ship {i}')
        print(f'\tto goal is {distance_tuples((ship.x, ship.y), ship.goal)}')
        print(f'\tto drone is {distance_tuples((ship.x, ship.y), (drone.x, drone.y))}')
        print(f'\tdrone to goal is {distance_tuples(ship.goal, (drone.x, drone.y))}')

    rendered = env.render()

    import cv2

    cv2.imshow("Canvas", rendered)
    cv2.waitKey(0)

    # max_tries = -1
    # all_tries = []
    # for i in range(10000000):
    #     tries = set_field(env_game)
    #     all_tries.append(tries)
    #     if tries > max_tries:
    #         max_tries = tries
    #         print(i, tries)
    #
    # print('average tries {sum(all_tries) / len(all_tries)}')
