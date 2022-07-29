from math import sqrt
from heapq import heappush, heappop
from copy import copy

from RvB_env import GameBoard, Ship
from RvBFieldSetup import set_field

viable_headings = {
    (-1, -1): [( 0, -1), (-1, -1), (-1,  0)],
    ( 0, -1): [( 1, -1), ( 0, -1), (-1, -1)],
    ( 1, -1): [( 1,  0), ( 1, -1), ( 0, -1)],
    ( 1,  0): [( 1,  1), ( 1,  0), ( 1, -1)],
    ( 1,  1): [( 0,  1), ( 1,  1), ( 1,  0)],
    ( 0,  1): [(-1,  1), ( 0,  1), ( 1,  1)],
    (-1,  1): [(-1,  0), (-1,  1), ( 0,  1)],
    (-1,  0): [(-1, -1), (-1,  0), (-1,  1)]
}


def a_star(game):
    ships = game.active_ships

    game_height = game.height
    game_width = game.width

    p1_path = []
    p2_path = []

    for idx, ship in enumerate(ships):

        queue = []
        visited = []

        distance = sqrt((ship.goal[0] - ship.x) ** 2 + (ship.goal[1] - ship.y) ** 2)
        heading = Ship.dirs[ship.heading_idx]
        entry = (distance, (ship.x, ship.y), heading, 0, [])
        heappush(queue, entry)
        visited.append((ship.x, ship.y, heading[0], heading[1]))

        goal = ship.goal
        drone = game.drones[0]
        dx = drone.x
        dy = drone.y

        if goal == (dx + 1, dy) or \
           goal == (dx - 1, dy) or \
           goal == (dx, dy + 1) or \
           goal == (dx, dy - 1) or \
           goal == (dx, dy):

            return None, None

        count = 0

        while len(queue) > 0:
            count += 1
            _, (ship_x, ship_y), heading, previous_distance, action_list = heappop(queue)

            visited.append((ship_x, ship_y, heading[0], heading[1]))

            if (ship_x, ship_y) == goal:
                if idx == 0:
                    p1_path = action_list
                else:
                    p2_path = action_list

                break

            for action_idx, direction in enumerate(viable_headings[heading]):
                new_ship_x = ship_x + direction[0]
                new_ship_y = ship_y + direction[1]

                if new_ship_x < 0 or new_ship_y < 0 or new_ship_x >= game_width or new_ship_y >= game_height:
                    continue

                distance = sqrt((ship.goal[0] - new_ship_x) ** 2 + (ship.goal[1] - new_ship_y) ** 2)
                travelled = previous_distance + 1

                new_ship_location = (new_ship_x, new_ship_y)

                if new_ship_location == (dx + 1, dy) or \
                   new_ship_location == (dx - 1, dy) or \
                   new_ship_location == (dx, dy + 1) or \
                   new_ship_location == (dx, dy - 1) or \
                   new_ship_location == (dx, dy):

                    continue

                new_action_list = copy(action_list)
                new_action_list.append(action_idx)

                if (new_ship_x, new_ship_y, direction[0], direction[1]) not in visited:
                    entry = (distance + previous_distance, (new_ship_x, new_ship_y), direction, travelled, new_action_list)
                    heappush(queue, entry)

    return p1_path, p2_path


def test(num):

    num_bad = 0

    for i in range(100):
        config = {'width': 10, 'height': 10, 'num_ships': 2, 'num_drones': 1, 'random_start_end': True}
        game = GameBoard(**config)
        set_field(game)

        ddg_path, sag_path = a_star(game)

        if ddg_path is None or len(ddg_path) == 0 or sag_path is None or len(sag_path) == 0:
            num_bad += 1

    return num_bad


if __name__ == '__main__':

    # bad_count = 0
    # for i in range(100000):
    #     num_bad = test(i)
    #     bad_count += num_bad
    #     if num_bad > 0:
    #         print(i, num_bad, bad_count)

    from multiprocessing import Pool, freeze_support
    nums = [x for x in range(10000)]
    with Pool() as pool:
        results = pool.map(test, nums)
    print('bad results', sum(results))

    # out of 1,000,000 and default field layout, 355324 were bad
    # out of 1,000,000 and augmented field layout, 0 were bad
