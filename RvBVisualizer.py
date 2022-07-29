import cv2
from ship import *
from drone import *


def get_midpoint(x1, y1, x2, y2):

    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw(ships, ship_types, drones, size=500, num_grids=10):

    grid_width = size / num_grids

    water = (250, 145, 84)
    gray = (198, 192, 191)
    black = (0, 0, 0)
    red = (0, 0, 255)
    light_red = (203, 204, 255)
    light_blue = (230, 216, 173)
    dark_blue = (139, 0, 0)
    light_gray = (192, 192, 192)

    # TODO: perhaps images should be made one time and kept around instead of redoing it
    ddg_img = cv2.imread('./imgs/ddg.png')
    ddg_size = (int(len(ddg_img[0]) * 45 / len(ddg_img[0])), int(len(ddg_img) * 45 / len(ddg_img[0])))
    ddg_img = cv2.resize(ddg_img, ddg_size)
    ddg_img[ddg_img > 10] = 255
    cv2.floodFill(ddg_img, None, seedPoint=(1, 1), newVal=light_gray)

    carrier_img = cv2.imread('./imgs/carrier.png')
    carrier_size = (int(len(carrier_img[0]) * 45 / len(carrier_img[0])), int(len(carrier_img) * 45 / len(carrier_img[0])))
    carrier_img = cv2.resize(carrier_img, carrier_size)
    carrier_img[carrier_img > 10] = 255
    cv2.floodFill(carrier_img, None, seedPoint=(1, 1), newVal=light_gray)
    cv2.floodFill(carrier_img, None, seedPoint=(1, carrier_size[1]-1), newVal=light_gray)
    cv2.floodFill(carrier_img, None, seedPoint=(carrier_size[0] - 1, carrier_size[1] - 1), newVal=light_gray)

    drone_img = cv2.imread('./imgs/drone.jpeg')
    drone_size = (int(len(drone_img[0]) * 45 / len(drone_img[0])), int(len(drone_img) * 45 / len(drone_img[0])))
    drone_img = cv2.resize(drone_img, drone_size)
    drone_img[drone_img > 10] = 255
    cv2.floodFill(drone_img, None, seedPoint=(1, 1), newVal=light_red)

    # initialize our canvas as a 300x300 pixel image with 3 channels
    # (Red, Green, and Blue) with a black background
    canvas = np.zeros((size, size, 3), dtype="uint8")

    # draw the water
    cv2.rectangle(canvas, (0, 0), (size, size), water, -1)

    for drone in drones:
        cv2.circle(canvas, (int(drone.y * grid_width + grid_width / 2), int(drone.x * grid_width + grid_width / 2)), int(grid_width * drone.get_detection_radius()), light_red, -1)

    for ship, ship_type in zip(ships, ship_types):
        cv2.circle(canvas, (int(ship.y * grid_width + grid_width / 2), int(ship.x * grid_width + grid_width / 2)), int(grid_width * ship.sense_radius), light_blue, -1)

    for ship, ship_type in zip(ships, ship_types):
        cv2.circle(canvas, (int(ship.y * grid_width + grid_width / 2), int(ship.x * grid_width + grid_width / 2)), int(grid_width * ship.comms_radius), light_gray, -1 )

    # draw the grid
    cv2.rectangle(canvas, (0, 0), (size, size), (0, 0, 0))
    for i in range(num_grids):
        starts = int(i * grid_width)
        cv2.line(canvas, (starts-1, 0), (starts-1, size), black)
        cv2.line(canvas, (starts, 0), (starts, size), gray)
        cv2.line(canvas, (0, starts-1), (size, starts-1), black)
        cv2.line(canvas, (0, starts), (size, starts), gray)

    # draw ships
    for ship, ship_type in zip(ships, ship_types):
        if ship_type == 'ddg':
            ship_img = ddg_img
            ship_size = ddg_size
        else:
            ship_img = carrier_img
            ship_size = carrier_size

        heading = Ship.dirs[ship.heading_idx]
        arrow_start = (int(ship.y * grid_width + grid_width / 2), int(ship.x * grid_width + grid_width / 2))
        arrow_end = get_midpoint(arrow_start[0], arrow_start[1], arrow_start[0] + heading[1] * grid_width, arrow_start[1] + heading[0] * grid_width)
        cv2.arrowedLine(canvas, arrow_start, arrow_end, black, 1)

        ship_x = int(ship.x * grid_width + (grid_width - ship_size[1]) / 2)
        ship_y = int(ship.y * grid_width + (grid_width - ship_size[0]) / 2)
        canvas[ship_x:ship_x+ship_size[1], ship_y:ship_y+ship_size[0]] = ship_img

    for ship, ship_type in zip(ships, ship_types):
        cv2.circle(canvas, (int(ship.y * grid_width + grid_width / 2), int(ship.x * grid_width + grid_width / 2)), int(grid_width * ship.comms_radius), (0, 0, 0))
        cv2.circle(canvas, (int(ship.y * grid_width + grid_width / 2), int(ship.x * grid_width + grid_width / 2)), int(grid_width * ship.sense_radius), dark_blue)

    for drone in drones:
        drone_x = int(drone.x * grid_width + (grid_width - drone_size[1]) / 2)
        drone_y = int(drone.y * grid_width + (grid_width - drone_size[0]) / 2)

        cv2.circle(canvas, (int(drone.y * grid_width + grid_width / 2), int(drone.x * grid_width + grid_width / 2)), int(grid_width * drone.get_detection_radius()), red, thickness=2)

        if drone.x >= 0 and drone.y >= 0 and drone.x < num_grids and drone.y < num_grids:
            canvas[drone_x:drone_x + drone_size[1], drone_y:drone_y + drone_size[0]] = drone_img

    for ship, ship_type in zip(ships, ship_types):
        if ship_type == 'ddg':
            text = 'D'
        else:
            text = 'S'

        cv2.circle(canvas, (int(ship.goal[1] * grid_width + grid_width / 2), int(ship.goal[0] * grid_width + grid_width / 2)), int(grid_width * 0.4), (255, 255, 255), -1)
        cv2.circle(canvas, (int(ship.goal[1] * grid_width + grid_width / 2), int(ship.goal[0] * grid_width + grid_width / 2)), int(grid_width * 0.4), dark_blue, thickness=2)
        origin = (int(ship.goal[1] * grid_width + grid_width / 2.8), int(ship.goal[0] * grid_width + grid_width / 1.5))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


if __name__ == '__main__':

    ship1 = Ship(x=1, y=3, goal=(3, 7), sense_radius=2, comms_radius=1, heading=(1, 0), effector_range=2, effector_count=0)
    ship2 = Ship(x=2, y=5, goal=(8, 4), sense_radius=2, comms_radius=1, heading=(1, 1), effector_range=2, effector_count=0)
    drone = Drone(x=7, y=7, detection_radius=1, speed=2)

    ships = [ship1, ship2]
    ship_types = ['ddg', 'sag']
    drones = [drone]

    img = draw(ships, ship_types, drones)

    cv2.imwrite('sim.png', img)
    cv2.imshow("Canvas", img)
    cv2.waitKey(0)
