from carla_objects import *


def calc_yaw(l1: carla.Location, l2: carla.Location) -> float:
    rad = np.arctan2(l2.y - l1.y, l2.x - l1.x)
    if rad < -np.pi:
        rad += 2 * np.pi
    if rad > np.pi:
        rad -= 2 * np.pi
    return rad


def calc_vehicle_yaw(v: Vehicle) -> float:
    rad = v.get_transform().rotation.yaw * 2 * np.pi / 360
    if rad < -np.pi:
        rad += 2 * np.pi
    if rad > np.pi:
        rad -= 2 * np.pi
    return rad


def calc_yaw_diff(yaw1: float, yaw2: float) -> float:
    diff = yaw1 - yaw2
    if diff < -np.pi:
        diff += 2 * np.pi
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff


def go_ahead_same_land_until_end(
        world: World,
        initial_location: carla.Location,
        waypoint_distance: float = 1.0,
        distance_error: float = 0.5
) -> list:
    waypoints = world.map.generate_waypoints(waypoint_distance)
    target = np.argmin([initial_location.distance(w.transform.location) for w in waypoints])
    start_waypoint = waypoints[target]
    next_waypoints = start_waypoint.next(waypoint_distance)
    if len(next_waypoints) == 0:
        return []
    waypoints = [next_waypoints[0]]
    while start_waypoint.transform.location.distance(
            waypoints[-1].transform.location) >= waypoint_distance - distance_error:
        next_waypoints = waypoints[-1].next(waypoint_distance)
        if len(next_waypoints) == 0:
            break
        waypoints.append(next_waypoints[0])
    return waypoints


def max_yaw_diff_in_the_future(waypoints: list, curr_index: int = 0, n: int = 10) -> float:
    check_list: list = []
    for i in range(n):
        check_list.append(waypoints[(curr_index + i) % len(waypoints)])

    max_e = 0
    for m in range(1, n - 1, 1):
        for i in range(0, m, 1):
            yaw1 = calc_yaw(check_list[i].transform.location, check_list[m].transform.location)
            for j in range(n - 1, m, -1):
                yaw2 = calc_yaw(check_list[m].transform.location, check_list[j].transform.location)
                e = calc_yaw_diff(yaw1, yaw2)
                if abs(e) > max_e:
                    max_e = abs(e)
    return max_e
