from carla_objects import *


def calc_yaw_diff(l1: carla.Location, v: Vehicle) -> float:
    yaw_path = np.arctan2(l1.y - v.get_location().y, l1.x - v.get_location().x)
    diff = yaw_path - v.get_transform().rotation.yaw * 2 * np.pi / 360
    if diff > np.pi:
        diff -= 2 * np.pi
    if diff < -np.pi:
        diff += 2 * np.pi
    return diff


try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race_1", (640, 480))

    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3", debug=False)
        runner.auto_pilot = True

        # Global Planner.
        waypoint_distance = 1.0
        distance_error = 0.5
        waypoints = world.map.generate_waypoints(waypoint_distance)
        target = np.argmin([runner.get_location().distance(w.transform.location) for w in waypoints])
        start_waypoint = waypoints[target]
        target = 0
        waypoints = [start_waypoint.next(waypoint_distance)[0]]
        while start_waypoint.transform.location.distance(waypoints[-1].transform.location) >= waypoint_distance - distance_error:
            waypoints.append(waypoints[-1].next(waypoint_distance)[0])
        for w in waypoints:
            world.debug_string(w.transform.location, "x")
            time.sleep(0.01)

        pre_yaw_diff: float = 0.0
        sum_yaw_diff: float = 0.0
        while not runner.has_collided and not world.is_done:
            curr_waypoint = world.map.get_waypoint(runner.get_location())
            next_waypoint = waypoints[target]

            distance = runner.get_location().distance(next_waypoint.transform.location)
            if distance < distance_error:
                target = (target + 1) % len(waypoints)
            world.debug_string(next_waypoint.transform.location, "X")

            yaw_diff = calc_yaw_diff(next_waypoint.transform.location, runner)
            max_future_yaw_diff = 0
            for i in range(20):
                future_target_1 = (target + i) % len(waypoints)
                future_target_2 = (target + i + 1) % len(waypoints)
                future_target_3 = (target + i + 1) % len(waypoints)
                future_waypoint_1 = waypoints[future_target_1]
                future_waypoint_2 = waypoints[future_target_2]
                future_waypoint_3 = waypoints[future_target_2]
                yaw_1 = np.arctan2(
                    future_waypoint_1.transform.location.y - future_waypoint_2.transform.location.y,
                    future_waypoint_1.transform.location.x - future_waypoint_2.transform.location.x
                )
                yaw_2 = np.arctan2(
                    future_waypoint_2.transform.location.y - future_waypoint_3.transform.location.y,
                    future_waypoint_2.transform.location.x - future_waypoint_3.transform.location.x
                )
                future_yaw_diff = yaw_1 - yaw_2
                if future_yaw_diff > np.pi:
                    future_yaw_diff -= 2 * np.pi
                if future_yaw_diff < -np.pi:
                    future_yaw_diff += 2 * np.pi
                if abs(future_yaw_diff) > max_future_yaw_diff:
                    max_future_yaw_diff = abs(future_yaw_diff)
                world.debug_string(future_waypoint_3.transform.location, "O", color=(255, 255, 0))

            events = world.key_handler()
            runner.key_handler(events)
            if runner.auto_pilot:
                runner.brake = 0
                runner.steer = max(-1, min(1, 1.5 * yaw_diff + 0.001 * sum_yaw_diff + 18.5 * (yaw_diff - pre_yaw_diff)))
                runner.throttle = min(1, 1 - (0.7 * max_future_yaw_diff + 0.7 * abs(runner.steer)) * runner.speed_kmh() / 45)
                if runner.throttle < 0:
                    runner.brake = -runner.throttle
                    runner.throttle = 0
                if runner.throttle < 0.5 and runner.speed_kmh() < 20:
                    runner.throttle = 1
                    runner.brake = 0
                if runner.speed_kmh() > 45:
                    runner.throttle = 0
                    runner.brake = 0.1
                pre_yaw_diff = yaw_diff
                sum_yaw_diff = max(-100, min(100, sum_yaw_diff + yaw_diff))
            runner.action()

            world.render_image(runner.rgb_image)
            world.render_text("%.1ffps" % world.clock.get_fps(), (10, 10))
            world.render_text("%.2fkm/h" % runner.speed_kmh(), (10, 30))
            world.render_text("Throttle: %.2f" % runner.throttle, (10, 50))
            world.render_text("Brake: %.2f" % runner.brake, (10, 70))
            world.render_text("Steer: %.2f" % runner.steer, (10, 90))
            world.render_text("Reverse: %d" % runner.reverse, (10, 110))
            world.render_text("L: %.2f, R: %.2f" % (runner.distance_left, runner.distance_right), (10, 130))
            world.render_text("Auto Pilot: " + str(runner.auto_pilot), (10, 150))
            world.redraw_display()
        print(pre_yaw_diff, sum_yaw_diff)
        runner.destroy()
except Exception as e:
    print(e)
finally:
    print("done.")
