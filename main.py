from carla_objects import *
from helper import *


try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race_1", (640, 480))

    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3", debug=False)
        box = carla.BoundingBox(
            runner.get_transform().location,
            carla.Vector3D(2, 2, 2)
        )
        world.draw_box(box, runner.get_transform().rotation, 0.5, life_time=90)
        runner.auto_pilot = True

        # -- Global Planner -- begin
        waypoint_distance: float = 2.0
        distance_error: float = 0.5
        waypoints = go_ahead_same_land_until_end(
            world,
            runner.get_location(),
            waypoint_distance,
            distance_error
        )
        curr_waypoint_index: int = 0

        for w in waypoints:
            world.draw_string(w.transform.location, "x")
            time.sleep(0.01)
        # -- Global Planner -- end

        t1 = time.time()
        e0: float = 0.0
        en: float = 0.0
        while not runner.has_collided and not world.is_done:
            # -- if arrived target waypoint, change to next one.
            next_waypoint = waypoints[curr_waypoint_index]
            distance = runner.get_location().distance(next_waypoint.transform.location)
            if distance < distance_error:
                curr_waypoint_index = (curr_waypoint_index + 1) % len(waypoints)
            world.draw_string(next_waypoint.transform.location, "X")
            if curr_waypoint_index == len(waypoints) - 1 and time.time() - t1 > 1:
                print(time.time() - t1)
                break

            yaw1 = calc_yaw(runner.get_location(), next_waypoint.transform.location)
            yaw2 = calc_vehicle_yaw(runner)
            e1 = calc_yaw_diff(yaw1, yaw2)
            ex = max_yaw_diff_in_the_future(waypoints, curr_waypoint_index, 20)
            for i in range(20):
                world.draw_string(waypoints[(curr_waypoint_index + i) % len(waypoints)].transform.location,
                                  "O",
                                  color=(255, 255, 0))

            events = world.key_handler()
            runner.key_handler(events)
            if runner.auto_pilot:
                runner.brake = 0
                runner.steer = max(-1, min(1, 1.2 * e1 + 0.0005 * en + 10.0 * (e1 - e0)))
                runner.throttle = min(1, 1 - 0.0007 * ex * runner.speed_kmh() ** 2)
                if runner.throttle < 0:
                    runner.brake = -runner.throttle
                    runner.throttle = 0
                # if runner.throttle < 0.5 and runner.speed_kmh() < 20:
                #     runner.throttle = 1
                #     runner.brake = 0
                # if runner.speed_kmh() > 45:
                #     runner.throttle = 0
                #     runner.brake = 0.1
                e0 = e1
                en = max(-100, min(100, en + e1))
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
        runner.destroy()
except Exception as e:
    print(e)
finally:
    print("done.")
