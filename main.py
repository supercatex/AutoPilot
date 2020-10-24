from helper import *


try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race_1", (640, 480))

    lap_speed = 0
    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3", debug=False)
        box = carla.BoundingBox(
            runner.get_transform().location,
            carla.Vector3D(2, 2, 2)
        )
        world.draw_box(box, runner.get_transform().rotation, 0.5, life_time=60)
        runner.auto_pilot = True

        # -- Global Planning -- begin
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
        # -- Global Planning -- end

        t1 = time.time()
        e0: float = 0.0
        en: float = 0.0
        while not runner.has_collided and not world.is_done:
            # -- Select the target waypoint -- begin
            next_waypoint = waypoints[curr_waypoint_index]
            distance = runner.get_location().distance(next_waypoint.transform.location)
            if distance < distance_error:
                curr_waypoint_index = (curr_waypoint_index + 1) % len(waypoints)
            world.draw_string(next_waypoint.transform.location, "X")
            if curr_waypoint_index == len(waypoints) - 1 and time.time() - t1 > 1:
                lap_speed = time.time() - t1
                print("%.2fs" % lap_speed)
                break
            # -- Select the target waypoint -- end

            # -- Local Planning -- begin
            n_future = 20
            kp, ki, kd, kf = 1.2, 0.0005, 10.0, 0.0007
            yaw1 = calc_yaw(runner.get_location(), next_waypoint.transform.location)
            yaw2 = calc_vehicle_yaw(runner)
            e1 = calc_yaw_diff(yaw1, yaw2)
            ex = max_yaw_diff_in_the_future(waypoints, curr_waypoint_index, n_future)
            for i in range(n_future):
                w = waypoints[(curr_waypoint_index + i) % len(waypoints)]
                world.draw_string(w.transform.location, "O", color=(255, 255, 0))

            events = world.key_handler()
            runner.key_handler(events)
            if runner.auto_pilot:
                runner.brake = 0
                runner.steer = max(-1.0, min(1.0, kp * e1 + ki * en + kd * (e1 - e0)))
                runner.throttle = min(1.0, 1.0 - kf * ex * runner.speed_kmh() ** 2)
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
                en = max(-100.0, min(100.0, en + e1))
            runner.action()
            # -- Local Planning -- end

            world.render_image(runner.rgb_image)
            world.render_text("Client: %.1ffps" % world.clock.get_fps(), (10, 10))
            world.render_text("Speed: %.2fkm/h" % runner.speed_kmh(), (10, 30))
            world.render_text("(W)  Throttle: %.2f" % runner.throttle, (10, 50))
            world.render_text("(Space) Brake: %.2f" % runner.brake, (10, 70))
            world.render_text("(A, D)  Steer: %.2f" % runner.steer, (10, 90))
            world.render_text("(S)   Reverse: %d" % runner.reverse, (10, 110))
            world.render_text("(P)Auto Pilot: " + str(runner.auto_pilot), (10, 130))
            world.render_text("L: %.2f, R: %.2f" % (runner.distance_left, runner.distance_right), (10, 150))
            world.render_text("Lap speed: %.2fs" % lap_speed, (450, 10))
            world.redraw_display()
        runner.destroy()
except Exception as e:
    print(e)
finally:
    print("done.")
