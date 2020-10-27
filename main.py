import cv2
from agents import *


client_fps = 20.0
pilot_mode = Vehicle.PID_PILOT
data_dir = None         # Only PID pilot
data_batch_size = 100   # Only PID pilot
in_shape = (60, 80, 1)
agent = None
agent_2 = None
follower = None

try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race_1", (640, 480))

    best_time = 0
    lap_speed = 0
    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3", debug=False)
        runner.auto_pilot = pilot_mode

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
        while not world.is_done:# and not runner.has_collided:
            # -- Key events -- begin
            events = world.key_handler()
            runner.key_handler(events)
            # -- Key events -- end

            # -- Select the target waypoint -- begin
            next_waypoint = waypoints[curr_waypoint_index]
            distance = runner.get_location().distance(next_waypoint.transform.location)
            if distance < distance_error:
                curr_waypoint_index = (curr_waypoint_index + 1) % len(waypoints)
            world.draw_string(next_waypoint.transform.location, "X")
            if curr_waypoint_index == len(waypoints) - 1 and time.time() - t1 > 1:
                lap_speed = time.time() - t1
                t1 = time.time()
                print("%.2fs" % lap_speed)
                # break
            # -- Select the target waypoint -- end

            # -- Camera snapshot -- begin
            img = runner.rgb_image.swapaxes(0, 1)
            img = cv2.resize(img, (in_shape[1], in_shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.array(img, dtype=np.float32) / 255
            img = np.reshape(img, in_shape)
            # -- Camera snapshot -- end

            # -- Agent running step -- begin
            if runner.auto_pilot == Vehicle.NO_PILOT:
                if agent is None:
                    # agent = Agent()
                    agent = PIDAgent(1.3, 0.0002, 3.0, 0.00055, data_dir, data_batch_size)
                agent.step(v=runner, waypoints=waypoints, cur_index=curr_waypoint_index, n_future=20)
                # agent.step(v=runner)

                if agent_2 is None:
                    agent_2 = FollowAgent(1.3, 0.0002, 3.0, 0.05, 0.0000003, 5.0, 5.0)
                if follower is None:
                    curr_waypoint_index = (curr_waypoint_index + 3) % len(waypoints)
                    w = waypoints[curr_waypoint_index]
                    # runner.actor.set_location(w.transform.location)
                    runner.actor.set_transform(w.transform)
                    follower = Vehicle(world, "follower", bp_filter="vehicle.*", start_tf=runner.start_tf, debug=False)
                runner_idx = int(np.argmin([w.transform.location.distance(runner.get_location()) for w in waypoints]))
                idx = np.argmin([w.transform.location.distance(follower.get_location()) for w in waypoints])
                ws = []
                while idx != runner_idx:
                    ws.append(waypoints[idx])
                    world.draw_string(
                        waypoints[idx].transform.location,
                        "O",
                        color=(0, 255, 0)
                    )
                    idx = (idx + 1) % len(waypoints)
                agent_2.step(v=follower, waypoints=ws)
                follower.throttle = agent_2.throttle
                follower.steer = agent_2.steer
                follower.brake = agent_2.brake
                print(follower.throttle, follower.steer, follower.brake)
                follower.action()

            if runner.auto_pilot == Vehicle.PID_PILOT:
                if agent is None:
                    if client_fps >= 60.0:
                        agent = PIDAgent(1.2, 0.0005, 10.0, 0.00075, data_dir, data_batch_size)
                    else:
                        agent = PIDAgent(1.3, 0.0002, 3.0, 0.00095, data_dir, data_batch_size)
                agent.step(v=runner, waypoints=waypoints, cur_index=curr_waypoint_index, n_future=20)
                for i in range(20):
                    world.draw_string(
                        waypoints[(curr_waypoint_index + i) % len(waypoints)].transform.location,
                        "O", color=(255, 255, 0)
                    )
                agent.add_record(img)

            if runner.auto_pilot == Vehicle.BC_PILOT:
                if agent is None:
                    agent = BCAgent("bc_model", "model_best")
                agent.step(s=img)

            if runner.auto_pilot == Vehicle.DQN_PILOT:
                if agent is None:
                    agent = DQNAgent("dqn_model", "model_best", 128, 10000)
                agent.step(s=img)

                # -- Remember game states -- begin
                s0 = [np.zeros(img.shape, dtype=np.float32), 0.0]
                a0 = [agent.throttle, agent.steer, agent.brake, agent.reverse]
                if len(agent.memory) > 0:
                    s0, a0 = agent.memory[-1][2:4]
                s1 = [img, runner.speed_kmh()]
                reward = runner.speed_kmh()
                terminate = 0
                if runner.has_collided:
                    reward = -100
                    terminate = 1
                    t2 = time.time()
                    print("Running time: %.2fs" % (t2 - t1))
                    if t2 - t1 > best_time:
                        best_time = t2 - t1
                        # agent.model.save("best_model_%d.h5" % best_time)
                agent.memory.append((s0, a0, s1, reward, terminate))
                # -- Remember game states -- end

                # -- After terminate train min-batch from memory -- begin
                if terminate:
                    agent.replay()
                # -- After terminate train min-batch from memory -- end
            # -- Agent running step -- end

            # -- Action -- begin
            runner.throttle = agent.throttle
            runner.steer = agent.steer
            runner.brake = agent.brake
            runner.action()
            # -- Action -- end

            # -- Rendering -- begin
            img = runner.rgb_image.copy()
            if follower is not None:
                tmp = follower.rgb_image
                tmp = cv2.resize(tmp, (180, 240))
                img[400:, 300:, :] = tmp
            world.render_image(img)

            world.render_text("Speed: %.2fkm/h" % runner.speed_kmh(), (10, 10))
            world.render_text("L: %.2f, R: %.2f" % (runner.distance_left, runner.distance_right), (10, 30))
            world.render_text("(W)  Throttle: %.2f" % runner.throttle, (10, 50))
            world.render_text("(Space) Brake: %.2f" % runner.brake, (10, 70))
            world.render_text("(A, D)  Steer: %.2f" % runner.steer, (10, 90))
            world.render_text("(S)   Reverse: %d" % runner.reverse, (10, 110))
            world.render_text("(P)Auto Pilot: " + str(runner.auto_pilot), (10, 130))
            world.render_text("Lap speed: %.2fs" % lap_speed, (450, 10))
            world.render_text(
                "Client: %.1ffps, Server: %.1ffps" % (
                    world.clock.get_fps(),
                    world.server_clock.get_fps()
                ), (280, 450))
            world.redraw_display(client_fps)
            # -- Rendering -- end
        runner.destroy()
except Exception as e:
    print(e)
finally:
    print("done.")
