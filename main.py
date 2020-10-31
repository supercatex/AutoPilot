import cv2
from agents import *


client_fps = 20.0
pilot_mode = Vehicle.DDQN_PILOT   # Here to change other pilot agent.
agent = None


in_shape = (15, 20, 1)
save_data = False
data_dir = "./data"     # Only for PID pilot collect data. (None: no data collection, default: "./data")
data_batch_size = 500   # Only for PID pilot collect data.
follow_agent = None     # Only for Follow pilot.
follower = None         # Only for Follow pilot.
ddqn = None
frames = None

try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race_1", (640, 480), (-100, -50, 85), (-70.0, 0.0, 0.0))

    best_time = 0
    lap_speed = 0
    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3", debug=pilot_mode == Vehicle.ASSIST_PILOT)
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
            time.sleep(0.0001)
        # -- Global Planning -- end

        t1 = time.time()
        while not world.is_done and not runner.has_collided:
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
            if curr_waypoint_index == len(waypoints) - 1 and time.time() - t1 > 1.0 or \
               runner.get_location().distance(waypoints[0].transform.location) < waypoint_distance * 2 and time.time() - t1 > 20.0:
                lap_speed = time.time() - t1
                t1 = time.time()
                print("%.2fs" % lap_speed)
                # break
            curve = max_yaw_diff_in_the_future(waypoints, curr_waypoint_index, 20)
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
                    agent = Agent(data_dir, data_batch_size)
                agent.step(v=runner)
                agent.throttle = 0.3
                if save_data:
                    agent.add_record(img)

            if runner.auto_pilot == Vehicle.ASSIST_PILOT:
                if agent is None:
                    agent = Agent()
                    runner.debug = True
                    runner.obstacle_right.debug = True
                agent.step(v=runner)
                runner_idx = int(np.argmin([w.transform.location.distance(runner.get_location()) for w in waypoints]))
                curve = max_yaw_diff_in_the_future(waypoints, runner_idx, 20)
                for i in range(20):
                    world.draw_string(
                        waypoints[(runner_idx + i + 3) % len(waypoints)].transform.location,
                        "O", color=(255, 255, 0)
                    )

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
                if save_data:
                    agent.add_record(img)

            if runner.auto_pilot == Vehicle.FOLLOW_PILOT:
                if agent is None:
                    agent = PIDAgent(1.3, 0.0002, 3.0, 0.00095, data_dir, data_batch_size)
                agent.step(v=runner, waypoints=waypoints, cur_index=curr_waypoint_index, n_future=20)

                if follow_agent is None:
                    follow_agent = FollowAgent(1.3, 0.0002, 3.0, 0.05, 0.0000003, 5.0, 5.0)
                if follower is None:
                    curr_waypoint_index = (curr_waypoint_index + 3) % len(waypoints)
                    w = waypoints[curr_waypoint_index]
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
                follow_agent.step(v=follower, waypoints=ws)
                follower.throttle = follow_agent.throttle
                follower.steer = follow_agent.steer
                follower.brake = follow_agent.brake
                follower.action()

            if runner.auto_pilot == Vehicle.BC_PILOT:
                if agent is None:
                    agent = BCAgent("./bc_model", "model_best")
                agent.step(s=img)

            if runner.auto_pilot == Vehicle.DQN_PILOT:
                img = cv2.resize(img, (20, 15))
                if agent is None:
                    agent = DQNAgent("./dqn_model", "model_best", (15, 20, 1), 128, 10000, 0.99, 0.1)
                a1 = agent.step(s=img)

                # -- Remember game states -- begin
                s0 = [np.zeros(img.shape, dtype=np.float32), 0.0]
                a0 = 0
                if len(agent.memory) > 0:
                    s0 = agent.memory[-1][2]
                    a0 = agent.memory[-1][3]
                s1 = [img, runner.speed_kmh()]
                reward = 5 - (runner.distance_left - runner.distance_right)
                terminate = False
                if runner.has_collided:
                    reward = -50
                    terminate = True
                    t2 = time.time()
                    print("Running time: %.2fs" % (t2 - t1))
                    if t2 - t1 > best_time:
                        best_time = t2 - t1
                        agent.model.save(agent.model_path)
                if terminate or reward != 0 or np.random.rand() >= 0.0:
                    agent.memory.append((s0, a0, s1, a1, reward, terminate))
                # -- Remember game states -- end

                # -- After terminate train min-batch from memory -- begin
                if terminate:
                    agent.replay()
                # -- After terminate train min-batch from memory -- end

            if runner.auto_pilot == Vehicle.DDQN_PILOT:
                img = runner.rgb_image.swapaxes(0, 1)
                img = cv2.resize(img, (30, 40))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.array(img, dtype=np.float32) / 255
                if agent is None:
                    from DDQN import DDQN
                    agent = Agent()
                    ddqn = DDQN(
                        "./ddqn_model", "model_best",
                        in_shape=img.shape + (1,), out_size=3,
                        epsilon=0.1,
                        update_steps=1000,
                        learning_rate=0.00025,
                        memory_size=100000
                    )
                    frames = deque(maxlen=ddqn.in_shape[2])
                frames.append(img)

                if len(frames) == frames.maxlen:
                    s0 = np.zeros(ddqn.in_shape, dtype=np.float32)
                    a0 = 1
                    if len(ddqn.memory) > 0:
                        s0 = ddqn.memory[-1][2]
                        a0 = ddqn.memory[-1][3]
                    s1 = np.array(frames, dtype=np.float32)
                    s1 = np.reshape(s1, ddqn.in_shape)
                    a1 = ddqn.step(state=s1)
                    r1 = 0.0
                    terminal = runner.has_collided
                    if terminal:
                        r1 = -1.0
                    elif runner.speed_kmh() > 0:
                        r1 = 0.1
                    elif runner.speed_kmh() == 0:
                        r1 = -0.1
                    ddqn.add(s0, a0, s1, a1, r1, terminal)
                    ddqn.replay()

                    agent.throttle = 0.3
                    agent.steer = 0.0
                    agent.brake = 0.0
                    agent.reverse = False
                    if a1 == 0:
                        agent.steer = -1.0
                    elif a1 == 1:
                        agent.throttle = 0.3
                    elif a1 == 2:
                        agent.steer = 1.0
            # -- Agent running step -- end

            # -- Action -- begin
            runner.throttle = agent.throttle
            runner.steer = agent.steer
            runner.brake = agent.brake
            runner.reverse = agent.reverse
            runner.action()
            # -- Action -- end

            # -- Rendering -- begin
            img = runner.rgb_image.copy()
            if follower is not None:
                tmp = follower.rgb_image
                tmp = cv2.resize(tmp, (180, 240))
                img[400:, 300:, :] = tmp
            world.render_image(img)

            world.render_text("(W)  Throttle: %.2f" % runner.throttle, (10, 40))
            world.render_text("(A, D)  Steer: %.2f" % runner.steer, (10, 60))
            world.render_text("(Space) Brake: %.2f" % runner.brake, (10, 80))
            world.render_text("(S)   Reverse: %d" % runner.reverse, (10, 100))
            world.render_text("(P)Auto Pilot: " + str(runner.auto_pilot), (10, 120))
            world.render_text("Lap speed: %.2fs" % (time.time() - t1), (390, 10), font_size=24, bold=True)
            if runner.auto_pilot != Vehicle.NO_PILOT:
                world.render_text("Speed: %.2fkm/h" % runner.speed_kmh(), (10, 10), font_size=24, bold=True, color=(255, 255, 255 - min(255.0, runner.speed_kmh() / 40 * 255)))
                world.render_text("L:%.1f" % runner.distance_left, (10, 450), font_size=24, color=(255, 255, min(255.0, runner.distance_left / 4 * 255)), bold=True)
                world.render_text("R:%.1f" % runner.distance_right, (540, 450), font_size=24, color=(255, 255, min(255.0, runner.distance_right / 4 * 255)), bold=True)
                world.render_text("Curve:%.1f" % (180 - curve * 180 / np.pi), (250, 430), font_size=24, color=(255, 255, 255 - min(255.0, curve / 2 * 255)), bold=True)
            else:
                world.render_text("Speed: %.2fkm/h" % runner.speed_kmh(), (10, 10), font_size=24, bold=True)

            world.render_text(
                "Client: %.1ffps, Server: %.1ffps" % (
                    world.clock.get_fps(),
                    world.server_clock.get_fps()
                ), (210, 460), font_size=12)
            world.redraw_display(client_fps)
            # -- Rendering -- end

        # -- Game Over screen -- begin
        # img = cv2.imread("bg-index.jpg")
        # img = cv2.resize(img, (640, 480))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # zeros = np.zeros(img.shape, dtype=np.uint8)
        # zeros = cv2.rectangle(zeros, (0, 120), (640, 360), (255, 255, 255), -1)
        # img = cv2.addWeighted(img, 1.0, zeros, 0.3, 0.0)
        # img = np.swapaxes(img, 0, 1)
        # t2 = time.time()
        # while not world.is_done:
        #     events = world.key_handler()
        #     world.render_image(img)
        #     if lap_speed > 0:
        #         world.render_text("Completed!", (50, 140), font_size=48, bold=True, color=(0, 0, 255))
        #         world.render_text("Lap speed: %.2fs" % lap_speed, (50, 200), font_size=36, bold=True, color=(0, 0, 255))
        #     else:
        #         world.render_text("You crashed!", (50, 140), font_size=48, bold=True, color=(255, 0, 0))
        #         world.render_text("Running time: %.2fs" % (t2 - t1), (50, 200), font_size=36, bold=True, color=(255, 0, 0))
        #     world.render_text("Press ESC to quit", (370, 365), font_size=24, bold=False, italic=True)
        #     world.redraw_display(client_fps)
        # -- Game Over screen -- end
        # world.is_done = False
        # follower = None
        runner.destroy()
        # world.destroy_actors()
except Exception as e:
    print(e)
finally:
    if ddqn is not None:
        ddqn.backup()
    print("done.")
