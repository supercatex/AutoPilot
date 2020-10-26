from helper import *
from tensorflow.keras import models
from tensorflow.keras import optimizers, losses
import os
from collections import deque
import cv2
import pickle
from datetime import datetime

client_fps = 60.0
pilot_mode = Vehicle.PID_PILOT
memory_size = 1000
is_save_memory = False
is_training = False

in_shape = (60, 80, 1)
model_dir = "./models"
model_name = "model"
model_path = os.path.join(model_dir, model_name + ".h5")
model = None
if os.path.exists(model_path):
    model = models.load_model(model_path)
    model.compile(
        optimizer=optimizers.Adam(),
        loss=[losses.mse, losses.mse, losses.mse]
    )

data_dir = "./data"

try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race_1", (640, 480))

    best_time = 0
    lap_speed = 0
    memory = deque(maxlen=memory_size)  # For DQN pilot -- replay memory
    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3", debug=False)
        runner.auto_pilot = pilot_mode

        # -- PID pilot Global Planning -- begin
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
        # -- PID pilot Global Planning -- end

        t1 = time.time()
        e0: float = 0.0                     # For PID pilot -- preview error
        en: float = 0.0                     # For PID pilot -- summary error
        while not runner.has_collided and not world.is_done:
            events = world.key_handler()
            runner.key_handler(events)

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

            if runner.auto_pilot == Vehicle.PID_PILOT:
                # -- Local Planning -- begin
                n_future = 20
                if client_fps >= 60.0:
                    kp, ki, kd, kf = 1.2, 0.0005, 10.0, 0.00075
                else:
                    kp, ki, kd, kf = 1.2, 0.0003, 5.0, 0.00195
                yaw1 = calc_yaw(runner.get_location(), next_waypoint.transform.location)
                yaw2 = calc_vehicle_yaw(runner)
                e1 = calc_yaw_diff(yaw1, yaw2)
                ex = max_yaw_diff_in_the_future(waypoints, curr_waypoint_index, n_future)
                for i in range(n_future):
                    w = waypoints[(curr_waypoint_index + i) % len(waypoints)]
                    world.draw_string(w.transform.location, "O", color=(255, 255, 0))

                runner.brake = 0
                runner.steer = max(-1.0, min(1.0, kp * e1 + ki * en + kd * (e1 - e0)))
                runner.throttle = min(1.0, 1.0 - kf * ex * runner.speed_kmh() ** 2)
                if runner.throttle < 0:
                    runner.brake = -runner.throttle
                    runner.throttle = 0
                e0 = e1
                en = max(-100.0, min(100.0, en + e1))
                # -- Local Planning -- end
            elif runner.auto_pilot == Vehicle.DQN_PILOT:
                if len(memory) > 0:
                    s = memory[-1]
                    x_in_1 = np.reshape(s[2][0], (1,) + in_shape)
                    # x_in_2 = np.array(float(runner.speed_kmh()))
                    # x_in_2 = np.reshape(x_in_2, (1, 1))
                    # x_in = [[x_in_1], [x_in_2]]
                    x_in = [x_in_1]
                    action = np.array(model.predict(x_in))[:, 0, 0]
                    print(action)
                    runner.throttle = float(action[0])
                    runner.steer = float(action[1])
                    runner.brake = 0.0

                    if runner.speed_kmh() < 20:
                        runner.throttle = 1.0

            # -- Remember game states -- begin
            img = runner.rgb_image.swapaxes(0, 1)
            img = cv2.resize(img, (in_shape[1], in_shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.array(img, dtype=np.float32) / 255
            img = np.reshape(img, in_shape)

            q = [1.0, 0.0, 0.0, False]
            s0 = [np.zeros(img.shape, dtype=np.float32), 0.0]
            a0 = [0.0, 0.0, 0.0, False]
            if len(memory) > 0:
                s0 = memory[-1][2]
                a0 = memory[-1][3]
                q = memory[-1][3]
                if memory[-1][5] == 0:
                    pass
                elif memory[-1][5] == 1:
                    q[1] = 1.0
                elif memory[-1][5] == 2:
                    q[1] = -1.0
                elif memory[-1][5] == 3:
                    q[0] = 0.0
                else:
                    q[2] = min(1.0, q[2] + 0.1)

            s1 = [img, runner.speed_kmh()]
            a1 = [runner.throttle, runner.steer, runner.brake, runner.reverse]
            reward = runner.speed_kmh()
            terminate = 0
            if runner.has_collided:
                reward = -100
                if runner.distance_left < runner.distance_right and runner.steer < 1.0:
                    terminate = 1
                    q[1] = 1.0
                elif runner.distance_left >= runner.distance_right and runner.steer > -1.0:
                    terminate = 2
                    q[1] = -1.0
                elif runner.throttle > 0:
                    terminate = 3
                    q[0] = q[0] * 0.5
                else:
                    terminate = 4
                    q[2] += 0.1
            else:
                q[1] = (runner.distance_right - runner.distance_left) * 0.3

            memory.append((s0, a0, s1, a1, reward, terminate, q))

            if is_save_memory and len(memory) == memory.maxlen:
                if not os.path.exists(data_dir):
                    os.mkdir(data_dir)
                f = open(os.path.join(data_dir, "data-%s.pickle" % datetime.now().strftime("%Y%m%d%H%M%S")), "wb")
                pickle.dump(memory, f)
                f.close()
                memory.clear()

            # -- Remember game states -- end
            runner.action()

            # -- Rendering -- begin
            world.render_image(runner.rgb_image)

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

        print("Memory size:", len(memory))
        print("Running time: %.2fs" % (time.time() - t1))
        if time.time() - t1 > best_time:
            best_time = time.time() - t1
            model.save("best_model_%d.h5" % best_time)

        if len(memory) >= 128:
            import random

            for i in range(1):
                batch = random.sample(memory, 128)
                s0, a0, s1, a1, reward, terminate, q = zip(*batch)
                state = np.array(s0)[:, 0]
                state = np.array([np.array(s, dtype=np.float32) for s in state])
                target = np.array(q)[:, :3]

                loss = model.train_on_batch(state, target)
                print(i + 1, "/ 30", loss)
            model.save(model_path)

except Exception as e:
    print(e)
finally:
    print("done.")
