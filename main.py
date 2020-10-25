from helper import *
from tensorflow.keras import models, layers, activations, optimizers, losses, metrics
import os
from collections import deque
import cv2


model_path = "model.h5"
if os.path.exists(model_path):
    model = models.load_model(model_path)
else:
    x_in = layers.Input((120, 160, 3))
    x = x_in
    x = layers.Conv2D(32, (3, 3), (1, 1), "same", activation=activations.relu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), (1, 1), "same", activation=activations.relu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(96, (3, 3), (1, 1), "same", activation=activations.relu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    throttle_out = layers.Dense(1, activation=activations.sigmoid)(x)
    steer_out = layers.Dense(1, activation=activations.tanh)(x)
    model = models.Model(x_in, [throttle_out, steer_out])
model.compile(
    optimizer=optimizers.Adam(),
    loss=[losses.mse, losses.mse],
    metrics=[metrics.mean_squared_error]
)

try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)

    world = World(client, "road_race_1", (640, 480))
    # time.sleep(3.0)
    # settings = carla.WorldSettings(synchronous_mode=True, no_rendering_mode=False, fixed_delta_seconds=1.0/30)
    # frame = world.carla_world.apply_settings(settings)

    lap_speed = 0
    while not world.is_done:
        runner = Vehicle(world, "runner", bp_filter="vehicle.tesla.model3", debug=False)
        # box = carla.BoundingBox(
        #     runner.get_transform().location,
        #     carla.Vector3D(2, 2, 2)
        # )
        # world.draw_box(box, runner.get_transform().rotation, 0.5, life_time=60)
        runner.auto_pilot = Vehicle.DQN_PILOT

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
        e0: float = 0.0                 # For PID pilot -- preview error
        en: float = 0.0                 # For PID pilot -- summary error
        memory = deque(maxlen=10000)    # For DQN pilot -- replay memory
        states = deque(maxlen=3)        # For DQN pilot -- input state
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
                kp, ki, kd, kf = 1.2, 0.0005, 10.0, 0.00075
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
                # if runner.throttle < 0.5 and runner.speed_kmh() < 20:
                #     runner.throttle = 1
                #     runner.brake = 0
                # if runner.speed_kmh() > 45:
                #     runner.throttle = 0
                #     runner.brake = 0.1
                e0 = e1
                en = max(-100.0, min(100.0, en + e1))
            elif runner.auto_pilot == Vehicle.DQN_PILOT:
                img = runner.rgb_image.swapaxes(0, 1)
                img = cv2.resize(img, (160, 120))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.array(img, dtype=np.float) / 255
                states.append(img)

                if len(states) == 3:
                    state_1 = np.ones((120, 160, 3))
                    state_1[:, :, 0] *= states[0]
                    state_1[:, :, 1] *= states[1]
                    state_1[:, :, 2] *= states[2]
                    x_in = np.reshape(state_1, (1, 120, 160, 3))
                    action = np.array(model.predict(x_in))[:, 0, 0]

                    yaw1 = calc_yaw(runner.get_location(), next_waypoint.transform.location)
                    yaw2 = calc_vehicle_yaw(runner)
                    diff = calc_yaw_diff(yaw1, yaw2)
                    distance = runner.get_location().distance(next_waypoint.transform.location)

                    target = [1 - runner.speed_kmh() / 40, action[1] + 0.9 * diff]
                    memory.append((state_1, target))

                    runner.throttle = float(action[0])
                    runner.steer = float(action[1])

            runner.action()
            # -- Local Planning -- end

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
            world.redraw_display()
            # -- Rendering -- end
        runner.destroy()

        if len(memory) >= 128:
            import random
            batch = random.sample(memory, 128)
            state, target = zip(*batch)
            state = np.array(state)
            target = np.array(target)
            loss = model.train_on_batch(state, target)
            print(loss)
            model.save(model_path)

except Exception as e:
    print(e)
finally:
    print("done.")
