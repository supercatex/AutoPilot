from helper import *
import os
from tensorflow.keras import models
import numpy as np
from collections import deque
import random
import pickle
from datetime import datetime
from tensorflow.keras import models, layers, activations
from tensorflow.keras import optimizers, losses
from tensorflow.keras.utils import plot_model


class Agent(object):
    def __init__(self, save_dir="./data", batch_size=500):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.reverse = 0.0
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.records = deque(maxlen=self.batch_size)

    def step(self, **kwargs):
        v: Vehicle = kwargs.get("v")
        self.throttle = v.throttle
        self.steer = v.steer
        self.brake = v.brake
        self.reverse = v.reverse

    def add_record(self, s):
        if self.save_dir is None:
            return
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.records.append((s, self.throttle, self.steer, self.brake, self.reverse))
        if len(self.records) >= self.batch_size:
            f = open(os.path.join(self.save_dir, "data-%s.pickle" % datetime.now().strftime("%Y%m%d%H%M%S")), "wb")
            pickle.dump(self.records, f)
            f.close()
            self.records.clear()


class PIDAgent(Agent):
    def __init__(self, kp, ki, kd, kf, save_dir=None, batch_size=100):
        super(PIDAgent, self).__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kf = kf
        self.cur_error = 0.0
        self.sum_error = 0.0
        self.pre_error = 0.0
        self.future_error = 0.0
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.records = deque(maxlen=batch_size)

    def step(self, **kwargs):
        v: Vehicle = kwargs.get("v")
        waypoints = kwargs.get("waypoints")
        cur_index = kwargs.get("cur_index")
        n_future = kwargs.get("n_future")

        yaw1 = calc_yaw(v.get_location(), waypoints[cur_index].transform.location)
        yaw2 = calc_vehicle_yaw(v)
        self.cur_error = calc_yaw_diff(yaw1, yaw2)
        self.future_error = max_yaw_diff_in_the_future(waypoints, cur_index, n_future)

        self.throttle = max(0.0, min(1.0, 1.0 - self.kf * self.future_error * v.speed_kmh() ** 2))

        self.steer = self.kp * self.cur_error
        self.steer += self.ki * self.sum_error
        self.steer += self.kd * (self.cur_error - self.pre_error)
        self.steer = max(-1.0, min(1.0, self.steer))

        self.brake = 0.0

        self.sum_error = max(-100.0, min(100.0, self.sum_error + self.cur_error))
        self.pre_error = self.cur_error

    def add_record(self, s):
        if self.save_dir is None:
            return
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.records.append((s, self.throttle, self.steer, self.brake, self.reverse))
        if len(self.records) >= self.batch_size:
            f = open(os.path.join(self.save_dir, "data-%s.pickle" % datetime.now().strftime("%Y%m%d%H%M%S")), "wb")
            pickle.dump(self.records, f)
            f.close()
            self.records.clear()


class BCAgent(Agent):
    def __init__(self, model_dir, model_name):
        super(BCAgent, self).__init__()
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name + ".h5")
        self.model = models.load_model(self.model_path)

    def step(self, **kwargs):
        s = kwargs.get("s")
        s = np.reshape(s, (1,) + s.shape)
        a = np.argmax(self.model.predict(s), axis=1)
        self.throttle = 0.3
        self.steer = 0.0
        self.brake = 0.0
        if a == 0:
            self.steer = -1.0
        elif a == 2:
            self.steer = 1.0


class DQNAgent(Agent):
    def __init__(self, model_dir, model_name, in_shape, batch_size=32, memory_size=10000, gamma=0.99, epsilon=1.0):
        super(DQNAgent, self).__init__()
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name + ".h5")
        self.in_shape = in_shape
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        if os.path.exists(self.model_path):
            print("Loading existed model...")
            self.model = models.load_model(self.model_path)
        else:
            self.model = self.new_model()
        self.model.compile(optimizer=optimizers.Adam(lr=0.0001 * 5), loss=losses.mse)
        # for i in range(1, 6):
        #     self.model.layers[i].trainable = False
        self.model2 = self.new_model()
        self.model2.set_weights(self.model.get_weights())
        self.steps = 0
        self.replace_iter = 100

    def new_model(self):
        x_in = layers.Input(self.in_shape, name="1-gray-image")
        x = x_in
        x = layers.Conv2D(32, (5, 5), (1, 1), "same", activation=activations.relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (5, 5), (1, 1), "same", activation=activations.relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), (1, 1), "same", activation=activations.relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation=activations.relu)(x)
        out = layers.Dense(3, activation=activations.linear, name="actions")(x)
        model = models.Model(x_in, out)
        if os.path.exists("./bc_model/model_best.h5"):
            model1 = models.load_model("./bc_model/model_best.h5")
            # for i in range(1, 9):
            #     model.layers[i].set_weights(model1.layers[i].get_weights())
            model.set_weights(model1.get_weights())

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        plot_model(model, os.path.join(self.model_dir, self.model_name + ".png"), show_shapes=True)
        return model

    def step(self, **kwargs):
        s = kwargs.get("s")
        if np.random.rand() <= self.epsilon:
            a = np.random.randint(0, 3)
            print("random:", a, ", epsilon:", self.epsilon)
        else:
            s = np.reshape(s, (1,) + s.shape)
            q = self.model.predict(s)
            print(q)
            a = np.argmax(q[0])

        self.throttle = 0.3
        self.steer = 0.0
        self.brake = 0.0
        if a == 0:
            self.steer = -1.0
        elif a == 1:
            self.throttle = 0.3
        else:
            self.steer = 1.0

        return a

    def replay(self):
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            s0, a0, s1, a1, reward, terminate = zip(*batch)
            s0 = np.array(s0)[:, 0]
            s0 = np.array([np.array(s, dtype=np.float32) for s in s0])
            s1 = np.array(s1)[:, 0]
            s1 = np.array([np.array(s, dtype=np.float32) for s in s1])
            qv = np.array(self.model.predict(s0))
            a1 = np.array(self.model.predict(s1))
            reward = np.array(reward, np.float32)
            terminate = np.array(terminate, np.bool)
            a2 = np.array(self.model2.predict(s1))
            a2 = a2[range(self.batch_size), np.argmax(a1, axis=1)]
            qv[range(self.batch_size), a0] = reward + self.gamma * a2 * np.invert(terminate)
            loss = self.model.train_on_batch(s0, qv)
            self.model.save(self.model_path)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.steps += 1
            print("steps:", self.steps)
            if self.steps % self.replace_iter == 0:
                self.model2.set_weights(self.model.get_weights())
            return loss


class FollowAgent(Agent):
    def __init__(self, kp_1, ki_1, kd_1, kp_2, ki_2, kd_2, distance):
        super(FollowAgent, self).__init__()
        self.kp_1 = kp_1
        self.ki_1 = ki_1
        self.kd_1 = kd_1
        self.kp_2 = kp_2
        self.ki_2 = ki_2
        self.kd_2 = kd_2
        self.distance = distance
        self.cur_error_1 = 0.0
        self.sum_error_1 = 0.0
        self.pre_error_1 = 0.0
        self.cur_error_2 = 0.0
        self.sum_error_2 = 0.0
        self.pre_error_2 = 0.0

    def step(self, **kwargs):
        v: Vehicle = kwargs.get("v")
        waypoints = kwargs.get("waypoints")

        if len(waypoints) == 0:
            self.throttle = 0.0
            self.brake = 1.0
            return

        yaw1 = calc_yaw(v.get_location(), waypoints[1].transform.location)
        yaw2 = calc_vehicle_yaw(v)
        self.cur_error_1 = calc_yaw_diff(yaw1, yaw2)

        self.steer = self.kp_1 * self.cur_error_1
        self.steer += self.ki_1 * self.sum_error_1
        self.steer += self.kd_1 * (self.cur_error_1 - self.pre_error_1)
        self.steer = max(-1.0, min(1.0, self.steer))

        self.cur_error_2 = 0.0
        for i in range(len(waypoints) - 1):
            self.cur_error_2 += waypoints[i].transform.location.distance(waypoints[i + 1].transform.location)
        self.cur_error_2 -= self.distance

        self.throttle = self.kp_2 * self.cur_error_2
        self.throttle += self.ki_2 * self.sum_error_2
        self.throttle += self.kd_2 * (self.cur_error_2 - self.pre_error_2)
        self.throttle = min(1.0, self.throttle)
        self.brake = 0.0
        if self.throttle < 0.0:
            self.brake = -self.throttle
            self.throttle = 0.0

        self.sum_error_1 = max(-100.0, min(100.0, self.sum_error_1 + self.cur_error_1))
        self.pre_error_1 = self.cur_error_1

        self.sum_error_2 = max(-100.0, min(100.0, self.sum_error_2 + self.cur_error_2))
        self.pre_error_2 = self.cur_error_2
