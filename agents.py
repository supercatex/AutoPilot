from helper import *
import os
from tensorflow.keras import models
import numpy as np
from collections import deque
import random
import pickle
from datetime import datetime


class Agent(object):
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.reverse = 0.0

    def step(self, **kwargs):
        v: Vehicle = kwargs.get("v")
        self.throttle = v.throttle
        self.steer = v.steer
        self.brake = v.brake
        self.reverse = v.reverse


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
        a = self.model.predict(s)
        print(a)
        self.throttle = float(a[0][0][0])
        self.steer = float(a[1][0][0])
        self.brake = 0.0


class DQNAgent(Agent):
    def __init__(self, model_dir, model_name, batch_size=32, memory_size=10000):
        super(DQNAgent, self).__init__()
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name + ".h5")
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = models.load_model(self.model_path)

    def replay(self):
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            s0, a0, s1, reward, terminate = zip(*batch)
            state = np.array(s0)[:, 0]
            state = np.array([np.array(s, dtype=np.float32) for s in state])
            target = np.array([])

            loss = self.model.train_on_batch(state, target)
            self.model.save(self.model_path)
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
        v1: Vehicle = kwargs.get("v1")
        v2: Vehicle = kwargs.get("v2")

        yaw1 = calc_vehicle_yaw(v1)
        yaw2 = calc_vehicle_yaw(v2)
        self.cur_error_1 = calc_yaw_diff(yaw1, yaw2)
        print(self.cur_error_1)

        self.steer = self.kp_1 * self.cur_error_1
        self.steer += self.ki_1 * self.sum_error_1
        self.steer += self.kd_1 * (self.cur_error_1 - self.pre_error_1)
        self.steer = max(-1.0, min(1.0, self.steer))

        self.cur_error_2 = v1.get_location().distance(v2.get_location()) - self.distance
        self.throttle = self.kp_2 * self.cur_error_2
        self.throttle += self.ki_2 * self.sum_error_2
        self.throttle += self.kd_2 * (self.cur_error_2 - self.pre_error_2)
        self.throttle = min(1.0, self.throttle)
        if self.throttle < 0:
            self.brake = -self.throttle
            self.throttle = 0

        self.sum_error_1 = max(-100.0, min(100.0, self.sum_error_1 + self.cur_error_1))
        self.pre_error_1 = self.cur_error_1

        self.sum_error_2 = max(-100.0, min(100.0, self.sum_error_2 + self.cur_error_2))
        self.pre_error_2 = self.cur_error_2
