import os
import numpy as np
from typing import Tuple
from collections import deque
from tensorflow.keras import models, layers, activations
from tensorflow.keras import optimizers, losses
import random
import pickle
import matplotlib.pyplot as plt

# DQN: https://arxiv.org/pdf/1312.5602.pdf
# Double-DQN: https://arxiv.org/pdf/1509.06461.pdf


class DDQN(object):
    def __init__(self, model_dir: str, model_name: str,
                 log_name: str = "log", memory_name: str = "memory",
                 params_name: str = "params",
                 in_shape: Tuple = (84, 84, 4), out_size: int = 2,
                 batch_size: int = 32, learning_rate: float = 0.00025,
                 memory_size: int = 1000000, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.999998, update_steps=10000):
        self.model_dir: str = model_dir
        self.model_name: str = model_name
        self.model_path: str = os.path.join(model_dir, model_name + ".h5")
        self.log_name: str = log_name
        self.log_path: str = os.path.join(model_dir, log_name + ".txt")
        self.memory_name: str = memory_name
        self.memory_path: str = os.path.join(model_dir, memory_name + ".pickle")
        self.params_name: str = params_name
        self.params_path: str = os.path.join(model_dir, params_name + ".txt")
        self.in_shape: Tuple = in_shape
        self.out_size: int = out_size
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.memory: deque = deque(maxlen=memory_size)
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self.update_steps: int = update_steps
        self.current_step: int = 0

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if os.path.exists(self.memory_path):
            print("Loading previous replay memory...")
            with open(self.memory_path, "rb") as f:
                self.memory = pickle.load(f)

        if os.path.exists(self.params_path):
            print("Loading previous parameters...")
            with open(self.params_path, "r") as f:
                self.current_step = int(f.readline())
                self.epsilon = float(f.readline())

        if os.path.exists(self.model_path):
            print("Loading existed model...")
            self.q_model: models.Model = models.load_model(self.model_path)
        else:
            print("Create a new model...")
            self.q_model: models.Model = self.build_model()

        self.q_model = self.compile_model(self.q_model)
        self.t_model = self.build_model()
        self.t_model.set_weights(self.q_model.get_weights())

    def build_model(self) -> models.Model:
        x_in = layers.Input(self.in_shape)
        x = x_in
        x = layers.Conv2D(32, 8, 4, "same", activation=activations.relu)(x)
        x = layers.Conv2D(64, 4, 2, "same", activation=activations.relu)(x)
        x = layers.Conv2D(64, 3, 1, "same", activation=activations.relu)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation=activations.relu)(x)
        x_out = layers.Dense(self.out_size, activation=activations.linear)(x)
        return models.Model(x_in, x_out)

    def compile_model(self, model: models.Model) -> models.Model:
        model.compile(
            optimizer=optimizers.RMSprop(self.learning_rate, momentum=0.95),
            loss=losses.mse
        )
        return model

    def add(self, s0, a0, s1, a1, r1, terminal):
        s0 = np.array(s0, dtype=np.float32)
        a0 = int(a0)    # range(out_size)
        s1 = np.array(s1, dtype=np.float32)
        a1 = int(a1)    # range(out_size)
        r1 = float(r1)  # [-1, 1]
        terminal = bool(terminal)
        if s0.shape != self.in_shape or s1.shape != self.in_shape:
            print("Shape of state not match!")
        if a0 < 0 or a0 >= self.out_size or a1 < 0 or a1 >= self.out_size:
            print("Action size is not match!")
        self.memory.append((s0, a0, s1, a1, r1, terminal))

    def replay(self):
        if len(self.memory) < self.batch_size:
            print("Not enough replay memory! %d/%d" % (len(self.memory), self.batch_size))
            return

        batch = random.sample(self.memory, self.batch_size)
        s0, a0, s1, _, r1, terminal = zip(*batch)
        s0 = np.array(s0, dtype=np.float32)
        s1 = np.array(s1, dtype=np.float32)

        q0 = np.array(self.q_model.predict(s0), dtype=np.float32)
        q1 = np.array(self.q_model.predict(s1), dtype=np.float32)
        a1 = np.argmax(q1, axis=1)
        qt = np.array(self.t_model.predict(s1), dtype=np.float32)
        if terminal:
            q0[range(self.batch_size), a0] = r1
        else:
            q0[range(self.batch_size), a0] = r1 + self.gamma * qt[range(self.batch_size), a1]

        self.q_model.train_on_batch(s0, q0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.current_step += 1
        if self.current_step % self.update_steps == 0:
            self.t_model.set_weights(self.q_model.get_weights())

        self.q_model.save(self.model_path)

        sum_reward = 0
        i = len(self.memory) - 2
        while i >= 0 and not self.memory[i][-1]:
            sum_reward += self.memory[i][-2]
            i -= 1
        with open(self.log_path, "a+") as f:
            f.write("%.2f\n" % sum_reward)

    def backup(self):
        with open(self.params_path, "w+") as f:
            f.write("%d\n%f\n" % (self.current_step, self.epsilon))

        with open(self.memory_path, "wb") as f:
            pickle.dump(self.memory, f)

        print("STEP:", self.current_step)
        print("Epsilon:", self.epsilon)
        print("Memory:", len(self.memory))

        self.show_history()

    def show_history(self):
        with open(self.log_path, "r") as f:
            data = list(map(float, f.readlines()))
        plt.plot(range(len(data)), data)
        plt.show()

    def step(self, **kwargs) -> int:
        if "state" not in kwargs.keys():
            print("State parameter not found!")
            return -1
        s = kwargs.get("state")
        if s.shape != self.in_shape:
            print("Shape of state is not match!")
            return -1

        if np.random.rand() <= self.epsilon:
            a = np.random.randint(0, self.out_size)
            print(self.current_step, ", Random action:", a, ", epsilon:", self.epsilon)
            return a
        else:
            s = np.reshape(s, (1,) + s.shape)
            q = self.q_model.predict(s)
            a = np.argmax(q, axis=1)[0]
            print(self.current_step, ", Action:", a, ", Q:", q)
            return a
