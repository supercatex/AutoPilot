from tensorflow.keras import models, layers, activations
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import os
import numpy as np
import pickle
from datetime import datetime

in_shape = (60, 80, 3)
model_dir = "./models"
model_name = "model"
model_path = os.path.join(model_dir, model_name + ".h5")
if os.path.exists(model_path):
    model = models.load_model(model_path)
else:
    x_in = layers.Input(in_shape, name="3-gray-image")
    x = x_in
    x = layers.Conv2D(32, (5, 5), (1, 1), "same", activation=activations.relu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), (1, 1), "same", activation=activations.relu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    throttle_out = layers.Dense(1, activation=activations.sigmoid, name="throttle")(x)
    steer_out = layers.Dense(1, activation=activations.tanh, name="steer")(x)
    brake_out = layers.Dense(1, activation=activations.sigmoid, name="brake")(x)
    model = models.Model(x_in, [throttle_out, steer_out, brake_out])
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    plot_model(model, os.path.join(model_dir, model_name + ".png"), show_shapes=True)
model.compile(
    optimizer=optimizers.Adam(),
    loss=[losses.mse, losses.mse, losses.mse]
)

print("Loading data...")
data_dir = "./data"
data = []
filenames = os.listdir(data_dir)
for i, name in enumerate(filenames):
    print(i + 1, "/", len(filenames))
    f = open(os.path.join(data_dir, name), "rb")
    memory = pickle.load(f)
    f.close()
    data.extend(memory)
data = np.array(data)

x = data[:, 0]
x = np.array([np.array(d, dtype=np.float32) for d in x])
y = np.array(data[:, 1:4], dtype=np.float32)

cb_tensorboard = TensorBoard(log_dir="./logs/%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
cb_reduceLR = ReduceLROnPlateau(monitor="loss", factor=0.5, min_lr=0.00001, patience=5)
cb_early_stop = EarlyStopping(monitor="loss", patience=10)
cb_check_point = ModelCheckpoint(os.path.join(model_dir, model_name + "_best.h5"), "loss", save_best_only=True)
callbacks = [cb_tensorboard, cb_reduceLR, cb_early_stop, cb_check_point]
model.fit(x, y, epochs=1000, batch_size=32, verbose=1, callbacks=callbacks)
model.save(model_path)
