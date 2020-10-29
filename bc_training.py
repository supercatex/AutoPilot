from tensorflow.keras import models, layers, activations
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import os
import numpy as np
import pickle
from datetime import datetime

# -- Behavior Cloning -- #

# -- Prepare model structure -- begin
in_shape = (15, 20, 1)
model_dir = "./bc_model"
model_name = "model"
model_path = os.path.join(model_dir, model_name + ".h5")
if os.path.exists(model_path):
    model = models.load_model(model_path)
else:
    x_in = layers.Input(in_shape, name="gray-image")
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
    out = layers.Dense(3, activation=activations.tanh, name="out")(x)

    model = models.Model(x_in, out)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    plot_model(model, os.path.join(model_dir, model_name + ".png"), show_shapes=True)
# -- Prepare model structure -- end

# -- Compile with optimizer and loss function -- begin
model.compile(optimizer=optimizers.Adam(), loss=losses.mse)
# -- Compile with optimizer and loss function -- end

# -- Loading training data -- begin
print("Loading data...")
data_dir = "./data"
data = []
filenames = os.listdir(data_dir)
for i, name in enumerate(filenames):
    print(i + 1, "/", len(filenames), name)
    f = open(os.path.join(data_dir, name), "rb")
    memory = pickle.load(f)
    f.close()
    data.extend(memory)
data = np.array(data)
# -- Loading training data -- end

# -- Formatting data for X, y -- begin
x = data[:, 0]
x = np.array([np.array(s, dtype=np.float32) for s in x])
y = np.array(data[:, 2], dtype=np.float32)
yy = []
for i in y:
    yyy = [0.0, 0.0, 0.0]
    if i < 0:
        yyy[0] = 1.0
    elif i == 0:
        yyy[1] = 1.0
    else:
        yyy[2] = 1.0
    yy.append(yyy)
y = np.array(yy, dtype=np.float32)
# -- Formatting data for X, y -- end

# -- Prepare training callback functions -- begin
cb_tensorboard = TensorBoard(log_dir="./logs/%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
cb_reduceLR = ReduceLROnPlateau(monitor="loss", factor=0.5, min_lr=0.000001, patience=5, verbose=1)
cb_early_stop = EarlyStopping(monitor="loss", patience=10, verbose=1)
cb_check_point = ModelCheckpoint(os.path.join(model_dir, model_name + "_best.h5"), "loss", save_best_only=True, verbose=1)
callbacks = [cb_tensorboard, cb_reduceLR, cb_early_stop, cb_check_point]
# -- Prepare training callback functions -- end

# -- Training and than save model -- begin
model.fit(x, y, epochs=1000, batch_size=32, verbose=1, callbacks=callbacks)
model.save(model_path)
# -- Training and than save model -- end
