# importing modules
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

# loading data and target
import numpy as np
data = np.load("data_processing/data.npy")
target = np.load("data_processing/target.npy")

# importing tensorflowjs for converting model to tensorflow.js
import tensorflowjs as tfjs

# defining model
model = Sequential()
model.add(Dense(64, input_shape=(4,), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))

# compiling model
model.compile(
    optimizer=RMSprop(lr=0.002),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# training model
model.fit(data, target, epochs=1000)

# evaluating model
results = model.evaluate(data, target)
print(results)

# saving the models
model.save("../models/keras_model.h5")
tfjs.converters.save_keras_model(model, "../models/tfjs_model")


