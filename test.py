import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os

from superloop import Model, SGU, RegisterMemory


CONFIG = {
    'model_outputs': 3,
    'recurrent_model': SGU,
    'recurrent_layers': 5, # number of recurrent layers
    'recurrent_units': 3, # number of recurrent units on each layer
    'superloop_models': [RegisterMemory], # classes used to build models used in the superloop
    'RegisterMemory': {
        'register_width': 32,
        'depth': 7,
    }
}


timesteps = 3
inputs = [keras.layers.Input(shape=(7,), name="Main/StepInput{}".format(i)) for i in range(timesteps)]
mymodelbuilder = Model(name="Main", config=CONFIG)
outputs = [None]*timesteps

for timestep in range(timesteps):
    outputs[timestep] = mymodelbuilder.build(inputs[timestep], skip_superloop=(timestep == timesteps-1))

model = keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

TensorboardDir = "{}/tensorboard_logs/superloop".format(os.environ['HOME'])
os.system("mkdir -p {}".format(TensorboardDir))

# https://keras.io/callbacks/#tensorboard
tensorboardcb = keras.callbacks.TensorBoard(
    log_dir=TensorboardDir,
    # histogram_freq=1, # in epochs # If printing histograms, validation_data must be provided, and cannot be a generator.
    write_graph=True,
    # write_grads=True,
    # write_images=True
)

batch_size = 32

class DataIterator:
    
    def __init__(self):
        # zero tensors for each input
        self.ins = [np.zeros((batch_size, t.shape[1])) for t in inputs]
        self.outs = [np.zeros((batch_size, t.shape[1])) for t in outputs]
    
    def __next__(self):
        """Returns the next batch of data. May run in different processes!"""
        return (self.ins, self.outs)

my_data = DataIterator()

model.fit_generator(
    generator=my_data,
    steps_per_epoch=1,
    epochs=1,
    verbose=1,
    workers=4,
    use_multiprocessing=True,
    callbacks=[tensorboardcb]
)

