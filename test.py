import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os

import superloop


CONFIG = {
    'timesteps': 3, # timesteps to unroll
    'model_name': 'Main', # name of the full model
    'model_inputs': 7, # number of inputs at each timestep (1D tensor)
    'model_outputs': 3,
    'recurrent_model': superloop.SGU,
    'recurrent_layers': 5, # number of recurrent layers
    'recurrent_units': 3, # number of recurrent units on each layer
    'superloop_models': [superloop.RegisterMemory], # classes used to build models used in the superloop
    'RegisterMemory': {
        'register_width': 32,
        'depth': 7,
    }
}

slmodel = superloop.Model(CONFIG)
(input, output) = slmodel.build_all()

model = keras.models.Model(inputs=input, outputs=output)
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
        self.ins = np.zeros((batch_size, input.shape[1], input.shape[2]))
        self.outs = np.zeros((batch_size, output.shape[1], output.shape[2]))
    
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

