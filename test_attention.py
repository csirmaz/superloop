import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os

import superloop

"""
Toy example for the attention superloop which should find the 2. value
"""

CONFIG = {
    'timesteps': 3, # timesteps to unroll
    'model_name': 'Main', # name of the full model
    'model_inputs': 2, # number of inputs at each timestep (1D tensor)
    'model_outputs': 0,
    'recurrent_model': superloop.SGU,
    'recurrent_layers': 5, # number of recurrent layers
    'recurrent_units': 3, # number of recurrent units on each layer
    'superloop_models': [superloop.Attention], # classes used to build models used in the superloop
    'Attention': {
        'datapoints': 8,
        'outputs': 1,
    }
}

slmodel = superloop.Model(CONFIG)
(input, dummy_output) = slmodel.build_all()

model = keras.models.Model(inputs=[input, slmodel.superloop_models[0].data], outputs=slmodel.superloop_models[0].position)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

TensorboardDir = "{}/tensorboard_logs/superloop_attn".format(os.environ['HOME'])
os.system("mkdir -p {}".format(TensorboardDir))

# https://keras.io/callbacks/#tensorboard
tensorboardcb = keras.callbacks.TensorBoard(
    log_dir=TensorboardDir,
    # histogram_freq=1, # in epochs # If printing histograms, validation_data must be provided, and cannot be a generator.
    write_graph=True,
    # write_grads=True,
    # write_images=True
)

print(input.shape)

data = np.array([
           [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [2.]],
           [[2.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
           [[3.], [2.], [0.], [0.], [0.], [0.], [0.], [0.]],
           [[3.], [0.], [0.], [0.], [0.], [0.], [0.], [2.]],
           [[1.], [0.], [0.], [0.], [0.], [0.], [2.], [0.]],
           [[1.], [0.], [0.], [0.], [0.], [0.], [2.], [3.]],
           [[1.], [0.], [2.], [0.], [0.], [0.], [0.], [3.]],
           [[1.], [0.], [2.], [3.], [0.], [0.], [0.], [3.]],
           [[2.], [0.], [1.], [3.], [0.], [0.], [0.], [3.]],
       ])

model.fit(
    x=[
       np.zeros((data.shape[0], input.shape[1], input.shape[2])),
       data
    ],
    y=np.array([
        7.,
        0.,
        1.,
        7.,
        6.,
        6.,
        2.,
        2.,
        0.
    ]),
    callbacks=[tensorboardcb]
)

 

