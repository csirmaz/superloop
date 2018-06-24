import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os

import superloop

"""
Toy example for the attention superloop which should find the last 2. value before a 9. value
"""

CONFIG = {
    'timesteps': 64, # timesteps to unroll
    'model_name': 'Main', # name of the full model
    'model_inputs': 2, # number of inputs at each timestep (1D tensor)
    'model_outputs': 0,
    'recurrent_model': superloop.SGU,
    'recurrent_layers': 16, # number of recurrent layers
    'recurrent_units': 8, # number of recurrent units on each layer
    'superloop_models': [superloop.Attention], # classes used to build models used in the superloop
    'Attention': {
        'datapoints': 12,
        'outputs': 1,
    }
}

slmodel = superloop.Model(CONFIG)
(input, dummy_output) = slmodel.build_all()

model = keras.models.Model(inputs=[input, slmodel.superloop_models[0].data], outputs=slmodel.superloop_models[0].position)
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.RMSprop(lr=0.0004),
              metrics=['accuracy'])
model.summary()

TensorboardDir = "{}/tensorboard_logs/superloop_attn_rmsprop_leak_slow_big".format(os.environ['HOME'])
os.system("mkdir -p {}".format(TensorboardDir))

# https://keras.io/callbacks/#tensorboard
tensorboardcb = keras.callbacks.TensorBoard(
    log_dir=TensorboardDir,
    # histogram_freq=1, # in epochs # If printing histograms, validation_data must be provided, and cannot be a generator.
    write_graph=True,
    # write_grads=True,
    # write_images=True
)

samples = 1024*128
datapoints = CONFIG['Attention']['datapoints']
data = np.zeros((samples, datapoints,1))
target = np.zeros((samples,))
for sample in range(samples):
    peak = np.random.randint(0, datapoints-1)+1
    solution = np.random.randint(0, peak)
    for d in range(datapoints):
        if d < solution:
            v = np.random.randint(0, 5) # any value
        elif d == solution:
            v = 2.
        elif d < peak:
            v = np.random.randint(0, 4)
            if v >= 2.:
                v += 1.
        elif d == peak:
            v = 9.
        else:
            v = np.random.randint(0, 5) # any value

        data[sample][d][0] = v
    target[sample] = solution

# for x in range(samples):        
#     print(data[x])
#     print(target[x])

model.fit(
    x=[
       np.zeros((data.shape[0], input.shape[1], input.shape[2])),
       data
    ],
    y=target,
    epochs=40000,
    callbacks=[tensorboardcb]
)

 

