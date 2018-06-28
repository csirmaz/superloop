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
    'recurrent_layers': 32, # number of recurrent layers
    'recurrent_units': 16, # number of recurrent units on each layer
    'superloop_models': [superloop.Attention], # classes used to build models used in the superloop
    'Attention': {
        'datapoints': 16,
        'outputs': 1,
    },
    'steps_per_epoch': 256,
    'batch_size': 64,
    'epochs': 10000,
    'tensorboard_logs': '{}/tensorboard_logs/superloop_attn_save2',
    'model_file': 'out2.h5',
    'save_model': True,
    'load_model': False
}

slmodel = superloop.Model(CONFIG)
(input, dummy_output) = slmodel.build_all()

if CONFIG['load_model']:
    slmodel.load_weights(CONFIG['model_file'])

model = keras.models.Model(inputs=[input, slmodel.superloop_models[0].data], outputs=slmodel.superloop_models[0].position)
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.RMSprop(lr=0.0004),
              metrics=['accuracy'])
model.summary()


# Callbacks

TensorboardDir = CONFIG['tensorboard_logs'].format(os.environ['HOME'])
os.system("mkdir -p {}".format(TensorboardDir))

# https://keras.io/callbacks/#tensorboard
tensorboardcb = keras.callbacks.TensorBoard(
    log_dir=TensorboardDir,
    # histogram_freq=1, # in epochs # If printing histograms, validation_data must be provided, and cannot be a generator.
    write_graph=True,
    # write_grads=True,
    # write_images=True
)


class MyCallback(keras.callbacks.Callback):
    """Custom training callback"""

    def __init__(self):
        super(MyCallback, self).__init__()
        self.loss = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Save model
        if self.loss is None or self.loss > logs['loss']:
            print("Better loss! {}".format(logs['loss']))
            self.loss = logs['loss']
            if CONFIG['save_model']:
                print("Saving model")
                slmodel.save_weights(CONFIG['model_file'])
                print("Saving done")


# Generate training data
class DataIterator:
    """Generate random training data
    
        This consists of lists of (datapoints) numbers...
    """
    def __init__(self, batch_size, datapoints, dummyin):
        self.batch_size = batch_size
        self.datapoints = datapoints
        self.dummyin = dummyin
        
    def __next__(self): 
        data = np.zeros((self.batch_size, self.datapoints, 1))
        target = np.zeros((self.batch_size,))
        for batch in range(self.batch_size):
            peak = np.random.randint(0, self.datapoints-1)+1
            solution = np.random.randint(0, peak)
            for d in range(self.datapoints):
                if d < solution:
                    v = np.random.randint(0, 5) # any value
                elif d == solution:
                    v = 2
                elif d < peak:
                    v = np.random.randint(0, 4)
                    if v >= 2:
                        v += 1
                elif d == peak:
                    v = 9
                else:
                    v = np.random.randint(0, 5) # any value

                data[batch][d][0] = v
            target[batch] = solution
            
        return ([self.dummyin, data], target)


dummyin = np.zeros((CONFIG['batch_size'], input.shape[1], input.shape[2]))
my_data = DataIterator(batch_size=CONFIG['batch_size'], datapoints=CONFIG['Attention']['datapoints'], dummyin=dummyin)

model.fit_generator(
    generator=my_data,
    steps_per_epoch=CONFIG['steps_per_epoch'],
    epochs=CONFIG['epochs'],
    verbose=1,
    workers=2,
    use_multiprocessing=True,
    callbacks=[MyCallback(), tensorboardcb]
)
