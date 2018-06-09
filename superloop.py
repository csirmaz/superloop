import keras
import os
import numpy as np


class Builder:
    """Implements building and rebuilding models from cached layers"""

    def __init__(self, name=None):
        self.name = name # Name of the model
        self._shared_layers = {} # layer cache, keyed on _layer_counter
        self._layer_counter = 0
        self._build_counter = 0

    def shared_layer(self, build_function, args, kwargs):
        """Either create a layer shared between all units, or return one from the cache"""
        key = self._layer_counter
        if key not in self._shared_layers:
            # Build the layer
            if self.name and ('name' in kwargs): # Generate a name
                kwargs['name'] = "{}/{}.{}".format(self.name, self._layer_counter, kwargs['name'])
            self._shared_layers[key] = build_function(*args, **kwargs)
        
        self._layer_counter += 1
        return self._shared_layers[key]

    def build(self, *args, **kwargs):
        """Build the model"""
        self._layer_counter = 0
        self._build_counter += 1
        return self._build_impl(*args, **kwargs)


class RecurrentUnit(Builder):
    """(Re)builds a layer of recurrent units"""
    
    def __init__(self, units, **kwargs):
        self.units = units # number of units on the layer
        super().__init__(**kwargs)
        # The internal input to use in the next timestep
        ## self.internal_var = keras.layers.Input(tensor=keras.backend.zeros((self.units)), name=self.name + '.ZeroIn')
        self.zero_in = keras.layers.Input(shape=(self.units,), name=self.name + '/ZeroIn')
        self.internal_var = self.zero_in
       
    def _build_impl(self, external_input): # TODO Abstract method
        """Implements building the unit itself using shared_layer()"""
        
        # Vanilla RNN
        
        x = keras.layers.concatenate(
            [external_input, self.internal_var],
            name="{}/ConcatRecur.{}".format(self.name, self._build_counter)
        )
        # TODO not arbitrary connections!!
        x = self.shared_layer(keras.layers.Dense, (), {'units': self.units, 'name': 'dense'})(x)
        x = self.shared_layer(keras.layers.Activation, ('relu',), {'name': 'relu'})(x)

        # Split layer: https://github.com/keras-team/keras/issues/890
        self.internal_var = x
        return x # external_output


class SuperLoopModel(Builder):
    """(Re)builds the model used in the superloop"""
    
    def __init__(self, **kwargs):
        self.outputs = 4 # Number of outputs. We need to know this in advance.
        super().__init__(**kwargs)
        # Use this instead of this model's output in the first timestep
        ## self.out = keras.layers.Input(tensor=keras.backend.zeros((self.outputs)), name=self.name + '.ZeroIn')
        self.zero_in = keras.layers.Input(shape=(self.outputs,), name=self.name + '/ZeroIn')
        self.out = self.zero_in

    def _build_impl(self, input): # TODO Abstract method
        """Implements building the unit itself using shared_layer()"""
        
        x = self.shared_layer(keras.layers.Dense, (), {'units': self.outputs, 'name': 'dense'})(input)
        
        self.out = x
        return self.out # output
        
class Model(Builder):
    """Builds the model in one timestep"""
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.config = {
            'recurrent_layers': 5, # number of recurrent layers
            'recurrent_units': 3, # number of recurrent units on each layer
        }

        # Array of builders for the recurrent layers
        self.recurrent_layers = [
            RecurrentUnit(units=self.config['recurrent_units'], name="{}/Recur{}".format(self.name, layerix)) 
            for layerix in range(self.config['recurrent_layers'])
        ]
        
        # The external system connected via the superloop ("X")
        self.superloop_model = SuperLoopModel(name="{}/Super".format(self.name))
        
        
    def get_internal_inputs(self):
        inputs = [l.zero_in for l in self.recurrent_layers]
        inputs.append(self.superloop_model.zero_in)
        return inputs
        

    def _build_impl(self, input): # TODO Abstract method
        """Implements building the model in one timestep"""
        x = keras.layers.concatenate(
            [input, self.superloop_model.out],
            name="{}/ConcatSuper.{}".format(self.name, self._build_counter)
        )
        for layerix in range(self.config['recurrent_layers']):

            x = self.shared_layer(keras.layers.Dense, (), {
                'units': self.config['recurrent_units'], 
                'name': "main_dense_{}".format(layerix) 
            })(x)

            x = self.recurrent_layers[layerix].build(x)
                
        self.superloop_model.build(x) # TODO split
        return x # output


# keras.layers.Input(shape=(10,), name="name1")
# x = keras.layers.Dense(units=11, name="name2")(x)
# keras.layers.TimeDistributed(keras.layers.Dense(12), name="name4")

timesteps = 3
inputs = [keras.layers.Input(shape=(7,), name="Main/StepInput.{}".format(i)) for i in range(timesteps)]
mymodelbuilder = Model(name="Main")
outputs = [None]*timesteps

for timestep in range(timesteps):
    outputs[timestep] = mymodelbuilder.build(inputs[timestep])

inputs.extend(mymodelbuilder.get_internal_inputs())

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

