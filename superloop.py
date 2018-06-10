import abc
import keras
from keras import backend as K
import numpy as np
import os


class Builder:
    """Implements building and rebuilding models from cached layers. Abstract class"""

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

    @abc.abstractmethod
    def _build_impl(self):
        """Actually build the model"""


def CropLayer(start, end, step=1, name=None):
    """Create a layer that crops out a range from a 1D (+batch) input tensor"""
    # See https://github.com/keras-team/keras/issues/890
    return keras.layers.Lambda(lambda x: x[:, start:end:step], output_shape=(int((end-start)/step),), name=name)
    # Could use K.slice but unsure how to manage batch dimension

def PrintTensor(message):
    return keras.layers.Lambda(lambda x: K.print_tensor(x, message=message))

class RecurrentUnit(Builder):
    """Builds a layer of recurrent units"""
    
    def __init__(self, units, **kwargs):
        self.units = units # number of units on the layer
        super().__init__(**kwargs)
        # The internal input to use in the next timestep
        # We use an input here as the first dimension (batch size) is not fixed,
        # so we cannot use a constant tensor.
        self.first_in = keras.layers.Input(shape=(self.units,), name=self.name + '/FirstIn')
        self.internal_var = self.first_in


    def _build_impl(self, external_input):
        """Implements building the unit itself using shared_layer()"""
        
        # We implement a very simple gated unit here
        #
        # Dense layer
        # Split the tensor into Input and Control
        # F = Sigmoid(Control)
        # Output = F*Internal + (1-F)*ReLU(Input)
        
        dense = self.shared_layer(keras.layers.Dense, (), {'units':self.units*2, 'name':'Dense'})(external_input)

        f = self.shared_layer(CropLayer, (0, self.units), {'name':'CropF'})(dense) # Probably no need to use shared_layer but the layer will be named properly
        f = self.shared_layer(keras.layers.Activation, ('hard_sigmoid',), {'name':'Sigm'})(f)
        # f = PrintTensor("f=sigmoid(ctrl)")(f) # DEBUG
        
        ones = self.shared_layer(keras.layers.Lambda, ((lambda x: K.ones_like(x)),), {'name':'Ones'})(f)
        # Unfortunately, keras.layers.Subtract &c. don't have names, so the graph is unusable
        ## onesf = self.shared_layer(keras.layers.Subtract, (), {})([ones, f])
        onesf = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]-x[1]),), {'name':'Sub'})([ones, f])
        # onesf = PrintTensor("1-f")(onesf) # DEBUG

        inp = self.shared_layer(CropLayer, (self.units, self.units*2), {'name':'CropIn'})(dense)
        inp = self.shared_layer(keras.layers.Activation, ('relu',), {'name':'ReLU'})(inp)
        # inp = PrintTensor("relu(inp)")(inp) # DEBUG
        ## inp = keras.layers.Multiply()([onesf, inp])
        inp = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]*x[1]),), {'name':'MultInp'})([onesf, inp])
        # inp = PrintTensor("(1-f)*relu(inp)")(inp) # DEBUG

        ## internal = self.shared_layer(keras.layers.Multiply, (), {})([f, self.internal_var])
        internal = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]*x[1]),), {'name':'MultInt'})([f, self.internal_var])
        ## out = self.shared_layer(keras.layers.Add, (), {})([inp, internal])
        out = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]+x[1]),), {'name':'Add'})([inp, internal])

        self.internal_var = out
        return out # external_output


class SuperLoopModel(Builder):
    """Builds the model used in the superloop"""
    
    def __init__(self, **kwargs):
        self.outputs = 4 # Number of outputs. We need to know this in advance.
        super().__init__(**kwargs)
        # Use this instead of this model's output in the first timestep
        ## self.out = keras.layers.Input(tensor=keras.backend.zeros((self.outputs)), name=self.name + '.ZeroIn')
        self.first_out = keras.layers.Input(shape=(self.outputs,), name=self.name + '/FirstOut')
        self.out = self.first_out # Always contains the output of the latest build

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
        """Return all the input placeholders automatically generated for the first timestep"""
        inputs = [l.first_in for l in self.recurrent_layers]
        inputs.append(self.superloop_model.first_out)
        return inputs
        

    def _build_impl(self, input): # TODO Abstract method
        """Implements building the model in one timestep"""
        x = keras.layers.concatenate(
            [input, self.superloop_model.out],
            name="{}/ConcatSuper{}".format(self.name, self._build_counter)
        )
        
        for layerix in range(self.config['recurrent_layers']):

            x = self.shared_layer(keras.layers.Dense, (), {
                'units': self.config['recurrent_units'], 
                'name': "Dense{}".format(layerix) 
            })(x)

            x = self.recurrent_layers[layerix].build(x)
                
        self.superloop_model.build(x) # TODO split
        return x # output



timesteps = 3
inputs = [keras.layers.Input(shape=(7,), name="Main/StepInput{}".format(i)) for i in range(timesteps)]
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

