import abc
import keras
from keras import backend as K
import numpy as np
import os


class Builder:
    """(Re)Builds models from cached layers. Used to build unrolled models
    and set up weight sharing between the same layers in different timesteps.
    Abstract class"""

    def __init__(self, name=None):
        self.name = name # Name of the model
        self._shared_layers = {} # layer cache, keyed on _layer_counter
        self._layer_counter = 0
        self._build_counter = 0

    def shared_layer(self, build_function, args, kwargs, skip_cache=False):
        """Either create a layer shared between all units, or return one from the cache"""
        key = self._layer_counter
        if skip_cache or key not in self._shared_layers:
            # Build the layer
            if self.name and ('name' in kwargs): # Generate a name
                kwargs['name'] = "{}/{}.{}".format(self.name, self._layer_counter, kwargs['name'])
            if skip_cache:
                return build_function(*args, **kwargs)
            else:
                self._shared_layers[key] = build_function(*args, **kwargs)
        
        self._layer_counter += 1
        return self._shared_layers[key]

    def skip_layer(self, count=1):
        self._layer_counter += count

    def build(self, *args, **kwargs):
        """Build the model"""
        self._layer_counter = 0
        r = self._build_impl(*args, **kwargs)
        self._build_counter += 1
        return r

    @abc.abstractmethod
    def _build_impl(self):
        """Actually build the model"""


def CropLayer(start, end, step=1, name=None):
    """Create a layer that crops out a range from a 1D (+batch) input tensor"""
    # See https://github.com/keras-team/keras/issues/890
    return keras.layers.Lambda(lambda x: x[:, start:end:step], output_shape=(int((end-start)/step),), name=name)
    # Could use K.slice but unsure how to manage batch dimension

def ExtendWithZeros(size_added, name=None):
    """Creates a layer that adds size_added 0's to a 1D tensor in all batches"""
    # See https://stackoverflow.com/questions/46465813/creating-constant-value-in-keras
    def func(x):
        zeros = K.zeros((1, size_added))
        batch_size = K.shape(x)[0]
        tiled_zeros = K.tile(zeros, (batch_size, 1))
        return K.concatenate([x, tiled_zeros])
    return keras.layers.Lambda(func, name=name)

def PrintTensor(message):
    """Create a layer that prints the tensor"""
    return keras.layers.Lambda(lambda x: K.print_tensor(x, message=message))


class RecurrentUnit(Builder):
    """Builds a layer of recurrent units"""
    
    def __init__(self, units, **kwargs):
        self.units = units # number of units on the layer
        super().__init__(**kwargs)
        # The internal input to use in the next timestep
        # We use an input here as the first dimension (batch size) is not fixed,
        # so we cannot use a constant tensor.
        self.internal_var = None


    def _build_impl(self, external_input):
        """Implements building the unit itself using shared_layer()"""
        
        # We implement a simple gated unit here
        # The output and next hidden/internal value are controlled by F:
        #   Output = F*Internal + (1-F)*ReLU(W3*ExternalInput)
        # (The activation here depends more on the external architecture)
        # (Alternatively, instead of ExternalInput we could use a combination 
        # of the internal state and the external input)
        # where F is a function of both the external input and the internal state
        #   F = Sigmoid(W1*Internal + W2*ExternalInput)
        # (The activation here needs to return a number in [0,1])

        if self._build_counter == 0:
            allin = self.shared_layer(ExtendWithZeros, (), {'size_added':self.units, 'name':'ExtendZero'})(external_input)
        else:
            self.skip_layer(1)
            allin = keras.layers.concatenate(
                [external_input, self.internal_var],
                name="{}/ConcatIn{}".format(self.name, self._build_counter)
            )
            
        f = self.shared_layer(keras.layers.Dense, (), {'units':self.units, 'name':'DenseCtrl'})(allin) # W1, W2
        f = self.shared_layer(keras.layers.Activation, ('hard_sigmoid',), {'name':'Sigm'})(f)
        # f = PrintTensor("f=sigmoid()")(f) # DEBUG
        
        ones = self.shared_layer(keras.layers.Lambda, ((lambda x: K.ones_like(x)),), {'name':'Ones'})(f)
        # Unfortunately, keras.layers.Subtract &c. don't have names, so the graph is unusable. We use Lambdas instead
        ## onesf = self.shared_layer(keras.layers.Subtract, (), {})([ones, f])
        onesf = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]-x[1]),), {'name':'Sub'})([ones, f])
        # onesf = PrintTensor("1-f")(onesf) # DEBUG

        inp = self.shared_layer(keras.layers.Dense, (), {'units':self.units, 'name':'DenseIn'})(external_input) # W3
        inp = self.shared_layer(keras.layers.Activation, ('relu',), {'name':'ReLU'})(inp)
        # inp = PrintTensor("relu(inp)")(inp) # DEBUG
        ## inp = keras.layers.Multiply()([onesf, inp])
        inp = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]*x[1]),), {'name':'MultIn'})([onesf, inp])
        # inp = PrintTensor("(1-f)*relu(inp)")(inp) # DEBUG

        if self._build_counter == 0:
            out = inp
            self.skip_layer(2)
        else:
            ## internal = self.shared_layer(keras.layers.Multiply, (), {})([f, self.internal_var])
            internal = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]*x[1]),), {'name':'MultH'})([f, self.internal_var])
            ## out = self.shared_layer(keras.layers.Add, (), {})([inp, internal])
            out = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]+x[1]),), {'name':'Add'})([inp, internal])

        self.internal_var = out
        return out # external_output


class SuperLoopModel(Builder):
    """Builds a model used in the superloop. Abstract class."""
    
    def __init__(self, inputs, outputs, **kwargs):
        self.outputs = outputs # Number of outputs (1D tensor). We need to know this in advance.
        self.inputs = inputs # Number of inputs (1D tensor)

        super().__init__(**kwargs)

        self.out = None # Always contains the output of the latest build

    def _build_impl(self, input): # TODO Abstract method
        """Implements building the unit itself using shared_layer()"""
        
        self.out = self._build_impl_impl(input)
        return self.out # output
        
    
    @abc.abstractmethod
    def _build_impl_impl(self, input):
        """Actually build the model"""


class ShortTermMemory(SuperLoopModel):
    
    def __init__(self, config, **kwargs):
        self.register_width = config['register_width']
        self.depth = config['depth']
        super().__init__(
            inputs=self.register_width + 2*self.depth, # store, recall
            outputs=self.register_width,
            **kwargs
        )
        self.memory = None
        
    def _build_impl_impl(self, input):
        store = self.shared_layer(CropLayer, (), {'start':0, 'end':self.depth, 'name':"CropStore"})(input)
        recall = self.shared_layer(CropLayer, (), {'start':self.depth, 'end':self.depth*2, 'name':"CropRecall"})(input)
        data = self.shared_layer(CropLayer, (), {'start':self.depth*2, 'end':self.inputs, 'name':"CropData"})(input)

        # memory = (1-store)*memory + store*data
        # out = memory * recall
        
        return data


CONFIG = {
    'model_outputs': 3,
    'recurrent_layers': 5, # number of recurrent layers
    'recurrent_units': 3, # number of recurrent units on each layer
    'superloop_models': [ShortTermMemory], # classes used to build models used in the superloop
    'ShortTermMemory': {
        'register_width': 32,
        'depth': 7,
    }
}


class Model(Builder):
    """Builds the full model in one timestep"""
    
    def __init__(self, config, **kwargs):

        super().__init__(**kwargs)

        # Array of builders for the recurrent layers
        self.recurrent_layers = [
            RecurrentUnit(units=config['recurrent_units'], name="{}/Recur{}".format(self.name, layerix)) 
            for layerix in range(config['recurrent_layers'])
        ]
        
        # The external systems connected via the superloop ("X")
        self.superloop_models = [
            modelclass(name="{}/{}".format(self.name, modelclass.__name__), config=config[modelclass.__name__])
            for modelclass in config['superloop_models']
        ]
        
        self.outputs = config['model_outputs']
        self.all_outputs = self.outputs + sum(s.inputs for s in self.superloop_models)
        
        
    def get_internal_inputs(self):
        """Return all the input placeholders automatically generated for the first timestep"""
        # TODO delete
        return []
        

    def _build_impl(self, input):
        """Implements building the model in one timestep"""
        
        if self._build_counter == 0:
            super_inputs = sum(s.outputs for s in self.superloop_models)
            x = self.shared_layer(ExtendWithZeros, (), {'size_added':super_inputs, 'name':'ExtendZero'})(input)
        else:
            inputs = [input]
            inputs.extend(s.out for s in self.superloop_models)
            x = keras.layers.concatenate(
                inputs,
                name="{}/ConcatSuper{}".format(self.name, self._build_counter)
            )
            self.skip_layer(1)
        
        for rlayer in self.recurrent_layers:
            x = rlayer.build(x)
            
        x = self.shared_layer(keras.layers.Dense, (), {
            'units': self.all_outputs, 
            'name':'DenseFinal'
        })(x)

        start = self.outputs
        for s in self.superloop_models:        
            s.build(self.shared_layer(CropLayer, (), {'start':start, 'end':start+s.inputs, 'name':"Crop{}".format(type(s).__name__)})(x))
            start += s.inputs
        return self.shared_layer(CropLayer, (), {'start':0, 'end':self.outputs, 'name':'CropOut'})(x) # output



timesteps = 3
inputs = [keras.layers.Input(shape=(7,), name="Main/StepInput{}".format(i)) for i in range(timesteps)]
mymodelbuilder = Model(name="Main", config=CONFIG)
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

