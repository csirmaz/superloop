import abc
import keras
from keras import backend as K
import tensorflow as tf


class Builder:
    """(Re)Builds models from cached layers. Used to build unrolled models
    and set up weight sharing between the same layers in different timesteps.
    Abstract class"""

    def __init__(self, name=None):
        self.name = name # Name of the model
        self._shared_layers = {} # layer cache, keyed on _layer_counter
        self._layer_counter = 0
        self._build_counter = 0
        
    def generate_name(self, name):
        return "{}/{}.{}".format(self.name, self._layer_counter, name)

    def shared_layer(self, build_function, args, kwargs, skip_cache=False):
        """Either create a layer shared between all units, or return one from the cache"""
        key = self._layer_counter
        if skip_cache or key not in self._shared_layers:
            # Build the layer
            if self.name and ('name' in kwargs): # Generate a name
                kwargs['name'] = self.generate_name(kwargs['name'])
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


class Model(Builder):
    """Builds the full model in one timestep"""
    
    def __init__(self, config, **kwargs):
        """
            config = {
                'model_outputs': 3, # number of outputs at each timestep
                'recurrent_model': SGU, # subclass of Builder
                'recurrent_layers': 5, # number of recurrent layers
                'recurrent_units': 3, # number of recurrent units on each layer
                'superloop_models': [RegisterMemory], # list of SuperLoopModel subclasses 
                    # used to build models used in the superloop
                    
                <sub-dicts keyed by the name of each superloop subclass;
                <these are passed to the superloop model constructors>
            }
        """

        super().__init__(**kwargs)

        # Array of builders for the recurrent layers
        self.recurrent_layers = [
            config['recurrent_model'](units=config['recurrent_units'], name="{}/Recur{}".format(self.name, layerix)) 
            for layerix in range(config['recurrent_layers'])
        ]
        
        # superloop models: the external systems connected via the superloop ("X")
        self.superloop_models = [
            modelclass(name="{}/{}".format(self.name, modelclass.__name__), config=config[modelclass.__name__])
            for modelclass in config['superloop_models']
        ]
        
        self.outputs = config['model_outputs']
        self.all_outputs = self.outputs + sum(s.inputs for s in self.superloop_models)

        
    def _build_impl(self, input, skip_superloop=False):
        """Implements building the model in one timestep"""
        
        print("Building timestep {}...".format(self._build_counter))
        
        if self._build_counter == 0:
            super_inputs = sum(s.outputs for s in self.superloop_models)
            x = self.shared_layer(ExtendWithZeros, (), {'size_added':super_inputs, 'name':'ExtendZero'})(input)
        else:
            inputs = [input]
            inputs.extend(s.out for s in self.superloop_models)
            if len(inputs) > 1:
                x = keras.layers.concatenate(
                    inputs,
                    name="{}/ConcatSuper{}".format(self.name, self._build_counter)
                )
            else:
                x = inputs[0]
            self.skip_layer(1)
        
        for rlayer in self.recurrent_layers:
            x = rlayer.build(x)
            
        x = self.shared_layer(keras.layers.Dense, (), {
            'units': self.all_outputs, 
            'name':'DenseFinal'
        })(x)
        
        output = self.shared_layer(CropLayer, (), {'start':0, 'end':self.outputs, 'name':'CropOut'})(x) # output

        if skip_superloop:
            return output

        start = self.outputs
        for s in self.superloop_models:        
            s.build(self.shared_layer(CropLayer, (), {'start':start, 'end':start+s.inputs, 'name':"Crop{}".format(type(s).__name__)})(x))
            start += s.inputs

        return output


def build_model(config):
    """
        config = {
            'timesteps': 16, # timesteps to unroll
            'model_name': 'Main', # name of the full model
            'model_inputs': 3, # number of inputs at each timestep (1D tensor)
            'model_outputs': 3, # number of outputs at each timestep (1D tensor)
            'recurrent_model': SGU, # subclass of Builder
            'recurrent_layers': 5, # number of recurrent layers
            'recurrent_units': 3, # number of recurrent units on each layer
            'superloop_models': [RegisterMemory], # list of SuperLoopModel subclasses 
                # used to build models used in the superloop
                
            <sub-dicts keyed by the name of each superloop subclass;
            <these are passed to the superloop model constructors>
        }
    """
    
    input = keras.layers.Input(shape=(config['timesteps'], config['model_inputs']), name="{}/Input".format(config['model_name']))
    
    builder = Model(name=config['model_name'], config=config)
    outputs = [None] * config['timesteps']
    
    for timestep in range(config['timesteps']):
        o = builder.build(
            keras.layers.Lambda( # split input tensor
                lambda x: x[:, timestep, :],
                name="{}/CropInput{}".format(config['model_name'], timestep)
            )(input),
            skip_superloop=(timestep == config['timesteps']-1)
        )
        outputs[timestep] = keras.layers.Lambda(
            lambda x: K.expand_dims(o, axis=-2), # (outputs) -> (1,outputs)
            name="{}/ExpandOut{}".format(config['model_name'], timestep)
        )(o)
        
    # Merge outputs
    output = keras.layers.Concatenate(axis=-2, name="{}/ConcatOut".format(config['model_name']))(outputs)
        
    return (input, output)
