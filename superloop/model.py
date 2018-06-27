import abc
import keras
from keras import backend as K
import numpy as np
import pickle
import h5py

class Builder:
    """(Re)Builds models from cached layers. Used to build unrolled models
    and set up weight sharing between the same layers in different timesteps.
    Abstract class"""

    def __init__(self, name):
        assert name # We need a name to be able to save/load layers
        self.name = name # Name of the model
        self._shared_layers = {} # layer cache, keyed on _layer_counter
        self._layers_to_save = {} # layers that can be saved, keyed on name
        self._layer_counter = 0
        self._build_counter = 0
        
    def generate_name(self, name):
        """Returns a name for a layer"""
        return "{}/{}.{}".format(self.name, self._layer_counter, name)
        
    def shared_layer(self, build_function, args, kwargs, skip_cache=False, save=False):
        """Either create a layer shared between all units, or return one from the cache"""

        assert 'name' in kwargs # We need a name to be able to save/load layers
        fullname = self.generate_name(kwargs['name'])

        key = self._layer_counter

        if skip_cache or key not in self._shared_layers:
            # Build the layer
            kwargs['name'] = fullname
            layer = build_function(*args, **kwargs)    
            if not skip_cache:
                self._shared_layers[key] = layer
        else:
            layer = self._shared_layers[key]

        if save:
            self._layers_to_save[fullname] = layer
        
        if not skip_cache:
            self._layer_counter += 1
        return layer

    def skip_layer(self, count=1):
        """Call this method if a shared layer is optional to ensure that later layers use the correct key"""
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

    def _save_local_weights(self, h5file):
        """Save the weights of save-able layers into the given h5py file object"""
        for name, layer in self._layers_to_save.items():
            self._save_layer_weights(layer, name, h5file)
            
    def _load_local_weights(self, h5file):
        """Load and overwite the weights of save-able layers"""
        for name, layer in self._layers_to_save.items():
            self._load_layer_weights(layer, name, h5file)

    def _save_layer_weights(self, layer, name, h5file):
        """Save the weights of one layer into a h5py file object"""
        # How to save binary data in h5py: http://docs.h5py.org/en/latest/strings.html
        # How to serialize objects: https://docs.python.org/3.3/library/pickle.html
        weights = layer.get_weights()
        h5file.create_dataset("{}/length".format(name), data=np.array([len(weights)]))
        for idx, array in enumerate(weights):
            h5file.create_dataset("{}/{}".format(name, idx), data=array)
        
    def _load_layer_weights(self, layer, name, h5file):
        """Load the weights of one layer from a h5py file object"""        
        group = h5file[name]
        length = group['length'][0]
        weights = [group["{}".format(idx)] for idx in range(length)]
        layer.set_weights(weights)    

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
                'timesteps': 16, # timesteps to unroll
                'model_name': 'Main', # name of the full model
                'model_inputs': 3, # number of inputs at each timestep (1D tensor)
                'model_outputs': 3, # number of outputs at each timestep (1D tensor) or 0 to disable
                'recurrent_model': SGU, # subclass of Builder
                'recurrent_layers': 5, # number of recurrent layers
                'recurrent_units': 3, # number of recurrent units on each layer
                'superloop_models': [RegisterMemory], # list of SuperLoopModel subclasses 
                    # used to build models used in the superloop
                    
                <sub-dicts keyed by the name of each superloop subclass;
                <these are passed to the superloop model constructors>
            }
        """

        super().__init__(name=config['model_name'], **kwargs)
        
        self.config = config

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


    def save_weights(self, h5filename):
        """Save weights from the model in a hdf5 file"""
        h5file = h5py.File(h5filename, "w")
        self._save_local_weights(h5file)
        for s in self.superloop_models:
            s._save_local_weights(h5file)
        for r in self.recurrent_layers:
            r._save_local_weights(h5file)
        
    
    def load_weights(self, h5filename):
        """Load and overwrite weights in the model from a hdf5 file"""
        h5file = h5py.File(h5filename, "r")
        self._load_local_weights(h5file)
        for s in self.superloop_models:
            s._load_local_weights(h5file)
        for r in self.recurrent_layers:
            r._load_local_weights(h5file)

        
    def _build_impl(self, input, skip_superloop=False):
        """Implements building the model in one timestep"""
        
        print("Building timestep {}...".format(self._build_counter))
        
        if self._build_counter == 0:
            # Use 0s as the input from the superloop in the first timestep
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

        # Recurrent layers        
        for rlayer in self.recurrent_layers:
            x = rlayer.build(x)
            
        x = self.shared_layer(keras.layers.Dense, (), {'units': self.all_outputs, 'name':'DenseFinal'}, save=True)(x)

        # Output: crop out data needed for the superloop models
        if self.outputs > 0:
            output = self.shared_layer(CropLayer, (), {'start':0, 'end':self.outputs, 'name':'CropOut'})(x) # output
        else:
            output = None

        if skip_superloop:
            return output

        start = self.outputs
        for s in self.superloop_models:        
            s.build(self.shared_layer(CropLayer, (), {'start':start, 'end':start+s.inputs, 'name':"Crop{}".format(type(s).__name__)})(x))
            start += s.inputs

        return output


    def build_all(self):
        """The main method to call to build the full model.
        """
    
        # The input to the RNN part
        input = keras.layers.Input(shape=(self.config['timesteps'], self.config['model_inputs']), name="{}/Input".format(self.config['model_name']))
        
        outputs = [None] * self.config['timesteps']
        
        for timestep in range(self.config['timesteps']):
            o = self.build(
                keras.layers.Lambda( # see 'input' of _build_impl. Split the input tensor
                    lambda x: x[:, timestep, :],
                    name="{}/CropInput{}".format(self.config['model_name'], timestep)
                )(input),
                skip_superloop=(timestep == self.config['timesteps']-1)
            )
            if self.config['model_outputs'] > 0:
                outputs[timestep] = keras.layers.Lambda(
                    lambda x: K.expand_dims(o, axis=-2), # (outputs) -> (1,outputs)
                    name="{}/ExpandOut{}".format(self.config['model_name'], timestep)
                )(o)
            
        # Merge outputs
        if self.config['model_outputs'] > 0:
            output = keras.layers.Concatenate(axis=-2, name="{}/ConcatOut".format(self.config['model_name']))(outputs)
        else:
            output = None
            
        return (input, output) # RNN input and output tensors
