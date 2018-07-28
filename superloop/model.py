import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import h5py

from .builder import Builder, ExtendWithZeros

# The below are only needed to parse the configuration
from .sgu import SGU
from .regmem import RegisterMemory
from .attention import Attention


class Model(Builder):
    """A Model object is responsible for building a full unrolled 'superloop' model.
    
    A superloop model has a central RNN part with an additional superloop that connects its
    output to its input in the next timestep via some external systems.
    """
    
    @classmethod
    def txt2class(cls, txt):
        """Translates a name into a class.
        
        Arguments:
        - txt -- the name.
        """
        return {
            'SGU': SGU,
            'RegisterMemory': RegisterMemory,
            'Attention': Attention
        }[txt]
        
    
    def __init__(self, config):
        """Constructor.
        
        Arguments:
        - config -- a dict with the following keys:
            - 'timesteps' -- int; timesteps to unroll
            - 'suppress_output' -- int; suppress RNN output for this many timesteps at the beginning
            - 'model_name' -- string; the name of the full model
            - 'model_inputs' -- int; number of RNN inputs at each timestep (1D tensor)
                or 0 to disable, in which case the returned input tensor must still be used as an input and set to 0
            - 'model_outputs' -- int; the number of RNN outputs at each timestep (1D tensor) or 0 to disable
            - 'recurrent_model' -- string; name of model class to use to build the RNN part, e.g. 'SGU'
            - 'recurrent_layers' -- int; number of recurrent layers in the RNN part
            - 'recurrent_units' -- int; number of recurrent units on each layer in the RNN part
                'superloop_models': [RegisterMemory], # list of SuperLoopModel subclass names 
                    # used to build models used in the superloop
                'printvalues': False, # whether to print some tensor values. If yes, use an int which is the maximum number of values displayed
                    
                <sub-dicts keyed by the name of each superloop subclass;
                <these are passed to the superloop model constructors>
            }
        """

        super().__init__(name=config['model_name'], printvalues=config['printvalues'])
        
        self.config = config

        # Array of builders for the recurrent layers
        self.recurrent_layers = [
            self.txt2class(config['recurrent_model'])(units=config['recurrent_units'], name="{}/Recur{}".format(self.name, layerix), printvalues=config['printvalues']) 
            for layerix in range(config['recurrent_layers'])
        ]
        
        # superloop models: the external systems connected via the superloop ("X")
        self.superloop_models = [
            modelclass(name="{}/{}".format(self.name, modelclass.__name__), printvalues=config['printvalues'], config=config[modelclass.__name__])
            for modelclass in (self.txt2class(modelclass) for modelclass in config['superloop_models'])
        ]
        
        self.outputs = config['model_outputs']
        self.noinput_input = None # set to the initial input from the superloop if model_inputs==0


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
        
        if self._build_counter == 0:
            # Use 0s as the input from the superloop in the first timestep
            super_inputs = sum(s.outputs for s in self.superloop_models)
            if input is not None: # some RNN inputs exist
                x = self.shared_layer(ExtendWithZeros, (), {'size_added':super_inputs, 'name':'ExtendZero'})(input)
            else:
                self.noinput_input = keras.layers.Input(shape=(super_inputs,), name="{}/NoInputInput".format(self.config['model_name']))
                x = self.noinput_input
                self.skip_layer(1)
        else:
            inputs = []
            if input is not None: inputs.append(input) # RNN inputs exist
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
        
        if self.outputs > 0 and self._build_counter >= self.config['suppress_output']:
            output = self.shared_layer(keras.layers.Dense, (), {'units': self.outputs, 'name':'DenseOutMain'}, save=True)(x)
        else:
            output = None
            self.skip_layer(1)

        if skip_superloop:
            return output

        for s in self.superloop_models:
            s.build(self.shared_layer(
                keras.layers.Dense, 
                (), 
                {'units':s.inputs, 'name':"DenseOut{}".format(type(s).__name__)}, 
                save=True, 
                initfn=s.init_dense
            )(x))

        return output


    def build_all(self):
        """The main method to call to build the full model.
        """
    
        # The input to the RNN part
        if self.config['model_inputs'] > 0:
            rnninput = keras.layers.Input(shape=(self.config['timesteps'], self.config['model_inputs']), name="{}/Input".format(self.config['model_name']))
        else:
            rnninput = None
        
        outputs = [None] * (self.config['timesteps'] - self.config['suppress_output'])
    
        for timestep in range(self.config['timesteps']):
            print("Building timestep {}/{}...".format(self._build_counter + 1, self.config['timesteps']))
        
            if rnninput is None:
                stepinput = None
            else:
                stepinput = keras.layers.Lambda( # see 'input' of _build_impl. Split the input tensor
                    lambda x: x[:, timestep, :],
                    name="{}/CropInput{}".format(self.config['model_name'], timestep)
                )(rnninput)

            o = self.build(stepinput, skip_superloop=(timestep == self.config['timesteps']-1))
            if self.config['model_outputs'] > 0 and timestep >= self.config['suppress_output']:
                outputs[timestep - self.config['suppress_output']] = keras.layers.Lambda(
                    lambda x: K.expand_dims(o, axis=-2), # (outputs) -> (1,outputs)
                    name="{}/ExpandOut{}".format(self.config['model_name'], timestep)
                )(o)
            
        # Merge outputs
        if self.config['model_outputs'] > 0:
            rnnoutput = keras.layers.Concatenate(axis=-2, name="{}/ConcatOut".format(self.config['model_name']))(outputs)
        else:
            rnnoutput = None
            
        if self.config['model_inputs'] == 0:
            # We return the dummy input here that is the input from the superloop in timestep 0.
            # No combination of Lamba, Input, K.constant, K.zeros, &c. worked.
            rnninput = self.noinput_input
        return (rnninput, rnnoutput) # RNN input and output tensors
