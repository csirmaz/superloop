import keras
from keras import backend as K
import tensorflow as tf

from .model import Builder, ExtendWithZeros


class SGU(Builder):
    """ Simple Gated Unit. Builds a layer of recurrent units."""
    
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
        
        # Unfortunately, keras.layers.Subtract &c. don't have names, so the graph is unusable. We use Lambdas instead
        inp = self.shared_layer(keras.layers.Dense, (), {'units':self.units, 'name':'DenseIn'})(external_input) # W3
        inp = self.shared_layer(keras.layers.Activation, ('relu',), {'name':'ReLU'})(inp)
        # inp = PrintTensor("relu(inp)")(inp) # DEBUG
        # We can do 1-x[0] due to broadcasting
        inp = self.shared_layer(keras.layers.Lambda, ((lambda x: (1.0 - x[0]) * x[1]),), {'name':'MinMultIn'})([f, inp])
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
