import keras
from keras import backend as K

from .model import Builder, ExtendWithZeros, PrintTensor

# TODO implement as layer? https://keras.io/layers/writing-your-own-keras-layers/

class SGU(Builder):
    """Builds a layer Simple Gated Units in a way that they can be unrolled in time."""
    
    def __init__(self, units, **kwargs):
        self.units = units # number of units on the layer
        super().__init__(**kwargs)
        # The hidden state / internal input to use in the next timestep
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
            
        f = self.shared_layer(keras.layers.Dense, (), 
            {'units':self.units, 'activation':'sigmoid', 'name':'DenseCtrl'}, save=True)(allin) # W1, W2
        # f = PrintTensor("f=sigmoid()")(f) # DEBUG
        
        inp = self.shared_layer(keras.layers.Dense, (), 
            {'units':self.units, 'name':'DenseIn'}, save=True)(external_input) # W3
        inp = self.shared_layer(keras.layers.LeakyReLU, (), {'alpha':0.1, 'name':'ReLU'})(inp)
        # inp = PrintTensor("relu(inp)")(inp) # DEBUG
        
        if self._build_counter == 0:
            # (1-f)*inp
            # We can do 1-x[0] due to broadcasting
            out = self.shared_layer(keras.layers.Lambda, ((lambda x: (1.0 - x[0]) * x[1]),), {'name':'nF_Inp'})([f, inp])
            self.skip_layer(1) # Order is important - Cache the correct layer
        else:
            self.skip_layer(1) # Order is important - Cache the correct layer
            # (1-f)*inp + f*internal
            out = self.shared_layer(keras.layers.Lambda, ((lambda x: (1.0-x[0])*x[1] + x[0]*x[2]),), {'name':'nF_Inp_F_Int'})([f, inp, self.internal_var])

        self.internal_var = out # new internal / hidden state
        return out # external_output
