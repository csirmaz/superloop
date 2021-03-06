import keras
from keras import backend as K

from .builder import Builder, ExtendWithZeros


# TODO implement as layer? https://keras.io/layers/writing-your-own-keras-layers/

class SGU(Builder):
    """Builds a layer of Simple Gated Units (SGUs) in a way that they can be unrolled in time.
    
    See README.md for a description of its structure.
    """
    
    def __init__(self, units, **kwargs):
        """Constructor.
        
        Arguments:
        - units -- int; the number of units on the layer.
        - any extra keyword arguments are passed to the superclass constructor.
        """
        self.units = units # number of units on the layer
        super().__init__(**kwargs)
        # The hidden state / internal input to use in the next timestep
        self.internal_var = None

    
    def _build_impl(self, external_input):
        """Internal method. Implements building the unit itself using shared_layer().
        
        Arguments:
        - external_input -- tensor; the input to the SGU (which is not its state from the previous timestep).
        """
        
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
            {'units':self.units, 'activation':'softsign', 'name':'DenseCtrl'}, save=True)(allin) # W1, W2 {NONLIN}
        f = self.shared_layer(keras.layers.Lambda, ((lambda x: (x[0] + 1.) / 2.),), {'name':'DenseCtrl2'})([f])
        # f = PrintTensor("f=sigmoid()")(f) # DEBUG
        
        inp = self.shared_layer(keras.layers.Dense, (), 
            {'units':self.units, 'name':'DenseIn'}, save=True)(external_input) # W3
        inp = self.shared_layer(keras.layers.LeakyReLU, (), {'alpha':0.1, 'name':'ReLU'})(inp) # {NONLIN}
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

        # During visualization, we display the internal state
        out = self.print_layer(out, "SGU_Internal")

        self.internal_var = out # new internal / hidden state
        return out # external_output
