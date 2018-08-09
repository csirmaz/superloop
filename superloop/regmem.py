import keras
from keras import backend as K
import tensorflow as tf

from .builder import SuperLoopModel, CropLayer


class RegisterMemory(SuperLoopModel):
    """Implements a memory of multiple registers that can be used as an external system in a superloop.
    
    There are D ("depth") registers, each W ("width") wide. The RNN part of the model can
    control writing and reading the registers, and receives the data read in the next timestep.
    
    Inputs of the model:
        - data (W) -- this is stored in the selected register(s)
        - store control (D) -- these control which register is written (a softmax is applied)
        - recall control (D) -- these control which register is read (a softmax is applied)
    Outputs of the model:
        - data (W) -- the data read from the registers
    """
    
    """    
    memory (depth,width)
    
    (depth,width)   (depth)   (depth,width)   (depth) (width)
    memory[t]     = (1-store)*memory[t-1]   + store  *data   (uses matrix multiplication)
    
    (width)   (depth) (depth,width)
    out     = recall *memory[t]     (uses element-wise multiplication)
    """
    
    def __init__(self, config, **kwargs):
        """Constructor.
        
        Arguments:
        - config -- a dict with the following keys:
            - 'register_width' -- the size of each register
            - 'depth' -- the number of registers
        - any extra keyword arguments are passed to the superclass constructor.
        """        
        self.register_width = config['register_width']
        self.depth = config['depth']
        super().__init__(
            inputs=self.register_width + 2*self.depth, # store, recall
            outputs=self.register_width,
            **kwargs
        )
        self.memory = None # The memory registers with (depth,width) shape


    def _build_impl_impl(self, input):
        """Internal method. Implements building the unit itself using shared_layer().
        
        Arguments:
        - input -- tensor; all inputs to the model
        """
        store = self.shared_layer(CropLayer, (), {'start':0, 'end':self.depth, 'name':"CropStore"})(input)
        recall = self.shared_layer(CropLayer, (), {'start':self.depth, 'end':self.depth*2, 'name':"CropRecall"})(input)
        data = self.shared_layer(CropLayer, (), {'start':self.depth*2, 'end':self.inputs, 'name':"CropData"})(input)
        
        store = self.shared_layer(keras.layers.Softmax, (), {'name':'Softmax'})(store) # {NONLIN}
        recall = self.shared_layer(keras.layers.Softmax, (), {'name':'Softmax'})(recall) # {NONLIN}

        # storing = store*data
        # store(depth) data(width) --> (depth,1) (1,width) --> storing(depth,width)
        storing = self.shared_layer(keras.layers.Lambda, ((
            lambda x: tf.matmul(K.expand_dims(x[0], axis=-1), K.expand_dims(x[1], axis=-2))
        ),), {'name':'Storing'})([store, data])
        
        # Calculation for (1-store)*memory - each register is multiplied by a scalar in the retain=(1-store) vector
        def retain_calc(x):
            _retain = 1.0 - x[0]
            _retain = K.expand_dims(_retain, axis=-1) # (depth) -> (depth,1)
            # We shouldn't need the below as multiplication will broadcast https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
            # _retain = K.repeat_elements(_retain, self.register_width, axis=-1) # (depth,1) -> (depth,width)
            return _retain*x[1] # element-wise multiplication
        
        if self._build_counter == 0:
            self.memory = storing
            self.skip_layer(2)
        else:
            # retain = (1-store)*memory
            retain = self.shared_layer(keras.layers.Lambda, (retain_calc,), {'name':'Retain'})([store, self.memory])
            self.memory = self.shared_layer(keras.layers.Lambda, ((lambda x: x[0]+x[1]),), {'name':'Add'})([retain, storing])
        
        # recalled = recall*memory
        # recall(depth) memory(depth,width) --> (1,depth) (depth,width) --> (1,width) --> data(width)
        recalled = self.shared_layer(keras.layers.Lambda, ((
            lambda x: K.squeeze(tf.matmul(K.expand_dims(x[0], axis=-2), x[1]), axis=1)
        ),), {'name':'Recall'})([recall, self.memory])
        return recalled
