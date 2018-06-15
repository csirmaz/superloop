import keras
from keras import backend as K
import tensorflow as tf

from .model import SuperLoopModel, ExtendWithZeros, PrintTensor

class Attention(SuperLoopModel):
    """Implements a superloop extension that allows the system to focus on different parts of the input.
    It focuses on one data point only, but for non-integer locations, interpolates between the two data points.
    """
    
    def __init__(self, config, **kwargs):
        """
            {
                'data': ... # the data to focus on (2D tensor) (datapoints,outputs)
            }
        """
        self.data = config['data']
        self.position = None
        super().__init__(
            inputs=1, # move control
            outputs=K.shape(self.data)[1],
            **kwargs
        )
    
    def _build_impl_impl(self, input):
        # Limit how much we can move
        move = self.shared_layer(keras.layers.Lambda, ((
            lambda x: K.hard_sigmoid(x) * 2. - 1. # -1. .. 1.
        ),), {'name':'SigmoidCtlr'})(input)
        
        # Add the move control to the position
        if self._build_counter == 0:
            self.position = move
            self.skip_layer()
        else:
            self.position = self.shared_layer(keras.layers.Lambda, ((
                lambda x: x[0]+x[1]
            ),), {'name':'AddPosition'})([self.position, move])

        # We 
        
        def select_impl(x):
            data = x[0] # (datapoints,outputs)
            position = x[1] # (batch_size,1)
            
            # batch_size = K.shape(position)[0]
            indices = K.arange(start=0, stop=K.shape(data)[0], dtype=position.dtype()) # (datapoints)
            # e.g. [0., 1., 2., 3., 4., 5., 6. 7.]
            
            indices = K.expand_dims(indices, axis=1) # (datapoints,1)
            position = K.expand_dims(position, axis=-2) # (batch_size,1,1)
            
            mask = K.minimum(K.maximum(position-indices, 0.), K.maximum(indices+2.-position, 0.)) # (batch_size,datapoints,1)
            #            ooo       ***
            #               ooo ***
            #                **|oo
            #             ***     ooo
            # (p-i) ******           ooooooooooo (i+2-p)
            #       -----|-----|-----|-----|-----
            #
            # result e.g. [.0, .0, .0, .2, .8, .0, .0]
            
            masked = mask * data # (batch_size,datapoints,outputs)
            return K.sum(masked, axis=-2) # (batch_size,outputs)            
        
        return self.shared_layer(keras.layers.Lambda, select_impl, {'name':'Select'})([self.data, self.position])
        
