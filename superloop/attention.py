import keras
from keras import backend as K
import numpy as np

from .model import SuperLoopModel, ExtendWithZeros

class Attention(SuperLoopModel):
    """Implements a superloop extension that allows the system to focus on different parts of the input.
    It focuses on one data point only, but for non-integer locations, interpolates between the two data points.
    
    Usage:
        self.data is an input tensor of the shape ([batch_size],datapoints,outputs)
    """
    
    # TODO Provide the current location as an output. Limit the location to the length of the data.
    # TODO Initialize weights & bias controlling this model so that initially it would step forward.
    
    def __init__(self, config, **kwargs):
        """
            {
                'datapoints': number (length)
                'outputs': number (width)
            }
        """
        super().__init__(
            inputs=1, # move control
            outputs=config['outputs'],
            **kwargs
        )
        self.datapoints = config['datapoints']
        self.data = keras.layers.Input(shape=(config['datapoints'],config['outputs']), name="{}/InputData".format(self.name))
        self.position = None

    def init_dense(self, layer):
        # Here we initialise the input Dense layer so that we would mostly go forward
        # We know input and output are 1D
        layer.set_weights([
            np.random.uniform(low=0.0, high=0.1, size=(layer.get_input_shape_at(0)[1], layer.get_output_shape_at(0)[1])), # weights - low
            np.array([0.9]) # bias - high
        ])
    
    def _build_impl_impl(self, input):
        # Limit how much we can move
        move = self.shared_layer(keras.layers.Lambda, ((
            lambda x: K.sigmoid(x) * 2. - 1. # -1. .. 1.
        ),), {'name':'SigmoidCtlr'})(input)
        
        # Add the move control to the position
        if self._build_counter == 0:
            self.position = move
            self.skip_layer()
        else:
            self.position = self.shared_layer(keras.layers.Lambda, ((
                lambda x: x[0] + x[1]
            ),), {'name':'AddPosition'})([self.position, move])
            
        self.position = self.print_layer(self.position, "Attn_Position")

        def select_impl(x):
            data = x[0] # (batch_size,datapoints,outputs)
            position = x[1] # (batch_size,1)
            
            # batch_size = K.shape(position)[0]
            indices = K.arange(start=0, stop=self.datapoints, dtype=position.dtype) # (datapoints)
            # e.g. [0., 1., 2., 3., 4., 5., 6. 7.]
            
            indices = K.expand_dims(indices, axis=1) # (datapoints,1)
            position = K.expand_dims(position, axis=-2) # (batch_size,1,1)
            
            # Version without 0 grad regions
            diff = position - indices
            mask = 1. / (1. + diff * diff)
            
            masked = mask * data # (batch_size,datapoints,outputs)
            return K.sum(masked, axis=-2) # (batch_size,outputs)            
        
        out = self.shared_layer(keras.layers.Lambda, (select_impl,), {'name':'Select'})([self.data, self.position])
        out = self.print_layer(out, "Attn_Out")
        return out
        
