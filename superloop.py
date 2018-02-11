import keras

class RecurrentUnit:
    
    def __init__(self, loop_size, name=None):
        self.name = name
        self.loop_size = loop_size
        self._shared_layers = {}
        self._layer_counter = 0
    
    def shared_layer(self, build_function, args, kwargs):
        """Either create a layer shared between all units, or return one from the cache"""
        key = self._layer_counter
        if key not in self._shared_layers:
            if self.name and ('name' in kwargs):
                kwargs['name'] = self.name + '.' + self._layer_counter + '.' + kwargs['name']
            self._shared_layers[key] = build_function(*args, **kwargs)
        
        self._layer_counter += 1
        return self._shared_layers[key]
    
    def get_unit(self, external_input, internal_input):
        """Build a unit using two inputs and return two outputs"""
        self._layer_counter = 0
        self.build(external_input, internal_input)
    
    def build(self, external_input, internal_input): # Abstract method
        """Implements building the unit itself using shared_layer()"""
        
        # Vanilla RNN
        
        x = self.shared_layer(keras.layers.concatenate, ([internal_input, external_input]), {'name': 'concat'})
        x = self.shared_layer(keras.layers.Dense, (), {'units': self.loop_size, 'name': 'dense'})(x)
        x = self.shared_layer(keras.layers.Activation, ('relu'), {'name': 'relu'})(x)

        # Split layer: https://github.com/keras-team/keras/issues/890
        return (x, x)


class Model:
    
    def __init__(self, superloop_models, superloop_sizes):



# keras.layers.Input(shape=(10,), name="name1")
# x = keras.layers.Dense(units=11, name="name2")(x)
# x = keras.layers.concatenate([x, y], name="name3")
# keras.layers.TimeDistributed(keras.layers.Dense(12), name="name4")

timesteps = 3

for timestep in range(timesteps):

    x = keras.layers.concatenate([input_at_timestep, superloop_out], name="concat{}.{}".format(timestep,0))
    
    for layernum in range(layersnum):
        x = connectorLayers[layernum](x)
        x, recurrentState[layernum] = recurrentLayers[layernum](x, recurrentState[layernum])


