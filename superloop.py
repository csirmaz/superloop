import keras

class Builder:
    """Implements building and rebuilding models from cached layers"""

    def __init__(self, name=None):
        self.name = name
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

    def build(self, *args, **kwargs):
        self._layer_counter = 0
        self._build_impl(*args, **kwargs)


class RecurrentUnit(Builder):
    
    def __init__(self, units, **kwargs):
        self.units = units
        super().__init__(**kwargs)
       
    def _build_impl(self, external_input, internal_input): # Abstract method
        """Implements building the unit itself using shared_layer()"""
        
        # Vanilla RNN
        
        if internal_input is not None:
            x = self.shared_layer(keras.layers.concatenate, ([external_input, internal_input]), {'name': 'concat'})
        # TODO not arbitrary connections!!
        x = self.shared_layer(keras.layers.Dense, (,), {'units': self.units, 'name': 'dense'})(x)
        x = self.shared_layer(keras.layers.Activation, ('relu'), {'name': 'relu'})(x)

        # Split layer: https://github.com/keras-team/keras/issues/890
        return (x, x) # (external_output, internal_output)


class Model(Builder):
    
    def __init__(self, superloop_models, superloop_sizes, **kwargs):
        super().__init__(**kwargs)

    def initialise(self): # Abstract method
        self.recurrent_layers_num = 5
        self.recurrent_units_num = 3
        self.recurrent_layers = [RecurrentUnit(units=self.recurrent_units_num, name="Recur{}".format(layernix)) 
            for layerix in range(self.recurrent_layers_num)]
        self.recurrent_internal = [None for layerix in range(self.recurrent_layers_num)]

    def _build_impl(self, input, superloop_input): # Abstract method
        """Implements building the mode"""

        x = self.shared_layer(keras.layers.concatenate, ([input, superloop_input]), {'name': 'main_concat'})
        for layernix in range(self.recurrent_layers_num):
            x = self.shared_layer(keras.layers.Dense, (,), {'units': self.recurrent_units_num, 'name': "main_dense".format(layerix) })(x)
            x, self.recurrent_internal[layerix] = self.recurrent_layers[layerix].build(x, self.recurrent_internal[layerix])



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


