import abc
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np


class Builder:
    """Builder objects can be called multiple times to build uniform parts of a model.
    
    A Builder object can be used to (re)build uniform, repeated parts of a model,
    optionally reusing the same layers. This way, Builder objects can build
    unrolled RNN models and set up weight sharing between the sme layers in
    different timesteps.
    
    This is an abstract class; a subclass must implement _build_impl that
    contains the recipe for building the model part.
    """


    def __init__(self, name, printvalues):
        """Constructor.

        Arguments:
        - name -- the name of the model
        - printvalues -- False or a number. If a number, the tensors on 
            which print_layer is called is dispayed while the model is
            evaluated. Max printvalues values are displayed from the tensor.
        """
        assert name # We need a name to be able to save/load layers
        self.name = name # Name of the model
        self.printvalues = printvalues # False or number TODO: should be bool
        
        self._shared_layers = {} # layer cache, keyed on _layer_counter TODO: key on name?
        self._layers_to_save = {} # layers weights of which canbe saved, keyed on their name
        self._layers_to_init = [] # array of layers to initialise
        self._layer_counter = 0 # shared layer index, reset at each build
        self._build_counter = 0

        
    def generate_name(self, name):
        """Returns a unique name for a shared layer.

        Arguments:
        - name -- the name of the layer
        """
        return "{}/{}.{}".format(self.name, self._layer_counter, name)

        
    def print_layer(self, layer, label):
        """Optionally add a step to the network that prints the values from the tensor.

        Arguments:
        - layer - the input tensor
        - label - a label to display in the output and in the layer name
        """
        if self.printvalues:
            # The message is machine readable
            name = "{}/{}.{}.{}".format(self.name, self._layer_counter, '{}'+label, self._build_counter)
            message = "<<{}>>".format(name.format(''))
            # We generate a new layer in each timestep
            # As a hack, we pass self.printvalues as the maximum size
            return keras.layers.Lambda((lambda x: tf.Print(x, [x], message=message, first_n=-1, summarize=self.printvalues)), name=name.format('Print_'))(layer)
        else:
            return layer

        
    def shared_layer(self, build_function, args, kwargs, skip_cache=False, save=False, initfn=None, initfnkwargs=None):
        """Either create a layer shared between builds, or return one from the cache.
        
        Arguments:
        - build_function -- function to call if the layer is to be built
        - args -- positional arguments for build_function
        - kwargs -- keyword arguments for build_function. Must contain 'name'
        - skip_cache -- optional; bool; if True, don't store the layer in the cache, but still increment _layer_counter
        - save -- optional; bool; if True, weights from the layer can be saved and loaded
        - initfn -- optional; None or callable; if callable, call this on the layer once its size is known
        - initfnkwargs -- optional; extra keyword arguments passed to initfn when it is called
        """

        assert 'name' in kwargs # We need a name to be able to save/load layers
        fullname = self.generate_name(kwargs['name'])

        key = self._layer_counter

        if skip_cache or key not in self._shared_layers:
            # Build the layer
            kwargs['name'] = fullname # Automatically enhance the name
            layer = build_function(*args, **kwargs)
            if initfn:
                self._layers_to_init.append({'layer':layer, 'initfn':initfn, 'initfnkwargs':initfnkwargs}) 
            if not skip_cache:
                self._shared_layers[key] = layer
        else:
            layer = self._shared_layers[key]

        if save:
            self._layers_to_save[fullname] = layer
        
        self._layer_counter += 1
        return layer


    def skip_layer(self, count=1):
        """Increment the layer counter by count.
        
        If a shared layer is used in an optional branch of the recipe, use skip_layer in
        the other branch to ensure that the layer index (counter) is consisent regardless
        of the branch taken.
        """
        self._layer_counter += count


    def build(self, *args, **kwargs):
        """Build the model once.
        
        All arguments are forwarded to _build_impl.
        """
        self._layer_counter = 0
        r = self._build_impl(*args, **kwargs)
    
        # Call the init functions    
        if self._build_counter == 0:
            for initlayer in self._layers_to_init:
                if initlayer['initfnkwargs']:
                    initlayer['initfn'](initlayer['layer'], **initlayer['initfnkwargs'])
                else:
                    initlayer['initfn'](initlayer['layer'])
        
        self._build_counter += 1
        return r


    @abc.abstractmethod
    def _build_impl(self):
        """Actually build the model once.
        
        This is an abstract method. Implement this in a subclass and use shared_layer etc.
        to specify the recipe of building the model.
        """

    def _save_local_weights(self, h5file):
        """Internal method. Save the weights of savable layers into the given h5py file object.
        
        Arguments:
        - h5file -- h5py file object
        """
        for name, layer in self._layers_to_save.items():
            self._save_layer_weights(layer, name, h5file)
            

    def _load_local_weights(self, h5file):
        """Internal method. Load and overwite the weights of savable layers.
        
        Arguments:
        - h5file -- h5py file object
        """
        for name, layer in self._layers_to_save.items():
            self._load_layer_weights(layer, name, h5file)


    def _save_layer_weights(self, layer, name, h5file):
        """Internal method. Save the weights of one layer into a h5py file object.
        
        Arguments:
        - layer -- a layer
        - name -- name to save the data under
        - h5file -- h5py file object
        """
        # How to save binary data in h5py: http://docs.h5py.org/en/latest/strings.html
        # How to serialize objects: https://docs.python.org/3.3/library/pickle.html
        weights = layer.get_weights()
        h5file.create_dataset("{}/length".format(name), data=np.array([len(weights)]))
        for idx, array in enumerate(weights):
            h5file.create_dataset("{}/{}".format(name, idx), data=array)

        
    def _load_layer_weights(self, layer, name, h5file):
        """Internal method. Load the weights of one layer from a h5py file object.
        
        Arguments:
        - layer -- a layer
        - name -- name to save the data under
        - h5file -- h5py file object
        """        
        group = h5file[name]
        length = group['length'][0]
        weights = [group["{}".format(idx)] for idx in range(length)]
        layer.set_weights(weights)    


class SuperLoopModel(Builder):
    """A SuperLoopModel object is used to build an external system connected to the RNN via the superloop.
    
    This is a subclass of Builder, but still and abstract class. Subclasses should override
    _build_impl_impl to provide the recipe for building the external system.
    """
    
    def __init__(self, inputs, outputs, **kwargs):
        """Constructor.
        
        Arguments:
        - inputs -- the number of inputs this system has.
        - outputs -- the number of outputs this system has.
        - any extra keyword arguments are forwarded to the superclass (Builder) constructor
        """
        self.outputs = outputs # Number of outputs (1D tensor). We need to know this in advance.
        self.inputs = inputs # Number of inputs (1D tensor)

        super().__init__(**kwargs)

        self.out = None # Always contains the output of the latest build

 
    def init_dense(self, layer):
        """Initialise the dense layer providing the input.
        
        Implement any necessary initialization in the subclass.
        
        Arguments:
        - layer -- the dense layer.
        """
        pass


    def _build_impl(self, input):
        """Internal method. Implements building the external system.
        
        Arguments:
        - input -- the input tensor.
        """
        self.out = self._build_impl_impl(input)
        return self.out # output


    @abc.abstractmethod
    def _build_impl_impl(self, input):
        """Actually build the external system once.
        
        This is an abstract method. Implement this in a subclass and use shared_layer etc.
        to specify the recipe of building the external system.
        """


def CropLayer(start, end, step=1, name=None):
    """Create a layer that crops out a range from a 1D (+ batch dimension) input tensor.
    
    Arguments:
    - start -- start index
    - end -- end index
    - step -- optional; step
    - name -- optional; name of the layer
    """
    # See https://github.com/keras-team/keras/issues/890
    return keras.layers.Lambda(lambda x: x[:, start:end:step], output_shape=(int((end-start)/step),), name=name)
    # Could use K.slice but unsure how to manage batch dimension


def ExtendWithZeros(size_added, name=None):
    """Create a layer that adds size_added 0's to a 1D (+ batch dimension) tensor in all batches.
    
    Arguments:
    - size_added -- the number of 0's to add
    - name -- optional; name of the layer
    """
    # This automatically infers the batch size from the shape of the input
    # See https://stackoverflow.com/questions/46465813/creating-constant-value-in-keras
    def func(x):
        batch_size = K.shape(x)[0]
        zeros = K.zeros((1, size_added))
        tiledzeros = K.tile(zeros, (batch_size, 1))
        return K.concatenate([x, tiledzeros])
    return keras.layers.Lambda(func, name=name)

