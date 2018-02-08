# superloop

A framework to build RNNs with a "superloop" in TensorFlow/Keras.

By superloop we mean an extra loop that connects the output of the RNN to
its input in the next time step via an external system ("X").

![superloop overview](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop1.png)

The external system can implement a special type of memory (derivable stack memory,
addressable memory, etc.), an attention system or other enhancement.

## Implementation

The framework implements the RNN and the superloop by unrolling both.
The diagram below illustrates one RNN unit (LSTM, GRU, etc.) and how it is
unrolled in time.

![unrolling one unit](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop2.png)

With two units (or, which is equivalent, layers of units) one after the other,
unrolling looks like this:

![unrolling two units](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop3.png)

It is easy to see that the internal loops of the RNN and the superloop need to be rolled
out at the same time:

![unrolling the superloop](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop4.png)

Representing the connecting layers in the diagram (fully connected, convolution, etc.) shows
the fundamental difference between the internal loops in the RNN and the superloop, as
the internal loops do not go through the connecting layers.

![connecting layers](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop5.png)

