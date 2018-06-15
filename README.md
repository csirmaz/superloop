# superloop

This project contains:

- A framework to build RNNs with a "superloop" in TensorFlow/Keras.
- An implementation of a simple gated recurrent unit ("SGU") we use instead of LSTM or GRU cells.
- An implementation of a register-based memory unit.

By "superloop" we mean an extra loop that connects the output of the RNN to
its input in the next time step via an external system ("X").

![superloop overview](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop1.png)

The external system can implement a special type of memory (derivable stack memory,
addressable memory, etc.), an attention system, or other enhancement.

## The superloop

The framework implements the RNN and the superloop by unrolling both.
The diagram below illustrates one RNN unit (LSTM, GRU, etc.) and how it is
unrolled in time.

![unrolling one unit](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop2.png)

With two units (or, which is equivalent, layers of units) one after the other,
unrolling looks like this:

![unrolling two units](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop3.png)

It is easy to see that the internal loops of the RNN and the superloop need to be unrolled
at the same time:

![unrolling the superloop](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop4.png)


## Simple Gated Recurrent Unit (SGU)

The SGU is implemented by a linear combination between the hidden state (internal input) and the (external) input, controlled
by both inputs.

![sgu](https://raw.githubusercontent.com/csirmaz/superloop/master/img/sgu1.png)

Where W1, W2, W3 are trainable weights. The switch represents a simple linear combination:

![switch](https://raw.githubusercontent.com/csirmaz/superloop/master/img/sgu2.png)
