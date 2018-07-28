# superloop

This project contains:

- A framework to build RNNs with a "superloop" in TensorFlow/Keras
- An implementation of a simple gated recurrent unit ("SGU") we use instead of LSTM or GRU cells
- An implementation of a register-based memory unit
- An implementation of an attention system that allows the network to move freely along the input data

A "superloop" is an extra loop that connects the output of the RNN to
its input in the next timestep via one or more external systems ("X").

![superloop overview](https://raw.githubusercontent.com/csirmaz/superloop/master/img/superloop1.png)

The external system can implement a special type of memory (derivable stack memory,
addressable memory, etc.), an attention system, or other enhancement.

## The superloop framework

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

**Builder** is a base class that allows building the same connections between layers
again and again, while reusing the layers themselves. This makes it possible to build
an unrolled model easily.

The **Model** class is used by client code. It implements the recipe for building the full model.

## "Simple Gated Recurrent Unit" (SGU)

We use SGU cells in the RNN part of the model.
The SGU is implemented by a linear combination between the hidden state (internal input) and the (external) input, controlled
by both inputs.

![sgu](https://raw.githubusercontent.com/csirmaz/superloop/master/img/sgu1.png)

Where W1, W2, W3 are trainable weights. The switch represents a simple linear combination:

![switch](https://raw.githubusercontent.com/csirmaz/superloop/master/img/sgu2.png)

## Attention System

## Register Memory