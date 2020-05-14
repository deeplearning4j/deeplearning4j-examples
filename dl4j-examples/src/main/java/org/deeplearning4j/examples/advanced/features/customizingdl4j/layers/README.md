
# Deeplearning4j: Custom Layer Example

This example adds a custom layer (i.e., one not defined in Deeplearning4j core).

The example itself contrived, but shows the basic configuration and testing process.

For the purposes of this example, we have implemented a a multi-layer perceptron (MLP)
layer very similar to DL4J's DenseLayer class. The difference here is that:

- An additional activation function is present in the configuration
- The standard activation function is applied to the first half of the layer outputs
- The other (new) activation function is applied to the second half of the outputs

## Writing Your Custom Layer

There are two components to adding a custom layer:

1. Adding the layer configuration class: extends org.deeplearning4j.nn.conf.layers.Layer
2. Adding the layer implementation class: implements org.deeplearning4j.nn.api.Layer

The configuration layer ((1) above) class handles the settings. It's the one you would
use when constructing a MultiLayerNetwork or ComputationGraph. You can add custom
settings here, and use them in your layer.

The implementation layer ((2) above) class has parameters, and handles network forward
pass, backpropagation, etc. It is created from the org.deeplearning4j.nn.conf.layers.Layer.instantiate(...)
method. In other words: the instantiate method is how we go from the configuration
to the implementation; MultiLayerNetwork or ComputationGraph will call this method
when initializing the network.

An example of these are CustomLayer (the configuration class) and CustomLayerImpl (the
implementation class). Both of these classes have extensive comments regarding
their methods.

You'll note that in Deeplearning4j there are two DenseLayer clases, two LSTM classes,
etc: the reason is because one is for the configuration, one is for the implementation.
We have not followed this "same name" pattern here to hopefully avoid confusion.

## Testing Your Custom Layer

Once you have added a custom layer, it is necessary to run some tests to ensure
it is correct.

These tests should at a minimum include the following:

1. Tests to ensure that the JSON configuration (to/from JSON) works correctly
   This is necessary for networks with your custom layer to function with both
   model serialization (saving) and Spark training.
2. Gradient checks to ensure that the implementation is correct

These tests are implemented in the CustomLayerExample class.
