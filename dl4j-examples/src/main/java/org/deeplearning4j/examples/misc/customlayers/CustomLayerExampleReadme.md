
# Deeplearning4j: Custom Layer Example

This example adds a custom layer (i.e., one not defined in Deeplearning4j core).

The example itself contrived, but shows the basic configuration and testing process.
We'll be implementing a Multi-Layer Perceptron (MLP) layer very similar to
DL4J's DenseLayer class. The difference here is that:
- An additional activation function is present in the configuration
- The standard activation function is applied to half of the layer outputs
- The other (new) activation function is applied to the other half of the outputs

## Writing Your Custom Layer

There are two components to adding a custom layer:

1. Adding the layer configuration class: extends org.deeplearning4j.nn.conf.layers.Layer
2. Adding the layer implementation class: implements org.deeplearning4j.nn.api.Layer

The configuration layer ((1) above) class handles the settings. It's the one you would
use when constructing a MultiLayerNetwork or ComputationGraph

The implementation layer ((2) above) class has parameters, and handles network forward
pass, backpropagation, etc. It is created from the org.deeplearning4j.nn.conf.layers.Layer.instantiate(...)
method.

An example of these are CustomLayerConfig and CustomLayer. Both of these classes have
extensive comments regarding the methods and implementation.

## Testing Your Custom Layer

Once you have added a custom layer, you'll want to run some tests to ensure it's correct.
These tests should probably incl

1. Tests to ensure that the JSON configuration (to/from JSON) works correctly
2. Gradient checks to ensure that the implementation is correct

These tests are implemented in the CustomLayerTests




Author: Alex Black
