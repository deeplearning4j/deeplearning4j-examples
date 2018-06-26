# SameDiff - Custom Layers

The examples in this directory/package show how to implement layers using SameDiff.

What is SameDiff? An automatic differentiation package build on top of ND4J. It can be considered analogous to tools
such as TensorFlow and PyTorch.  

DL4J has supported custom layers for a long time - however, using SameDiff layers has some advantages:
1. Only the forward pass has to be defined: i.e., you don't need to manually work out gradient calculation
2. The layer definition is simpler: fewer methods to implement

## The Examples

PROPOSED:
1. Minimal feed forward layer example
2. Convolutional layer, with extra features (DL4J global configuration inheritance, etc)
3. Graph vertex
4. Lambda layer
5. Output layer
6. RNN layer
