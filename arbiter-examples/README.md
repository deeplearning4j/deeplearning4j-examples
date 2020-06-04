## Eclipse Deeplearning4j: Arbiter Examples

This project contains a set of examples that demonstrate useage of the Arbiter library for hyperparameter tuning of Deeplearning4J models. More information on Arbiter can be found [here](https://deeplearning4j.konduit.ai/arbiter/overview).

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

The examples in this project and what they demonstrate are briefly described below. This is also the recommended order to explore them in.

### Quickstart
* [BasicHyperparameterOptimizationExample.java](./src/main/java/org/deeplearning4j/arbiterexamples/quickstart/BasicHyperparameterOptimizationExample.java)
Conduct random search on two network hyperparameters, and display the search progress in the Arbiter web-based UI.

### Advanced
* [BaseGeneticHyperparameterOptimizationExample.java](./src/main/java/org/deeplearning4j/arbiterexamples/advanced/genetic/BaseGeneticHyperparameterOptimizationExample.java)
Basic hyperparameter optimization example using the genetic candidate generator of Arbiter to conduct a search.
* [CustomGeneticHyperparameterOptimizationExample.java](./src/main/java/org/deeplearning4j/arbiterexamples/advanced/genetic/CustomGeneticHyperparameterOptimizationExample.java)
Change the default behavior of the genetic candidate generator.
