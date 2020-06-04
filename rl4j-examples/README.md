## Eclipse Deeplearning4j: RL4J Examples

This project contains a set of examples that demonstrate how to use reinforcement learning with RL4J.

The examples in this project along with a short summary are listed below. This is also the recommended order to explore them in.

[Go back](../README.md) to the main repository page to explore other features/functionality of the **Eclipse Deeplearning4J** ecosystem. File an issue [here](https://github.com/eclipse/deeplearning4j-examples/issues) to request new features.

## Quickstart
There is no quickstart with RL4J at the moment. All examples require you to have some reinforcement learning experience to fully understand them.

## Advanced
* [Cartpole](./src/main/java/org/deeplearning4j/rl4j/examples/advanced/cartpole)
This example shows how to train the classic cartpole environment with DQN and A3C. When running the A3C example, make sure to change the model storage location to something convenient for you.

* [Arcade Learning Environment](./src/main/java/org/deeplearning4j/rl4j/examples/advanced/ale)
This example can only be run with the Atari Pong ROM `pong.bin`. It demonstrates how to use the Arcade Learning Environment (i.e. an Atari 2600 emulator that was made for reinforcement learning) with A3C and DQN.
