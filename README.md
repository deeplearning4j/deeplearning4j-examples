DL4J Release 0.4 Examples 
=========================

[![Join the chat at https://gitter.im/deeplearning4j/dl4j-0.4-examples](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/dl4j-0.4-examples?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
Repository of Deeplearning4J neural net examples:

- Convolutional Neural Nets
- Deep-belief Neural Nets
- Glove Example
- Restricted Boltzmann Machines
- Recurrent Neural Nets
- Recursive Neural Nets
- TSNE
- Word2Vec

---

## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.

## Performance

| **Model Name**      | **Accuracy** | **F1** | **Training**  |
|---------------------|--------------|--------|---------------|
| CNNIris             | 0.7          | 0.75   | batch         | 
| CNNMnist            | 0.73         | 0.74   | batch         |
| DBNIris             | 0.33         | 0.5    | full          | 
| DBNMnistFull        | 0.10         | 0.35   | batch         |
| DBNMnistSingleLayer | 0.41         | 0.61   | full          |
| MLPBackpropIris     | 0.55         | 0.70   | full          |
| RBMIris             |              | NA     | full          |
| TSNEStandard        |              | NA     | NA            |
| Word2VecRawText     |              | NA     | batch         |
    

* Accuracy and F1 depends on how many examples the model is trained on.
