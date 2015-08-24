DL4J Release 0.4 Examples 
=========================
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


| **Model Name**      | **Accuracy** | **F1** | **Status**   | **Training**  |
|---------------------|--------------|--------|--------------|---------------|
| CNNIris             | 0.70         | 0.76   | Tune         | full          | 
| CNNMnist            | 0.3          | 0.3    | Tune         | batch         |
| DBNIris             | 0.4          | 0.66   | Tune         | full          | 
| DBNMnistFull        |              |        | Rerun        | batch         |
| DBNMnistSingleLayer | 0.17         | 0.39   | Tune/Fix     | full          |
| MLPBackpropIris     | 0.55         | 0.70   | Tune         | batch         |
| RBMIris             |              | NA     | Tune         | full          |
| TSNEStandard        |              | NA     | Tune         | NA            |
| Word2VecRawText     |              | NA     |              | batch         |
    

* Accuracy and F1 depends on how many examples the model is trained on.
* Some networks need adjustments for seed to work (e.g. RNTN)
