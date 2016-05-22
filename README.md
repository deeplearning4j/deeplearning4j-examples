DL4J Release 0.4 Examples 
=========================
Repository of Deeplearning4J neural net examples:

- Convolutional Neural Nets
- Recurrent Neural Nets
- Deep-belief Neural Nets
- Restricted Boltzmann Machines
- Recursive Neural Nets
- TSNE
- Word2Vec & GloVe
- Anomaly Detection

---

## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.



## How to Run

if you have [maven](https://maven.apache.org/install.html) installed, run mvn on command line. Default maven goal, will compile and run all examples for you.


    mvn

if you want to run only one example, try

    mvn compile exec:java -Dexec.mainClass="org.deeplearning4j.examples.misc.csv.CSVExample"

