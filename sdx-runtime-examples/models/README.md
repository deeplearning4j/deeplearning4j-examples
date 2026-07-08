# Shared example models

`mlp.sdz` — the fixture used by every `*-end-to-end` example:
`probs = softmax(relu(x·W1 + b1)·W2 + b2)` with deterministic
linspace-initialized weights, input `x: [batch, 4] float32`, output
`probs: [batch, 3] float32`.

Plan input contract (what `sdxGetNumInputs`/`sdxGetInputName` report —
external inputs cover constants, variables/weights, AND placeholders):
`w1 [4,8]`, `b1 [8]`, `w2 [8,3]`, `b2 [3]`, `x [batch,4]`.

Canonical verification vector (baked into each example):

```
x[2,4]     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
probs[2,3] = [0.44481823, 0.3220363, 0.23314552,
              0.4567148,  0.31961477, 0.22367041]
```

Regenerate with:

```bash
cd ../java-end-to-end
mvn -q compile exec:java -Dexec.mainClass=org.nd4j.examples.sdx.GenerateExampleModel
```
