Issue Summary: Though this example runs it seems to always predict one class. This example also does not follow a lot of the best practices we encourage users to employ like writing a custom iterator.
Action Items: Rework custom iterator probably to use a csv record recorder + embedding layer for one hot (or even datavec). Confirm training works and investigate why test seems to give only class no matter what.
Github Issue: https://github.com/eclipse/deeplearning4j-examples/issues/972
