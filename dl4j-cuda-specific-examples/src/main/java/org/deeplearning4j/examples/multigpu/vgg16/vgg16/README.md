Demonstrates use of the dl4j transfer learning API which allows users to 

	- construct a model based off an existing model
	- "freeze" certain layers
	- fine tune learning parameters

Explore preferably in the following order.
 1. Read TransferLearning.md
 2. Run "EditLastLayerOthersFrozen" to modify just the last layer in org.deeplearning4j.transferlearning.vgg16 and use it to fit the dataset. This is expected to run a while depending on your hardware.
 3. Build the same architecture in (2) but with featurized datasets
 	* Run "FeaturizedPreSave" which will featurize ~3000 images by passing them through org.deeplearning4j.transferlearning.vgg16. This is also expected to run a while depending on your hardware.
 	* Run "FitFromFeaturize" which will show you how to fit to presaved data so you can iterate quicker with different learning parameters. Fitting with the presaved dataset is very quick.
 4. "EditAtBottleneckOthersFrozen" for a look into how to rework model architecure by adding/removing vertices
 5. "FineTuneFromBlockFour" to show you how to continue training on a saved transfer learning model
