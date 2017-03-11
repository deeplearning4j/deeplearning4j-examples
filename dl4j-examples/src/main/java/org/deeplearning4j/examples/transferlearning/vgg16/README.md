Demonstrates use of the dl4j transfer learning API which allows users to 
	- construct a model based off an existing model
	- "freeze" certain layers
	- fine tune learning parameters

Explore preferably in the following order.
 1. Read BLAH_BLAH_website
 2. Run "EditLastLayerOthersFrozen" to modify just the last layer in vgg16 and use it to fit the dataset. Expect this to run a "while"
 3. Build the same architecture in (2) but with featurized datasets
 	a. Run "FeaturizedPreSave" which will featurize ~3000 images by passing them through vgg16. Expect this to run "a while"
 	b. Run "FitFromFeaturize" which will show you how to fit to presaved data so you can iterate quicker with different learning parameters
 4. "EditAtBottleneckOthersFrozen" for a look into how to rework model architecure by adding/removing vertices
 5. "FineTuneFromBlockFour" to show you how to continue training on a saved transfer learning model
