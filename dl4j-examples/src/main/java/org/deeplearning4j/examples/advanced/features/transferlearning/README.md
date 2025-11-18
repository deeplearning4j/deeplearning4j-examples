# Transfer Learning Examples – DeepLearning4J

This folder demonstrates multiple ways to use the DL4J Transfer Learning API.  
Transfer learning allows you to take an existing pretrained model (such as VGG16) and adapt it to a new dataset by:

- Freezing certain layers  
- Modifying the architecture  
- Replacing the final classifier  
- Fine-tuning deeper layers  
- Speeding up training through featurized datasets  

Official documentation:  
🔗 https://deeplearning4j.konduit.ai/tuning-and-training/transfer-learning

---

# 📁 Folder Overview

The transfer learning examples are grouped into three major strategies, depending on how much of the pretrained model you want to modify or retrain.

---

## 1️⃣ **Edit Last Layer Only** (`editlastlayer/`)

### ➤ `EditLastLayerOthersFrozen.java`
Replaces only the final output layer of VGG16 and freezes all previous layers.

**Use Case:**
- New dataset is small  
- New dataset is similar (e.g., flower classification)  
- Extremely fast training  
- Very little overfitting risk  

---

### ⚡ Presaving Featurized Data (`editlastlayer/presave/`)
This subfolder contains examples for speeding up transfer learning.

#### ➤ `FeaturizedPreSave.java`
Runs the dataset once through the frozen layers and **saves the output features** to disk.

#### ➤ `FitFromFeaturized.java`
Uses the presaved features to train the classifier **without repeating the forward pass**.

**Why this helps:**  
Huge speed-up when training for many epochs or trying multiple learning rates.

---

## 2️⃣ **Edit from Bottleneck** (`editfrombottleneck/`)

### ➤ `EditAtBottleneckOthersFrozen.java`
Modifies the model at the **bottleneck layer** (middle of the network).  
Allows adding/removing vertices, changing layer shapes, or inserting new layers.

**Use Case:**
- New dataset is somewhat different from ImageNet  
- Need more flexibility in how features are combined  
- Want deeper customization but still freeze earlier layers  

This is a more advanced example showing how to modify the computation graph.

---

## 3️⃣ **Fine-Tune Some Layers Only** (`finetuneonly/`)

### ➤ `FineTuneFromBlockFour.java`
Loads a previously saved model and fine-tunes only the last few blocks while keeping earlier layers frozen.

**Use Case:**
- Dataset is moderately different  
- Need the network to adapt more deeply  
- Fine-tuning block 4+ is common with VGG/ResNet architectures  
- Allows training more parameters without overfitting as much as full fine-tuning  

---

## 4️⃣ **Dataset Iterators** (`iterators/`)

These classes prepare the flower dataset for transfer learning.

### ➤ `FlowerDataSetIterator.java`
Loads images from disk and prepares them for VGG16-style input.

### ➤ `FlowerDataSetIteratorFeaturized.java`
Loads presaved (featurized) data for fast training.

---

# 🚀 How to Run Any Transfer Learning Example

Use:

mvn -q exec:java -Dexec.mainClass="org.deeplearning4j.examples.advanced.features.transferlearning.<folder>.<ClassName>"


Examples:



mvn -q exec:java -Dexec.mainClass="org.deeplearning4j.examples.advanced.features.transferlearning.editlastlayer.EditLastLayerOthersFrozen"

mvn -q exec:java -Dexec.mainClass="org.deeplearning4j.examples.advanced.features.transferlearning.editfrombottleneck.EditAtBottleneckOthersFrozen"

mvn -q exec:java -Dexec.mainClass="org.deeplearning4j.examples.advanced.features.transferlearning.finetuneonly.FineTuneFromBlockFour"


---

# 🎯 Summary of Strategies

| Strategy | Layers Trained | Best For |
|---------|----------------|----------|
| **Edit Last Layer Only** | Only final Dense layer | Small, similar datasets; fastest training |
| **Edit from Bottleneck** | Middle layers | More control; moderate dataset differences |
| **Fine-Tune Block Four** | Last few blocks | Moderate differences; more expressive training |
| **Full Featurization** | Only classifier | Fast experiments; repeated runs |

---

# 🙌 Why This README Matters

The original README was brief and lacked conceptual explanations.  
This improved version:

- Documents each example clearly  
- Explains when to use each transfer learning strategy  
- Provides run commands  
- Helps new users understand DL4J transfer learning best practices  
- Improves readability and usefulness of advanced examples  

This makes transfer learning in DL4J easier to understand and use.