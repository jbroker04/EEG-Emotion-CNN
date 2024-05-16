**Emotion Classification with CNN (DEAP Dataset)**
==============================================

**Introduction**
---------------

This repository contains a Convolutional Neural Network (CNN) model for classifying emotions using Electroencephalography (EEG) signals from the DEAP dataset. The goal of this project is to develop a machine learning model that can accurately classify emotions into different categories (e.g. happy, sad, neutral, etc.) based on EEG signals.

**Dataset**
------------

The DEAP dataset is a widely used benchmark for emotion recognition tasks. It consists of EEG recordings from 32 participants who watched 40 music videos, and self-reported their emotional states (valence, arousal, liking, and dominance).

**Model Architecture**
---------------------

The CNN model is designed to extract features from the EEG signals and classify them into different emotional categories. The model architecture consists of:

* Conv1D layers for feature extraction
* BatchNormalization layers for normalizing the activations
* MaxPooling1D layers for downsampling
* Flatten layer for feature flattening
* Dense layers for classification
* Dropout layers for regularization

**Hyperparameter Tuning**
-------------------------

To optimize the model's performance, we have experimented with different combinations of hyperparameters, including:

* Loss functions: Binary Cross Entropy (BCE), Categorical Cross Entropy (CCE)
* Optimizers: Adam, Adamax, Nadam, RMSProp, Stochastic Gradient Descent (SGD), Adagrad, Adadelta

**Dependencies**
-------------------

* Python 3.8+
* Keras 2.4+
* Scikit-learn 0.24+
* NumPy 1.20+
* Pandas 1.3+

**Results**
----------

The best-performing model achieved an accuracy of 85% on the test set using the Adamax optimizer and Binary Cross Entropy loss function.

**Future Work**
--------------

* Experiment with different CNN architectures and hyperparameters
* Incorporate additional features from other modalities (e.g. facial expressions, speech)
* Investigate the use of transfer learning and domain adaptation techniques
* Use my dataset for training and testing with EEG and EOG signals


**Acknowledgments**
----------------

I would like to thank the DEAP dataset creators for making the dataset publicly available.
