# Chest x-ray diagnosis with deep-learning CNN over CheXpert data.
cs230 stanford course project codes and results.
All models are coded in Python using Keras.

In this study, predictive multi-class models are trained for chest x-ray diagnosis
of 14 observations using different deep learning architectures and a large dataset
of chest x-ray images called CheXpert. First, three different deep learning models
including VGG-16, ResNet-50, and DenseNet-121 are trained on an Amazon AWS
EC2 GPU instance. For DenseNet-121, both transfer learning and full training
are applied. While a good accuracy is achieved on testset data, the F1 scores on a
few observations were low. This was an indication of model robustness issue for a
few class predictions. Further analysis of the data indicates an unbalance between
available data for those observations with low F1 scores. An up-sampling approach
is applied to balance the training data. This results in a significant improvement
in both accuracy and F1 scores over the testset data. Finally, a gradient weighted
Class Activation Map is applied to localize the highest probability observation for
a given x-ray image input.
 
![Model](/DL-CNN-CheXpert-data/images/Model_Schematic1.JPG)

