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
To access CheXpert chest x-ray data chech: https://stanfordmlgroup.github.io/competitions/chexpert/
Data includes about 200,000 images labeled for 14 observations.

A schematic view of the model is shown below.
 
![Model](https://github.com/hborhan/DL-CNN-CheXpert-data/blob/master/images/Model_Schematic1.JPG)

Summary of the uploaded codes. All jupyter notebooks have comments to follow.
- [Model_CheXpert_DenseNet121_R1_TransferLearning.ipynb] Training DenseNet-121 model with transfer learning approach.
- [Model_CheXpert_DenseNet121_R2_FullTraining.ipynb] Training DenseNet-121 model with training all parameters (~8 million)
- [Model_CheXpert_DenseNet121_R1_TransferLearning.ipynb] Training DenseNet-121 after balancing data
- [Gradient Weighted Class Activation Map Visualize - Sample 1.ipynb] Generate localized image with weighted gradient Class Activation Map (there are 2 samples)
- [CS230_Report_hborhan.pdf] The report file

I found the following references very udeful:
[1] https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24
    Multi-label image classification Tutorial with Keras ImageDataGenerator - very helpful to learn how to read images from csv files with location address. Just be careful with backslash in the file address if you are using windows (add .r before address in the code).
[2] https://towardsdatascience.com/setting-up-and-using-jupyter-notebooks-on-aws-61a9648db6c5
    Setup and use Jupyter (IPython) Notebooks on AWS. I recommend to use chrome in the last step to open the notebook.
    

