# Flexible-Neural-Network
A neural network model in TensorFlow that is very versatile, in that changing the number of layers or the sizes of each of those layers is done just by changing the value of a parameter. Also includes script to load data to the model from .csv files and to transform images into numpy arrays and .csv files compatible with the network.


As a sort of proof of concept I used this model to train a neural network to identify digits from images and was able to achieve 100% training and 97% testing accuracy without much fine-tuning (regularization would be the first thing to try). The script to train the model and predict using the model is still found in the frontend file
