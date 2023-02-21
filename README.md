# Assessing the Accuracy of Neural Network Models for Predicting Fracture Length, Width and Height

## Overview
This repository contains a Jupyter notebook implementing a neural network model that predicts the fracture dimensions in a hydraulic fracturing operation based on four input parameters. The synthetic data used for the model was generated using the Perkins-Kern-Nordgren (PKN) model.

The notebook uses the 'numpy', 'sklearn', 'tensorflow.keras', and 'matplotlib' libraries to generate and preprocess the data, define and train the neural network model, and visualize the results.

## Hydraulic Fracturing and Artificial Neural Networks
Hydraulic fracturing, commonly known as fracking, is a process of injecting a mixture of water, sand, and chemicals into the underground rock formations to extract natural gas or oil. It is a complex process that requires a thorough understanding of the subsurface geology, fluid mechanics, and chemical reactions.

In recent years, the use of machine learning techniques, particularly artificial neural networks (ANN), has gained popularity in the field of hydraulic fracturing for predicting various parameters such as fracture length, width, height, and proppant distribution.

This repository contains a Jupyter Notebook that demonstrates how to use an ANN to predict the fracture dimensions in hydraulic fracturing based on the Perkins-Kern-Nordgren (PKN) model. The steps involved in this notebook are as follows:

**1- Generate Synthetic Data**: We first generate synthetic data using the PKN model, which considers various input parameters such as injection rate, fluid viscosity, proppant concentration, and rock properties to predict the fracture dimensions.

**2- Data Preprocessing**: We then preprocess the generated data by normalizing the input and output variables using the StandardScaler function from sklearn.preprocessing.

**3- Cross-Validation**: We define a 5-fold cross-validation using KFold function from sklearn.model_selection.

**4- Define Neural Network Architecture**: We define the neural network architecture using the Sequential model from tensorflow.keras.models. Our model consists of two hidden layers, each with 16 neurons and a relu activation function, followed by an output layer with 3 neurons.

**5- Compile and Train the Model**: We compile the model using the adam optimizer and mean squared error (MSE) as the loss function. We then train the model using the cross-validation technique defined earlier.

**6- Evaluate the Model**: We evaluate the model using the test dataset and calculate the MSE, RMSE, and R-squared values. Additionally, we plot the predicted vs actual values and distribution of residuals for each output dimension.

The results of this model on the test dataset are as follows:

Test loss (MSE): 0.0042
Test RMSE: 0.0651
Test R^2: 0.9950

These results indicate that the ANN model is highly accurate in predicting the fracture dimensions and can be used as a valuable tool in the field of hydraulic fracturing.


## Usage
To run the notebook, simply open it in Jupyter Notebook or JupyterLab and run each cell sequentially. The notebook contains detailed explanations of each step in the process.
