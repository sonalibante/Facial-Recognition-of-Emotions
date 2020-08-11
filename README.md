# Software Engineering Project
# Facial Recognition of Emotions
This is a project for CS5394 Summer 2020 class

## Abstract

Facial Emotions are one of the most important aspects of human communication. Facial expression recognition (FER) provides machines a way of sensing emotions that can be considered one of the mostly used Artificial Intelligence and pattern analysis applications. One of the relatively new and promising trends in using facial expressions to classify learners’ emotions is the development and use of software programs that automate the process of coding using advanced machine learning technologies.

## Project Description

- There are many different types of emotions that have an influence on how we live and interact with others. Among those, the 6 important emotions are Happiness, Sadness, Fear, Disgust, Anger, Surprise. 

- The aim of the project is to build efficient face recognition with keras by Deep Learning. We train and design a deep Convolutional Neural Networks (CNN) using tf.keras which is TensorFlow’s high level API for building and training deep learning models. Keras is running on top of tensorflow, using Keras in deep learning allows for easy and fast prototyping as well as running seamlessly on CPU and GPU. This framework is written in Python code which is easy to debug and allows ease for extensibility. 

- CNN is primarily used here because, it is one of the popular Deep Artificial Neural Networks which are majorly used in image recognition, image clustering and classification, object detection. CNN uses relatively less preprocessing when compared with the other algorithms of image processing. CNN consists of different layers. There are input layer and output layer. Between these layers there are some multiple hidden layers. There is no limitation for hidden layers present in the network. Input layer takes the input and train specifically and gives output from the output layer. With the help of CNN, we can use the large amount of data with more effectively and accurately.

- Face detection part is done using OpenCV(Open Source Computer Vision) where the face detection classifiers automatically detects faces and draws boundary boxes around them to analyze. Here OpenCV is opted because, it mainly used for Image Processing, like read or write images, face detection and its features, text recognition in images, detection of shapes, modifying the image quality and colors, for developing augmented reality apps. 
Once the development of visual expression recognition model is done in keras, we train the network using the dataset from Kaggle – facial expression dataset. Once the model is trained and saved, then it is deployed with web interface using Flask to make predictions for inference and make it functional for further developments. A video can be fed, where it detects emotions in that. Webcam can also be accessed where it detects Facial emotions through live faces.

- All these can be implemented in Jupyter Notebook through Anaconda Navigator or through Google Colab. 

## REQUIREMENTS

- Use Anaconda Navigator https://www.anaconda.com/products/individual. Prefer Graphical Installer while download.
- To train the model, use Google Colab. https://colab.research.google.com/ if there is no powerful GPU available in your personal computer.

## INSTALLATION

* Anaconda Navigator - Then launch Jupyter Notebook to code
* Create your own virtual environment. Follow the below link for set up
``` https://docs.anaconda.com/anaconda/navigator/getting-started/ ```
* Install the libraries.
    - Python version 3.6 and above
    - Keras
    - Tensorflow 2.0.0 and above
    - Numpy
    - Pandas
    - Matplotlib
    - OpenCV
    - Flask
    
  ## TO RUN THE PROJECT
  
  - In the Anaconda Navigator, create a virtual environment with Keras, Tensorflow, OpenCV and Flask installed
  - Then in Command Prompt/Terminal type `pip install pipenv`
  - `pipenv install`
  - `python main.py`
  - Then go to any web browser like Google Chrome, type localhost:5000, to see the output.
  - Change camera.py file to webcam or video feed to see the respective output.
    




