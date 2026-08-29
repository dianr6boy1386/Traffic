# Traffic

## by Dyan Ahmadi

### overview

An AI that identifies which traffic sign appears in a photograph, built using TensorFlow to train a convolutional neural network on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Explanation  

This project implements a neural network that classifies traffic signs from images, using TensorFlow and the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The dataset contains thousands of images spanning 43 categories of road signs, such as speed limits, yield signs, stop signs, and various warning signs.

The program works in two main stages:

1. **Data loading and preprocessing** — Images are read from a directory structured by category (folders numbered 0–42), resized to a uniform 30x30 pixel resolution using OpenCV, and converted into NumPy arrays. The dataset is then split into training and testing sets using scikit-learn.

2. **Model building and training** — A convolutional neural network (CNN) is built with Keras, consisting of convolutional and max-pooling layers to extract visual features, followed by a dense hidden layer with dropout to reduce overfitting, and a final softmax output layer with 43 units (one per traffic sign category). The model is trained over 10 epochs and evaluated on the test set to measure its accuracy.

The trained model can be saved to an .keras file and then used by `predict.py` to make predictions on new images without retraining. 

This project explores core machine learning concepts including image preprocessing, convolutional neural networks, train/test splitting, and multi-class classification.

## Additional Details

The CS50 project guide only demanded us making the model but never using it, I however wanted to know if it works and if so, how good is it,
so I developed `predict.py` from scrach and sure enough tested it and implemented it to identify and show confidence alongside it.


## Project Structure

```
traffic/
├── gtsrb/
├── traffic.py
├── predict.py
├── requirements.txt
└── README.md
```

### Getting the Dataset (gtsrb)
 
1. Download the GTSRB dataset from [https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip)
2. Unzip the downloaded file.
3. Move the resulting `gtsrb` folder into the project directory, alongside `traffic.py`.


#### Files :

- **gtsrb/** — The dataset directory (downloaded separately). Contains 43 subfolders, numbered 0–42, each holding the traffic sign images for that category.

- **traffic.py** — Main program. Loads and preprocesses the traffic sign image dataset, builds and trains a convolutional neural network using TensorFlow/Keras, evaluates its performance on a test set, and optionally saves the trained model to a file.

- **predict.py** — Standalone script for using a trained model to classify a single traffic sign image. Loads a saved model file (e.g. `model.keras`), reads and resizes the given image the same way `traffic.py` does, then prints the predicted category number, its corresponding sign name (e.g. "14 (Stop)"), and the model's confidence in that prediction.

- **requirements.txt** — Lists the Python dependencies required to run the project (e.g. TensorFlow, OpenCV, scikit-learn, NumPy).

- **README.md** — This file, describing the project and its structure.


## Run
 
Install the required dependencies:
 
```
pip install -r requirements.txt
```
 
Run the program, providing the `gtsrb` data directory and, optionally, a filename to save the trained model to:
 
```
python traffic.py gtsrb model.keras
```

Then download any traffic sign image you want and run this for the already trained model `model.keras` to predict, replace `path\to\some_sign.jpg` with whatever sign image you downloaded : 

```
python predict.py model.keras path\to\some_sign.jpg 
```

## Credit 

Part of this project is a part of CS50's Introduction to Artificial Intelligence with Python.
