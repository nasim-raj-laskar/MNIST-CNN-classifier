# MNIST Handwritten Digit Classification
This project implements a machine learning model to classify handwritten digits from the MNIST dataset, using TensorFlow and Keras, directly within a Jupyter notebook.

## Project Overview
The notebook demonstrates:

- Data loading and preprocessing of the MNIST dataset.
- Building and training a Convolutional Neural Network (CNN) to classify digits.
- Evaluation of model performance.
- Visualization of sample predictions.

The project does not require any external datasets since the MNIST data is automatically downloaded when the code is run.

## Requirements
Make sure you have the following installed before running the notebook:

- Python 3.x
- Jupyter Notebook
- Required libraries:
   - tensorflow
   - numpy
   - pandas
   - matplotlib
     
You can install the dependencies using the following command:

```
pip install tensorflow numpy pandas matplotlib
```
## Usage

- Clone or download the repository:
```
git clone https://github.com/your-username/mnist-classification-notebook.git
```
- Open the Jupyter notebook:
```
jupyter notebook mnist_classification.ipynb
```
- Run all cells in the notebook to:
  - Download and preprocess the MNIST dataset.
  - Build and train the CNN model.
  - Evaluate the model’s performance on the test set.
  - Visualize the model’s predictions.
## Notebook Outline
- Introduction: Brief overview of the MNIST dataset and problem setup.
- Data Loading and Preprocessing:
  - Automatically downloads the dataset and prepares it for training.
- Model Building:
  - Constructs a CNN using Keras with layers like Convolution, Pooling, and Fully Connected layers.
- Model Training and Evaluation:
  - Trains the model on the training data and evaluates performance on the test set.
- Visualization:
  - Displays sample predictions and the model’s accuracy and loss over epochs.
    
## Results
- The model achieves high accuracy (~99%) on the MNIST test set.
- Sample images from the dataset are displayed with predicted and actual labels.
## License

This project is licensed under the MIT License .
