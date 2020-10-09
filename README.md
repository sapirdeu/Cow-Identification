# Cow-Identification

### The problem:
Given a cow, identify the closest looking cow from the list of cows in the dataset.

### The main idea:
- In the `main.py` file, we first process the training set. 
- Next, call the `kerasModel()` function located in the `train.py` file, to construct a neural network model that identifies the cows.
- After the training phase, we use the test set and check accuracy.
- We got an accuracy percentage higher than 90%.

## Installation:
Clone this repository to your local machine and run:
```
cd Cow-Identification
pip install Keras
pip3 install --upgrade tensorflow-gpu --user
python main.py
```

---------------------------

### References:
Dataset: https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17
