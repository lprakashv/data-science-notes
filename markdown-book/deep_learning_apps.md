# Neural Networks

## Keras

A high level framework for building neural networks with only a few lines of code

### Backends
- Keras is a front-end layer.
- It depends on a backen-end library.
- It supports multiple back ends.

Keras uses either **Tensorflow** or **Theano** behind the scenes to do all its calculations.

#### Theano
- Created at MILA (Montreal Institute for Learning Algorithms) at the University of Montreal
- Works well with Python
- GPU acceleration

#### Tensorflow
- Created at Google
- GPU acceleration
- Distributed computing
- **When to use?**
    - Researching new types of machine learning models
    - Building a large-scale system to support many users
    - If processing and memory efficiency is more important than time saved while coding

### Keras
- Industry best practices are built in.
- The default settings in Keras are designed to give you good results in most cases.
- **When to use?**
    - Education and experimentation
    - Prototyping
    - Production system that don't have highly specialized requirements

### Keras + Tensorflow
- High level
- Fast experimentation
- Write less code

## Code sneak-peek

### Creating a Model

```python
model = keras.models.Sequential()
model.add(keras.layers.Dense())
## .. add mode layers ..
model.compile(loss='mean_squared_error', optimizer='adam')
```

### Training Phase

```python
mode.lfit(training_data, expected_output)
```

### Testing Phase

```python
error_rate = model.evaluate(testing_data, expected_output)

mode.save("training_model.h5")
```

### Evaluation Phase
```python
model = keras.models.load_model('trained_model.h5')

predictions = mode.predict(new_data)
```

## Keras Sequential Model API
- The easiest way to build neural networks in Keras
- Create an empty sequential model object and then add layers to it in sequence

```python
model = keras.models.Sequential()
model.add(Dense(32, input_dim=9))
model.add(Dense(128))
model.add(Dense(1))
```

### Customizing Layers

```python
model.add(Dense(number_of_neurons, activation='relu'))
```

### Layer Options

- Layer activation function
- Initializer function for node weights
- Regularization function for node weights

**But the default settings are a good start!**

### Other Types of Layers Supported

#### Convolutional layers
Example: `keras.layers.convolutional.Conv2D()`

#### Recurrent layers
Example: `keras.layers.recurrent.LSTM()`


```python
import numpy as np

a = np.array([5, 6, 3, 1])
b = [9, 10, 7]

a.sum()
```




    15




```python

```
