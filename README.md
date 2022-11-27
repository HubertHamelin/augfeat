# AugFeat
AugFeat is a Python library that provides data augmentation in feature space. It is an implementation of the method
described in the paper 'Dataset Augmentation in Feature Space' written by Terrance DeVries and Graham W. Taylor in 2017.


## Installation
Use the package manager pip to install AugFeat.
```bash
pip install augfeat
```

## Usage
There are a few limitations for now if using this library. However using it is extremely simple.

### Prerequisites
The dataset on which you want to perform data augmentation operations has to respect the following:
- All classes are in a single main directory, its name has no importance.
- Each class has its own directory inside the main directory, named with the class name.
- All elements of a class are inside the corresponding class directory, names have no importance.
- Version 0.1 (first release) only handles numpy datasets.
- All elements of a single class must have the exact same shapes.

### How to use it
```python
from augfeat import Balancer, config

# It's up to you to choose which class in your dataset will be augmented, and how much.
dataset_path = './your/main/dataset/directory/path'
class_name = 'one_of_your_classes_name'
augmentation_target = 42

# Create Balancer instance.
balancer = Balancer(dataset_path, config.DataTypes.NUMPY)

# Call augment_class to create new data relevant to your original class.
balancer.augment_class(class_name, augmentation_target, config.AUTOENCODER_TRAINING_CONFIG_MEDIUM)
```

### Results example
Newly created elements will be saved on disk each time the augment_class method is called. After checking if the quality
is up to your expectations, you can choose  to merge the augmented elements with your original data, or keep them 
separated.

Here are examples of results obtained respectively for the MNIST dataset and the UJI Pen Characters dataset.


### Configuration details

## Roadmap
Extend formats handled by the library:
- 3D numpy matrices and higher dimensions.
- Images (png)
- Dataframes (within restrictive conditions)

## Contributing


## Authors and acknowledgment

## License


## Project Status
