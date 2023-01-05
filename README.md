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
- All elements of a single class must have the exact same shapes, which means no missing data.

### How to use it
```python
from augfeat.balancer import Balancer
from augfeat.custom_types import DataTypes
from augfeat import config

# It's up to you to choose which class in your dataset will be augmented, and how much.
dataset_path = './your/main/dataset/directory/path'
target_path = './your/target/directory/path/for/augmented/data'
class_name = 'one_of_your_classes_name'
augmentation_target = 42

# Create Balancer instance.
balancer = Balancer(dataset_path, target_path, DataTypes.NUMPY)

# Call augment_class to create new data relevant to your original class.
balancer.augment_class(class_name, augmentation_target, config.AUTOENCODER_TRAINING_CONFIG_MEDIUM)
```

### Results example
Newly created elements will be saved on disk each time the augment_class method is called. After checking if the quality
is up to your expectations, you can choose  to merge the augmented elements with your original data, or keep them 
separated.

Here are examples of results obtained respectively for the MNIST dataset and the UJI Pen Characters dataset.


### Configuration details
You can either use one of the default configurations from the config.py file, or your own configuration. 
See the example below:
```python
autoencoder_config = {
    'latent_dim': 128,  # higher dim => can take more complex data but harder to train.
    'dropout_rate': 0.2,
    'epochs': 200,  # depends on your data.
    'batch_size': 128,  # higher => faster but careful with your available memory.
    'learning_rate': 1e-3
}
```

## Roadmap
Extend formats handled by the library:
- 3D numpy matrices and higher dimensions.
- Images (png)
- Dataframes (within restrictive conditions)

Enhance augmented data quality:
- Post processing enhancement of generated elements
- Add other types of autoencoders to better suit various data (CNN for images ...)
- Learning rate scheduler for the autoencoder class 
- Reverse the order of input sequences as suggested in the original paper

Extend possible execution modes:
- Add distinct CPU and GPU execution mode options

Make user experience easier:
- Add a more extensive and comprehensive configuration for both the balancer and the autoencoder

## Contributing
Just create a new branch to experiment your functionalities, and make a merge request.

## Authors and acknowledgment
- Hubert Hamelin : keep in mind this implementation is personal, and differs from the one which was used in the original
paper.

## License
MIT License, see LICENSE file.

## Project Status
1.0.0 is the very first release ! See the Roadmap section to see everything that is left to do.
