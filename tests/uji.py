import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from src.augfeat.balancer import Balancer
from src.augfeat.config import AUTOENCODER_TRAINING_CONFIG_MEDIUM
from src.augfeat.custom_types import DataTypes
import tensorflow as tf

# Source: https://www.kaggle.com/code/sorokin/convert-ujipenchars2-to-the-quick-draw-dataset

# TODO: add Dataframe format for data with cols (label, drawing) -> (str, np.array)
# TODO: or convert the uji dataset in previously handled format (NUMPY in repos)
# TODO: Split train and test sets later ?

IMG_DIM = 56


def stack_it(drawing):
    # unwrap the list
    in_strokes = [(xi, yi, i) for i, (x, y) in enumerate(drawing) for xi, yi in zip(x, y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:, 2] = [1] + np.diff(c_strokes[:, 2]).tolist()
    c_strokes[:, 2] += 1  # since 0 is no stroke
    return c_strokes


def plot_examples(df: pd.DataFrame):
    df1 = df[df.index.get_level_values("character").isin(["a", "r", "m"])]
    print(len(df), len(df1))
    print(df1["drawing"].head())

    fig, m_axs = plt.subplots(3, 3, figsize=(16, 16))
    rand_idxs = np.random.choice(range(len(df1)), size=9)
    for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
        image = stack_it(df1.iloc[c_id]["drawing"])
        lab_idx = np.cumsum(image[:, 2] - 1)
        for i in np.unique(lab_idx):
            c_ax.plot(image[lab_idx == i, 0], np.max(image[:, 1]) - image[lab_idx == i, 1], '.-')
        c_ax.axis('off')
        c_ax.set_title(df1.index.values[c_id])
    plt.show()


def uji_to_dataframe():
    df = pd.DataFrame({
        'subset': pd.Series(dtype='string'),
        'site': pd.Series(dtype='string'),
        'writer': pd.Series(dtype='string'),
        'character': pd.Series(dtype='string'),
        'repetitions': pd.Series(dtype='int'),
        'numstrokes': pd.Series(dtype='int'),
        'numpoints': pd.Series(dtype='int'),
        'drawing': pd.Series(dtype='object')
    })

    df = df.set_index(["subset", "site", "writer", "character", "repetitions"])

    with open("ujipenchars2.txt", "r") as file:
        for line in file:
            words = line.strip().split()
            if words[0] == "//":
                continue
            elif words[0] == "WORD":
                info = words[2].split("_")
                session = info[2].split("-")
                index = (info[0], info[1], session[0], words[1], int(session[1]))
                df.loc[index] = [None, None, None]
            elif words[0] == "NUMSTROKES":
                df.loc[index]["numstrokes"] = int(words[1])
                df.loc[index]["numpoints"] = 0
                df.loc[index]["drawing"] = []
            elif words[0] == "POINTS":
                numpoints = int(words[1])
                points = words[3:]
                x = [int(x) for x in points[0::2]]
                y = [int(y) for y in points[1::2]]
                assert len(x) == numpoints, (len(x), numpoints)
                assert len(y) == numpoints, (len(y), numpoints)
                df.loc[index]["numpoints"] += numpoints
                df.loc[index]["drawing"].append([x, y])

    return df


def uji_dataframe_to_numpy_dataset(df: pd.DataFrame):

    # Création du master repo
    if not os.path.isdir('../../Experiments/uji'):
        os.mkdir('../../Experiments/uji')

    # Création des répertoires de classes en loop
    # Remplissage de ces répertoires en loop
    counter = 0
    for index, row in tqdm(df.iterrows(), desc='generating numpy arrays', total=len(df)):

        _, _, _, character, repetitions = index
        repo_path = f'../Experiments/uji/{character}'
        if not os.path.isdir(repo_path):
            os.mkdir(repo_path)
        element_path = repo_path + f'/{character}_{str(counter)}'

        # generate a picture
        plt.figure()
        image = stack_it(row['drawing'])
        lab_idx = np.cumsum(image[:, 2] - 1)
        for i in np.unique(lab_idx):
            plt.plot(image[lab_idx == i, 0], np.max(image[:, 1]) - image[lab_idx == i, 1], '-')
        plt.axis('off')
        plt.savefig(element_path + '.png')

        # load image
        img = Image.open(element_path + '.png')

        # convert to grayscale
        img = ImageOps.grayscale(img)
        img = ImageOps.invert(img)

        # local normalization
        img = img.resize((IMG_DIM, IMG_DIM))

        # transform into a numpy array
        img_array = np.array(img)

        # save numpy array
        np.save(element_path + '.npy', img_array)
        counter += 1

        # delete picture
        os.remove(element_path + '.png')

"""
# Transform uji dataset into images converted to numpy arrays
print('Start reading file...')
df = uji_to_dataframe()
print('Finished reading file.')
uji_dataframe_to_numpy_dataset(df)
"""

class_name = '@'

with tf.device('cpu:0'):
    dataset_path = '../../Experiments/uji'
    balancer = Balancer(dataset_path, DataTypes.NUMPY)
    balancer.augment_class(class_name, 9, AUTOENCODER_TRAINING_CONFIG_MEDIUM)

# Results visualisation
augmented_path = f'../Experiments/augmented_dataset/{class_name}'
fig, axs = plt.subplots(3, 3)
for file_name, ax in zip(os.listdir(augmented_path), axs.flat):
    array = np.load(os.path.join(augmented_path, file_name))
    ax.imshow(array, cmap=plt.cm.binary)
plt.show()
