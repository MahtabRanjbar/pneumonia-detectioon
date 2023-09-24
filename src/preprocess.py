# import system libs
import pathlib
from typing import List

import pandas as pd
import tensorflow as tf

from config import Config


def create_image_dataframe(filepaths: List[str]):
    """
    Creates a DataFrame with filepaths and corresponding labels.

    Args:
        filepaths (List[str]): A list of filepaths.

    Returns:
        pd.DataFrame: A DataFrame containing the filepaths and labels
    """

    labels = [pathlib.Path(filepath).parent.name for filepath in filepaths]

    filepath_series = pd.Series(filepaths, name="Filepath").astype(str)
    labels_series = pd.Series(labels, name="Label")

    # Concatenate filepaths and labels
    df = pd.concat([filepath_series, labels_series], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1, random_state=Config.seed).reset_index(drop=True)

    return df


def create_gen(train_df, test_df):
    """
    Create image data generators for training, validation, and testing.

    Returns:
        train_generator (ImageDataGenerator): Image data generator for training data.
        test_generator (ImageDataGenerator): Image data generator for testing data.
        train_images (DirectoryIterator): Iterator for training images.
        val_images (DirectoryIterator): Iterator for validation images.
        test_images (DirectoryIterator): Iterator for testing images.
    """
    # Define common image data generator arguments
    common_args = {
        "preprocessing_function": tf.keras.applications.mobilenet_v2.preprocess_input,
        "class_mode": "categorical",
        "batch_size": 32,
        "seed": 0,
        "target_size": (224, 224),
    }

    # Define augmentation arguments
    augmentation_args = {
        "rotation_range": 30,
        "zoom_range": 0.15,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.15,
        "horizontal_flip": True,
        "fill_mode": "nearest",
    }

    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, **augmentation_args
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    # Flow from DataFrame arguments
    flow_args = {"x_col": "Filepath", "y_col": "Label", "color_mode": "rgb"}

    # Flow from DataFrame for training images
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df, subset="training", **common_args, **flow_args
    )

    # Flow from DataFrame for validation images
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        shuffle=False,
        subset="validation",
        **common_args,
        **flow_args
    )

    # Flow from DataFrame for test images
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df, shuffle=False, **common_args, **flow_args
    )

    return train_generator, test_generator, train_images, val_images, test_images
