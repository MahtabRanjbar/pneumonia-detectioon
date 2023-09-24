# Import system libs
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
# Import deep learning libraries
import tensorflow as tf

# Import custom modules
from config import Config
from evaluation import (accuracy_score, display_confusion_matrix,
                        evaluate_model, plot_training_history,
                        save_classification_report)
from model import create_model
from preprocess import create_gen, create_image_dataframe


def main():
    # Create a list with the filepaths for training validating and testing
    dir_ = Path(Config.train_data_path)
    train_file_paths = list(dir_.glob(r"**/*.jpeg"))

    dir_ = Path(Config.val_data_path)
    test_file_paths = list(dir_.glob(r"**/*.jpeg"))

    dir_ = Path(Config.test_data_path)
    val_file_paths = list(dir_.glob(r"**/*.jpeg"))

    # Create dataframes for train, validation, and test sets
    train_df = create_image_dataframe(train_file_paths)
    val_df = create_image_dataframe(val_file_paths)
    # Combine train_df and val_df
    train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    test_df = create_image_dataframe(test_file_paths)

    # Create the generators
    (
        train_generator,
        test_generator,
        train_images,
        val_images,
        test_images,
    ) = create_gen(train_df, test_df)

    # Dictionary with the models
    models = {
        "DenseNet121": {"model": tf.keras.applications.DenseNet121, "perf": 0},
        "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
        "EfficientNetB7": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "InceptionResNetV2": {
            "model": tf.keras.applications.InceptionResNetV2,
            "perf": 0,
        },
        "InceptionV3": {"model": tf.keras.applications.InceptionV3, "perf": 0},
        "MobileNetV3Large": {
            "model": tf.keras.applications.MobileNetV3Large,
            "perf": 0,
        },
        "ResNet101": {"model": tf.keras.applications.ResNet101, "perf": 0},
        "ResNet50": {"model": tf.keras.applications.ResNet50, "perf": 0},
        "VGG19": {"model": tf.keras.applications.VGG19, "perf": 0},
        "Xception": {"model": tf.keras.applications.Xception, "perf": 0},
    }

    # Fit the models
    for name, model in models.items():

        # Get the model
        m = create_model(model["model"])
        models[name]["model"] = m

        start = perf_counter()

        # Fit the model
        history = m.fit(
            train_images, validation_data=val_images, epochs=Config.epochs, verbose=1
        )

        # Sav the duration and the val_accuracy
        duration = perf_counter() - start
        duration = round(duration, 2)
        models[name]["perf"] = duration
        print(f"{name:20} trained in {duration} sec")

        val_acc = history.history["val_accuracy"]
        models[name]["val_acc"] = [round(v, 4) for v in val_acc]

    # save predictions of each model
    for name, model in models.items():

        # Predict the label of the test_images
        pred = models[name]["model"].predict(test_images)
        pred = np.argmax(pred, axis=1)

        # Map the label
        labels = train_images.class_indices
        labels = dict((v, k) for k, v in labels.items())
        pred = [labels[k] for k in pred]

        y_test = list(test_df.Label)
        acc = accuracy_score(y_test, pred)
        models[name]["acc"] = round(acc, 4)

    # Create a DataFrame with the results
    models_result = []

    for name, v in models.items():
        models_result.append(
            [
                name,
                models[name]["val_acc"][-1],
                models[name]["acc"],
                models[name]["perf"],
            ]
        )

    df_results = pd.DataFrame(
        models_result,
        columns=["model", "val_accuracy", "accuracy (test set)", "Training time (sec)"],
    )
    df_results.sort_values(by="accuracy (test set)", ascending=False, inplace=True)
    df_results.reset_index(inplace=True, drop=True)
    # Save df_results to a text file
    df_results.to_csv(Config.model_report, index=False, sep='\t')
    print(df_results)

    # getting the best model
    acc = df_results.iloc[0]["accuracy (test set)"]
    best_model = df_results.iloc[0]["model"]
    print(f"Best model: {best_model}")
    print(f"Accuracy on the test set: {acc * 100:.2f}%")

    # Save the best model
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.h5")
    models[best_model]["model"].save(save_path)
    print(f"Best model saved at: {save_path}")

    # Predict the labels of the test_images for the best model
    pred = models[best_model]["model"].predict(test_images)
    pred = np.argmax(pred, axis=1)

    # Map the label
    labels = train_images.class_indices
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in pred]

    y_test = list(test_df.Label)
    acc = accuracy_score(y_test, pred)
    models[name]["acc"] = round(acc, 4)

    # Display confusion matrix
    display_confusion_matrix(y_test, pred, save_path=Config.confusion_matrix_save_path)

    # Save classification report
    save_classification_report(
        y_test, pred, save_path=Config.classification_report_path
    )


if __name__ == "main":
    main()
