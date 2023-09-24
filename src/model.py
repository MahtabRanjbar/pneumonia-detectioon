import tensorflow as tf
from tensorflow.keras.layers import Dense


def create_model(model_class):
    """
    Create a custom model based on a model class.

    Args:
        model_class (tf.keras.Model): Model class.

    Returns:
        model (tf.keras.Model): Custom model.
    """
    # Load the pretrained model
    base_model = model_class(weights="imagenet", include_top=False, pooling="avg")
    base_model.trainable = False

    # Add custom layers on top of the base model
    x = Dense(128, activation="relu")(base_model.output)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(2, activation="softmax")(x)

    # Create the custom model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
