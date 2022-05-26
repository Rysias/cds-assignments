import tensorflow as tf
import tensorflow_hub as hub
import logging

# add basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def define_model(dropout) -> tf.keras.Model:
    model_path = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
    hub_layer = hub.KerasLayer(
        model_path, input_shape=[], dtype=tf.string, trainable=False
    )
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    return model


def compile_model(model: tf.keras.Model,) -> None:
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def create_model(dropout=0.0) -> tf.keras.Model:
    model = define_model(dropout=dropout)
    compile_model(model)
    return model
