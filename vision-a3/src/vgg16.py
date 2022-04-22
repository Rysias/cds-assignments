import tensorflow as tf

# Create logging
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
)


def load_vgg16(input_shape: tf.TensorShape) -> tf.keras.Model:
    # Load the VGG16 model
    logger.info("Loading VGG16 model...")
    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights="imagenet",
        classifier_activation="softmax",
        input_shape=input_shape,
    )
    base_model.trainable = False
    return base_model


def create_model(
    base_model: tf.keras.Model, input_shape: int, num_classes: int = 10,
) -> tf.keras.Model:
    logging.info("Creating model...")
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)  # 10 classes

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    return tf.keras.Model(inputs, outputs)


def compile_model(model: tf.keras.Model, learning_rate: float = 0.001) -> None:
    logger.info("Compiling model...")
    model.compile(
        optimizer="adam",
        learning_rate=learning_rate,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

def finetuneable_vgg16(input_shape, learning_rate=0.001) -> tf.keras.Model:
    """
    Loads the VGG16 model and returns a finetunable model.
    """
    vgg16 = load_vgg16(input_shape)
    model = create_model(vgg16, input_shape)
    compile_model(model, learning_rate=learning_rate))
    return model 