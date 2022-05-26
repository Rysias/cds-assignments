"""
Adapted from https://github.com/RayWilliam46/FineTune-DistilBERT/
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import tensorflow as tf
from tensorflow.keras import backend as K
from transformers import DistilBertTokenizerFast  # type: ignore
from transformers import TFDistilBertModel, DistilBertConfig  # type: ignore
import logging

logging.basicConfig(level=logging.INFO)

# Set parameters:
params = {
    "MAX_LENGTH": 128,
    "EPOCHS": 6,
    "LEARNING_RATE": 5e-5,
    "FT_EPOCHS": 2,
    "OPTIMIZER": "adam",
    "FL_GAMMA": 2.0,
    "FL_ALPHA": 0.2,
    "BATCH_SIZE": 64,
    "NUM_STEPS": 140_000 // 64,
    "DISTILBERT_DROPOUT": 0.2,
    "DISTILBERT_ATT_DROPOUT": 0.2,
    "LAYER_DROPOUT": 0.2,
    "KERNEL_INITIALIZER": "GlorotNormal",
    "BIAS_INITIALIZER": "zeros",
    "POS_PROBA_THRESHOLD": 0.5,
    "ADDED_LAYERS": "Dense 256, Dense 32, Dropout 0.2",
    "LR_SCHEDULE": "5e-5 for 6 epochs, Fine-tune w/ adam for 2 epochs @2e-5",
    "FREEZING": "All DistilBERT layers frozen for 6 epochs, then unfrozen for 2",
    "CALLBACKS": "[early_stopping w/ patience=0]",
    "RANDOM_STATE": 42,
}


def batch_encode(tokenizer, texts, batch_size=256, max_length=128):
    """""" """
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.
    
    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """ """"""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch,
            max_length=max_length,
            padding="max_length",  # implements dynamic padding
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        input_ids.extend(inputs["input_ids"])
        attention_mask.extend(inputs["attention_mask"])
    # DEBUG STUFF
    target_len = len(input_ids[0])
    assert all(len(ids) == target_len for ids in input_ids), ValueError("we found our problem!")
    logging.info(f"{input_ids[:5] = }")    
    input_tensors = tf.convert_to_tensor(input_ids)
    logging.info(f"{attention_mask[:5] = }")
    attention_mask_tensors = tf.convert_to_tensor(attention_mask)
    return input_tensors, attention_mask_tensors


def focal_loss(gamma=2.0, alpha=0.2):
    """""" """
    Function that computes the focal loss.
    
    Code adapted from https://gist.github.com/mkocabas/62dcd2f14ad21f3b25eac2d39ec2cc95
    """ """"""

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )

    return focal_loss_fixed


def build_model(transformer, max_length=128):
    """
    Template for building a model off of the BERT or DistilBERT architecture
    for a binary classification task.
    
    Input:
      - transformer:  a base Hugging Face transformer model object (BERT or DistilBERT)
                      with no added classification head attached.
      - max_length:   integer controlling the maximum number of encoded tokens 
                      in a given sequence.
    
    Output:
      - model:        a compiled tf.keras.Model with added classification layers 
                      on top of the base pre-trained model architecture.
    """

    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=params["RANDOM_STATE"])

    # Define input layers
    input_ids_layer = tf.keras.layers.Input(
        shape=(max_length,), name="input_ids", dtype="int32"
    )
    input_attention_layer = tf.keras.layers.Input(
        shape=(max_length,), name="input_attention", dtype="int32"
    )

    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]

    # We only care about DistilBERT's output for the [CLS] token, which is located
    # at index 0.  Splicing out the [CLS] tokens gives us 2D data.
    cls_token = last_hidden_state[:, 0, :]

    D1 = tf.keras.layers.Dropout(params["LAYER_DROPOUT"], seed=params["RANDOM_STATE"])(
        cls_token
    )

    X = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer=weight_initializer,
        bias_initializer="zeros",
    )(D1)

    D3 = tf.keras.layers.Dropout(params["LAYER_DROPOUT"], seed=params["RANDOM_STATE"])(
        X
    )

    # Define a single node that makes up the output layer (for binary classification)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=weight_initializer,  # CONSIDER USING CONSTRAINT
        bias_initializer="zeros",
    )(D3)

    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

    # Compile the model
    model.compile(
        tf.keras.optimizers.Adam(lr=params["LEARNING_RATE"]),
        loss=focal_loss(),
        metrics=["accuracy"],
    )

    return model


# Load training data
train_raw = pd.read_csv("input/train.csv")
train_raw["label"] = (
    train_raw["threat"] | train_raw["insult"] | train_raw["severe_toxic"]
)
train = train_raw[["comment_text", "label"]].rename({"comment_text": "text"}, axis=1)

# train test split
train_df, val_df = train_test_split(train, test_size=0.1, random_state=542)

X_train = train_df["text"]
X_valid = val_df["text"]
y_train = train_df["label"]
y_valid = val_df["label"]

# Load test data
test = pd.read_csv("input/VideoCommentsThreatCorpus.csv")
X_test = test["text"]
y_test = test["label"]

logging.info(f"{X_test.head() = }")

# Check data
logging.info(f"Our training data has {len(X_train.index)} rows.")
logging.info(f"Our validation data has {len(X_valid.index)} rows.")
logging.info(f"Our test data has {len(X_test.index)} rows.")


# Instantiate DistilBERT tokenizer...we use the Fast version to optimize runtime
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Encode X_train
X_train_ids, X_train_attention = batch_encode(tokenizer, X_train.tolist())

# Encode X_valid
X_valid_ids, X_valid_attention = batch_encode(tokenizer, X_valid.tolist())

# Encode X_test
X_test_ids, X_test_attention = batch_encode(tokenizer, X_test.tolist())


# The bare, pre-trained DistilBERT transformer model outputting raw hidden-states
# and without any specific head on top.
config = DistilBertConfig(
    dropout=params["DISTILBERT_DROPOUT"],
    attention_dropout=params["DISTILBERT_ATT_DROPOUT"],
    output_hidden_states=True,
)
distilBERT = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

# Freeze DistilBERT layers to preserve pre-trained weights
for layer in distilBERT.layers:
    layer.trainable = False

# Build model
model = build_model(distilBERT)

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", mode="min", min_delta=0, patience=0, restore_best_weights=True
)

# Train the model
train_history1 = model.fit(
    x=[X_train_ids, X_train_attention],
    y=y_train.to_numpy(),
    epochs=params["EPOCHS"],
    batch_size=params["BATCH_SIZE"],
    steps_per_epoch=params["NUM_STEPS"],
    validation_data=([X_valid_ids, X_valid_attention], y_valid.to_numpy()),
    callbacks=[early_stopping],
    verbose=2,
)


# Generate predictions
y_pred = model.predict([X_test_ids, X_test_attention])
y_pred_thresh = np.where(y_pred >= params["POS_PROBA_THRESHOLD"], 1, 0)

# Get evaluation results
accuracy = accuracy_score(y_test, y_pred_thresh)
auc_roc = roc_auc_score(y_test, y_pred)

# Log the ROC curve
fpr, tpr, thresholds = roc_curve(y_test.to_numpy(), y_pred)


logging.info("Accuracy:  ", accuracy)  # 0.9218
logging.info("ROC-AUC:   ", auc_roc)  # 0.9691

