from typing import List, Union
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

import keras
from keras.models import Sequential, load_model
from keras.layers import (
    Dense,
    Dropout,
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier

# Constants
AMINOACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "Y",
]
CHARGED_AA = ["R", "H", "K", "D", "E"]


def f1(y_true, y_pred):
    """
    Compute the F1 score using TensorFlow/Keras.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels (probabilities).

    Returns:
        f1: Scalar F1 score.
    """
    y_pred = tf.round(y_pred)
    y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


def build_model(
    num_units: List,
    regularization: float,
    dropout_rate: float,
    activation: str,
    loss: str,
    optimizer: str,
    input_dim: int = 104,
):
    """
    Build a modular sequential model.

    Args:
        num_units (int): Number of units in each Dense layer.
        regularization (float): L2 regularization strength.
        dropout_rate (float): Dropout rate for the Dropout layers.
        activation (str): Activation function for the Dense layers.
        loss (str): Loss function for compiling the model
        optimizer: Optimizer for adjusting the weights of the model.
        input_dim (int): Input dimension of the model.

    Returns:
        keras.models.Sequential: The compiled Keras model.
    """

    num_layers = len(num_units)
    layers = []

    # Add the first Dense layer with input_shape specified
    layers.append(
        Dense(
            units=num_units[0],
            input_shape=(input_dim,),
            activation=activation,
            kernel_initializer="random_normal",
            kernel_regularizer=l2(regularization),
        )
    )

    if dropout_rate > 0:
        layers.append(Dropout(rate=dropout_rate))

    # Add intermediate Dense layers
    for i in range(1, num_layers):
        layers.append(
            Dense(
                units=num_units[i],
                activation=activation,
                kernel_initializer="random_normal",
                kernel_regularizer=l2(regularization),
            )
        )

        if dropout_rate > 0:
            layers.append(Dropout(rate=dropout_rate))

    # Add the output layer
    layers.append(Dense(units=1, activation="sigmoid"))

    # Create the model
    model = Sequential(layers)

    # Define metrics
    metrics = [
        # keras.metrics.FalseNegatives(name="fn"),
        # keras.metrics.FalsePositives(name="fp"),
        # keras.metrics.TrueNegatives(name="tn"),
        # keras.metrics.TruePositives(name="tp"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        # keras.metrics.F1Score(name="f1", average="micro"),
        f1,  # Custom F1 score
    ]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def train_model(
    X_train: np.array,
    y_train: np.array,
    model_params: dict,
    early_stop_params: dict = None,
    checkpoint_params: dict = None,
    class_weight: dict = None,
    epochs: int = 50,
    batch_size: int = 64,
):

    # Define callbacks
    callbacks = []
    if early_stop_params:
        callbacks.append(EarlyStopping(**early_stop_params))
    if checkpoint_params:
        callbacks.append(ModelCheckpoint(**checkpoint_params))

    # Build the model
    model = build_model(**model_params)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    return model, history


def cross_validate_model(
    X_data: np.array,
    y_data: np.array,
    model_params: dict,
    cv_splits: int = 5,
    seed: int = 42,
    **train_kwargs,
):
    stf_kfold = StratifiedKFold(n_splits=cv_splits, random_state=seed)
    cv_scores = []

    for train_idx, val_idx in stf_kfold.split(X_data, y_data):
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        # Compute class weights
        class_weight = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        train_kwargs["class_weight"] = {0: class_weight[0], 1: class_weight[1]}

        # Train the model
        model, history = train_model(X_train, y_train, model_params, **train_kwargs)
        scores = model.evaluate(X_val, y_val, verbose=2)

        cv_scores.append(scores)

    return cv_scores


def cross_validate_rf_model(
    X_data: np.array,
    y_data: np.array,
    model_params: dict,
    cv_splits: int = 5,
    seed: int = 42,
):
    stf_kfold = StratifiedKFold(n_splits=cv_splits, random_state=seed)
    cv_scores = []

    for train_idx, val_idx in stf_kfold.split(X_data, y_data):
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        # Train the model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        model_score = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

        cv_scores.append(model_score)

    return cv_scores


def predict(model, X_test, output_file):
    y_test = model.predict(X_test)
    y_labels = np.round(y_test).astype(int)  # Convert probabilities to binary labels
    output_df = pd.DataFrame(data=y_labels, columns=["Active"])
    output_df.to_csv(output_file, header=False, index=False)

    print(f"Predictions saved to {output_file}")
    return


"""
Grid search for the best parameters. Not used because it's time consuming.
"""


def find_best_model_params(X_data, y_data):
    epochs = 100
    batch_sizes = [64, 128]
    optimizers = ["Adam"]
    num_units = [[50, 50], [100, 100]]
    dropout_rate = [0.3]
    reg_lambda = [0.001, 0.1]
    activation = ["relu"]
    loss = ["binary_crossentropy"]

    model_param = dict(
        batch_size=batch_sizes,
        num_units=num_units,
        regularization=reg_lambda,
        dropout_rate=dropout_rate,
        activation=activation,
        loss=loss,
        optimizer=optimizers,
    )

    f1_scorer = make_scorer(f1_score)

    cv_model = KerasClassifier(build_fn=build_model, epochs=epochs)
    grid = GridSearchCV(
        estimator=cv_model, param_grid=model_param, n_jobs=2, cv=5, scoring=f1_scorer
    )
    grid_result = grid.fit(X_data, y_data)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    mean_cv_score = grid_result.cv_results_["mean_test_score"]
    params = grid_result.cv_results_["params"]

    for mean, param in zip(mean_cv_score, params):
        print("%f with: %r" % (mean, param))

    return


"""
Onehot encoding used to transform aminoacids to feature vectors, which can be served as input to the Neural Network
"""


def encode_protein(train_file, test_file):

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Convert string to individual amino acids
    amino_seq_train = train_data["Sequence"].values.tolist()
    amino_acid_train = list(map(lambda l: [x for x in l], amino_seq_train))

    amino_seq_test = test_data["Sequence"].values.tolist()
    amino_acid_test = list(map(lambda l: [x for x in l], amino_seq_test))

    # Use one-hot encoding to transform amino acids to feature vectors
    # Positional encoding is used to encode the amino acids (i.e. the position of the amino acid in the sequence)
    one_hot = OneHotEncoder(categories=4 * [AMINOACIDS])
    one_hot.fit(amino_acid_train)

    X_train = one_hot.transform(amino_acid_train).toarray()
    X_test = one_hot.transform(amino_acid_test).toarray()
    X_train, X_test = list(map(lambda l: list(l), X_train)), list(
        map(lambda l: list(l), X_test)
    )

    # Add charged amino acids as an additional feature
    one_hot_charged = OneHotEncoder(
        categories=4 * [CHARGED_AA], handle_unknown="ignore"
    )
    one_hot_charged.fit(amino_acid_train)

    charged_acids_train = one_hot_charged.transform(amino_acid_train).toarray()
    charged_acids_test = one_hot_charged.transform(amino_acid_test).toarray()
    charged_acids_train, charged_acids_test = list(
        map(lambda l: list(l), charged_acids_train)
    ), list(map(lambda l: list(l), charged_acids_test))

    # Concatenate the charged amino acid features to the original feature vectors
    X_train = np.array(list(map(list.__add__, X_train, charged_acids_train)))
    X_test = np.array(list(map(list.__add__, X_test, charged_acids_test)))
    y_train = np.array(train_data["Active"].values.tolist())

    # print(one_hot.categories_)

    return X_train, y_train, X_test


def train_val_split(
    X_train: np.array, y_train: np.array, val_size: float = 0.2, seed: int = 42
):
    X_train, y_train, X_val, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed
    )

    return X_train, y_train, X_val, y_val


"""
Oversample the minority class to compensate for the large class imbalance
"""


def oversample(
    X_data: np.array, y_data: np.array, sampling_factor: float = 0.2, seed: int = 42
):

    print("Over-sampling the minority class")

    oversampler = RandomOverSampler(
        sampling_strategy=sampling_factor, random_state=seed
    )
    oversampled_X, oversampled_y = oversampler.fit_resample(X_data, y_data)

    return oversampled_X, oversampled_y


def main():

    with open("config/params.yaml") as f:
        params = yaml.safe_load(f)

    # Set the seed for reproducibility
    SEED = params["seed"]
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    CV_SPLITS = params["cv_splits"]
    SAMPLING_FACTOR = params["sampling_factor"]

    # Load the data files
    train_file = params["files"]["train"]
    test_file = params["files"]["test"]
    output_file = params["files"]["output"]

    # Load the model parameters
    nn_params = params["nn"]
    nn_train_params = params["nn_train"]
    early_stop_params = params["early_stop"]
    checkpoint_params = params["checkpoint"]
    rf_params = params["rf"]

    # Encode the protein sequences using one-hot encoding
    X_train, y_train, X_test = encode_protein(train_file, test_file)
    label_count = np.bincount(y_train)
    print(f"Class 0: {label_count[0]}, Class 1: {label_count[1]}")

    # Split the training data into training and validation sets
    # X_train, y_train, X_val, y_val = train_val_split(X_train, y_train)

    # Oversample the minority class
    X_train, y_train = oversample(
        X_train, y_train, sampling_factor=SAMPLING_FACTOR, seed=SEED
    )
    label_count = np.bincount(y_train)
    print(f"Class 0: {label_count[0]}, Class 1: {label_count[1]}")

    # Train the Neural Network model
    model, history = train_model(
        X_train,
        y_train,
        nn_params,
        early_stop_params=early_stop_params,
        checkpoint_params=checkpoint_params,
        **nn_train_params,
    )

    # Predict on the test set
    predict(model, X_test, output_file)


if __name__ == "__main__":
    main()
