import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import imblearn
import itertools as it
import os
from statistics import mean
from shutil import copy2
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    BatchNormalization,
    InputLayer,
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from keras import backend as K

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
network_bias = None

"""
Source: https://stackoverflow.com/questions/55496696/how-to-correctly-implement-f1-score-as-scoring-metric-in-grid-search-with-keras
"""


def f1_score(y_true, y_pred):  # taken from old keras source code
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


"""
Create a Neural Network with the specified parameters
"""


def create_model(
    num_units, reg_lambda, dropout_rate, optimizer, input_dim=104, initial_bias=False
):
    if initial_bias:
        model = Sequential(
            [
                Dense(
                    units=num_units,
                    input_shape=(input_dim,),
                    activation="relu",
                    kernel_initializer="random_normal",
                    kernel_regularizer=l2(reg_lambda),
                ),
                Dropout(rate=dropout_rate),
                Dense(
                    units=num_units,
                    activation="relu",
                    kernel_initializer="random_normal",
                    kernel_regularizer=l2(reg_lambda),
                ),
                Dropout(rate=dropout_rate),
                Dense(units=1, activation="sigmoid", bias_initializer=network_bias),
            ]
        )
    else:
        model = Sequential(
            [
                Dense(
                    units=num_units,
                    input_shape=(input_dim,),
                    activation="relu",
                    kernel_initializer="random_normal",
                    kernel_regularizer=l2(reg_lambda),
                ),
                Dropout(rate=dropout_rate),
                Dense(
                    units=num_units,
                    activation="relu",
                    kernel_initializer="random_normal",
                    kernel_regularizer=l2(reg_lambda),
                ),
                Dropout(rate=dropout_rate),
                Dense(units=1, activation="sigmoid"),
            ]
        )

    metrics = [
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        f1_score,
    ]

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

    return model


"""
Grid search for the best parameters. Not used because it's too complicated and time consuming.
"""


def find_best_model_params(X_data, y_data):
    epochs = 100
    batch_sizes = [100, 200]
    optimizers = ["SGD", "Adam"]
    num_units = [20, 50, 100]
    dropout_rate = [0.3, 0.7]
    reg_lambda = [0.001, 0.01]

    model_param = dict(
        batch_size=batch_sizes,
        optimizer=optimizers,
        num_units=num_units,
        dropout_rate=dropout_rate,
        reg_lambda=reg_lambda,
    )
    f1_scorer = make_scorer(f1_score)

    cv_model = KerasClassifier(build_fn=create_model, epochs=epochs)
    grid = GridSearchCV(
        estimator=cv_model, param_grid=model_param, n_jobs=2, cv=5, scoring=f1_scorer
    )
    grid_result = grid.fit(X_data, y_data)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_["mean_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, param in zip(means, params):
        print("%f with: %r" % (mean, param))

    return None


def train_deep_NN(
    X_data,
    y_data,
    label_count,
    model_temp_file,
    best_model_files,
    curr_model_files,
    model_param,
):

    num_units = model_param["num_units"]
    reg_lambda = model_param["reg_lambda"]
    dropout_rate = model_param["dropout_rate"]
    optimizer = model_param["optimizer"]
    epochs = model_param["epochs"]
    batch_size = model_param["batch_size"]
    weighted = model_param["weighted"]
    bias = model_param["bias"]
    cv_splits = model_param["cv_splits"]

    network_bias_const = np.log([label_count[1] / label_count[0]])
    global network_bias
    network_bias = keras.initializers.Constant(
        network_bias_const
    )  # Custom bias for the neural network

    input_dim = len(X_data[0])
    early_stop = EarlyStopping(monitor="loss", mode="min", patience=15)
    checkpoint = ModelCheckpoint(
        model_temp_file,
        monitor="val_f1",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    models_cv = {}
    model_attr = model_param  # Model parameters and validation score

    if not os.path.isfile(best_model_files["attributes"]):
        best_f1_score = 0
    else:
        scores_df = pd.read_csv(best_model_files["attributes"])
        best_f1_score = scores_df["f1"].values.tolist()[
            0
        ]  # Get the f1 score of the best model if exists

    model_scores_df = pd.DataFrame()
    val_idx_per_split = []

    cv_counter = 1
    stf_kfold = StratifiedKFold(n_splits=cv_splits)
    for train_idx, val_idx in stf_kfold.split(
        X_data, y_data
    ):  # Train different models on different train-test splits and evaluate them
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        val_idx_per_split.append(val_idx)

        # X_train, y_train, label_count = Oversample(X_train,y_train,sampling_factor=0.3)

        bin_count = np.bincount(y_train)
        num_samples = len(y_data)
        w_0 = (1 / bin_count[0]) * num_samples / 2.0
        w_1 = (1 / bin_count[1]) * num_samples / 2.0
        class_weight = {
            0: w_0,
            1: w_1,
        }  # Custom weights to compensate for class imbalance ==> Leads to bad results
        # class_weight = {0: 1, 1: bin_count[0]/bin_count[1]}

        model = create_model(
            num_units=num_units,
            reg_lambda=reg_lambda,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            input_dim=input_dim,
            initial_bias=bias,
        )

        if weighted:
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, checkpoint],
                validation_data=(X_val, y_val),
                class_weight=class_weight,
            )
        else:
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, checkpoint],
                validation_data=(X_val, y_val),
            )

        trained_model = create_model(
            num_units=num_units,
            reg_lambda=reg_lambda,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
        )
        trained_model.load_weights(model_temp_file)
        models_cv[cv_counter] = trained_model

        scores = trained_model.evaluate(X_val, y_val, verbose=2)

        print(
            "\n-------------------------------------------------------------------------Fold %d-------------------------------------------------------------------------"
            % (cv_counter)
        )
        for metric, score in zip(model.metrics_names, scores):
            if metric == "f1":
                p, r = scores[-3], scores[-2]
                real_f1 = 2 * p * r / (p + r)
                scores[-1] = real_f1
                print("%s: %f" % (metric, real_f1))
            else:
                print("%s: %f" % (metric, score), end="\t")
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------\n"
        )

        model_scores_df = model_scores_df.append(
            dict((m, sc) for m, sc in zip(model.metrics_names, scores)),
            ignore_index=True,
        )
        cv_counter += 1

    model_scores_df.to_csv(curr_model_files["scores"], index=False, float_format="%.5f")

    best_avg_model, overall_best_f1_score = get_best_model(
        models_cv, X_data, y_data, val_idx_per_split
    )
    best_avg_model.save_weights(curr_model_files["model"])
    model_attr["f1"] = overall_best_f1_score
    curr_model_attr_df = pd.DataFrame(model_attr, index=[0])
    curr_model_attr_df.to_csv(
        curr_model_files["attributes"], index=False, float_format="%.5f"
    )

    best_model = create_model(
        num_units=num_units,
        reg_lambda=reg_lambda,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
    )

    if not os.path.isfile(best_model_files["model"]):
        copy2(curr_model_files["model"], best_model_files["model"])
        copy2(curr_model_files["attributes"], best_model_files["attributes"])
        copy2(curr_model_files["scores"], best_model_files["scores"])

        best_model.load_weights(best_model_files["model"])

    else:
        new_model = False
        if overall_best_f1_score > best_f1_score:
            best_avg_model.save_weights(best_model_files["model"])
            copy2(curr_model_files["attributes"], best_model_files["attributes"])
            copy2(curr_model_files["scores"], best_model_files["scores"])
            best_model = best_avg_model
            new_model = True

    return best_model, new_model


def Create_Predictions(model, X_test, output_file):
    y_test = model.predict_classes(X_test)
    output_df = pd.DataFrame(data=y_test, columns=["Active"])
    output_df.to_csv(output_file, header=False, index=False)
    return None


"""
Retrieve the model from all trained models that achieves the best average validation f1 score
"""


def get_best_model(models, X_data, y_data, val_idx_per_split):
    avg_model_scores = []
    for i in range(len(models)):
        curr_model = models[i + 1]
        model_score = []
        for j in range(len(val_idx_per_split)):
            val_idx = val_idx_per_split[j]
            X_val, y_val = X_data[val_idx], y_data[val_idx]
            val_score = curr_model.evaluate(X_val, y_val)
            p, r = val_score[-3], val_score[-2]
            real_f1 = 2 * p * r / (p + r)
            val_score[-1] = real_f1
            model_score.append(real_f1)

        avg_model_scores.append(mean(model_score))

    print(avg_model_scores)
    best_f1_score = max(avg_model_scores)
    best_model_idx = avg_model_scores.index(best_f1_score)
    best_model = models[best_model_idx + 1]
    return best_model, best_f1_score


"""
Onehot encoding used to transform aminoacids to feature vectors, which can be served as input to the Neural Network
"""


def encode_protein(train_file, test_file):

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    amino_seq_train = train_data["Sequence"].values.tolist()
    amino_acid_train = list(map(lambda l: [x for x in l], amino_seq_train))

    amino_seq_test = test_data["Sequence"].values.tolist()
    amino_acid_test = list(map(lambda l: [x for x in l], amino_seq_test))

    one_hot = OneHotEncoder(categories=4 * [AMINOACIDS])
    one_hot.fit(amino_acid_train)

    X_train = one_hot.transform(amino_acid_train).toarray()
    X_test = one_hot.transform(amino_acid_test).toarray()
    X_train, X_test = list(map(lambda l: list(l), X_train)), list(
        map(lambda l: list(l), X_test)
    )
    y_train = np.array(train_data["Active"].values.tolist())

    one_hot_charged = OneHotEncoder(
        categories=4 * [CHARGED_AA], handle_unknown="ignore"
    )
    one_hot_charged.fit(amino_acid_train)

    charged_acids_train = one_hot_charged.transform(amino_acid_train).toarray()
    charged_acids_test = one_hot_charged.transform(amino_acid_test).toarray()
    charged_acids_train, charged_acids_test = list(
        map(lambda l: list(l), charged_acids_train)
    ), list(map(lambda l: list(l), charged_acids_test))

    X_train = np.array(list(map(list.__add__, X_train, charged_acids_train)))
    X_test = np.array(list(map(list.__add__, X_test, charged_acids_test)))
    # print(one_hot.categories_)
    return X_train, y_train, X_test


"""
Oversample the minority class to compensate for the large class imbalance
"""


def oversample(X_data, y_data, sampling_factor):
    # def Oversample(train_data,sampling_factor=0.2):
    # X_data, y_data = np.array(train_data.iloc[:,:-1].values.tolist()), np.array(train_data.iloc[:,-1].values.tolist())

    oversampler = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy=sampling_factor
    )
    oversampled_X, oversampled_y = oversampler.fit_resample(X_data, y_data)

    label_count = np.bincount(oversampled_y)
    # print(label_count)
    return oversampled_X, oversampled_y, label_count


def main():
    train_file = "train.csv"
    test_file = "test.csv"
    model_temp_file = "model_temp.h5"

    best_model_file = "best_model.h5"
    best_model_attr_file = "best_model_attr.csv"
    best_model_scores_file = "best_model_scores.csv"
    curr_model_file = "curr_model.h5"
    curr_model_scores_file = "curr_model_scores.csv"
    curr_model_attr_file = "curr_model_attr.csv"
    output_file = "predictions.csv"
    output2_file = "curr_predictions.csv"

    best_model_files = {
        "model": best_model_file,
        "attributes": best_model_attr_file,
        "scores": best_model_scores_file,
    }
    curr_model_files = {
        "model": curr_model_file,
        "attributes": curr_model_attr_file,
        "scores": curr_model_scores_file,
    }

    model_param = {
        "num_units": 100,
        "reg_lambda": 0.0,
        "dropout_rate": 0.3,
        "optimizer": "adam",
        "epochs": 250,
        "batch_size": 500,
        "cv_splits": 5,
        "weighted": False,
        "bias": True,
    }

    X_train, y_train, X_test = encode_protein(train_file, test_file)
    label_count = np.bincount(y_train)

    best_model, new_model = train_deep_NN(
        X_train,
        y_train,
        label_count,
        model_temp_file,
        best_model_files,
        curr_model_files,
        model_param,
    )

    if new_model:
        best_model.load_weights(best_model_file)
        Create_Predictions(best_model, X_test, output_file)
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print("New best predictions available...")
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
    else:
        curr_model = create_model(
            num_units=model_param["num_units"],
            reg_lambda=model_param["reg_lambda"],
            dropout_rate=model_param["dropout_rate"],
            optimizer=model_param["optimizer"],
        )
        curr_model.load_weights(curr_model_file)
        Create_Predictions(curr_model, X_test, output2_file)
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print("New predictions available...")
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------"
        )

    return None


if __name__ == "__main__":
    main()
