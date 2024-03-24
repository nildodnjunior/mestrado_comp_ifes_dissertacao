import logging

try:
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True
    try:
        tf.logging.set_verbosity(tf.logging.ERROR)
    except AttributeError:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import scipy

from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.models.classifiers.lstm_base import _get_optimizer
from asreview.models.classifiers.utils import _set_class_weight


class NN2LayerClassifier2(BaseTrainClassifier):

    name = "nn_2l_example"
    label = "Teste de criação de extensão com rede neural"

    def __init__(self):

        super().__init__()
        def __init__(self,
                 dense_width=128,
                 optimizer='rmsprop',
                 learn_rate=1.0,
                 regularization=0.01,
                 verbose=0,
                 epochs=35,
                 batch_size=32,
                 shuffle=False,
                 class_weight=30.0):
                super(NN2LayerClassifier2, self).__init__()
                self.dense_width = int(dense_width)
                self.optimizer = optimizer
                self.learn_rate = learn_rate
                self.regularization = regularization
                self.verbose = verbose
                self.epochs = int(epochs)
                self.batch_size = int(batch_size)
                self.shuffle = shuffle
                self.class_weight = class_weight

                self._model = None
                self.input_dim = None

    def fit(self, X, y):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        if self._model is None or X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            self._model = _create_dense_nn_model(
                self.input_dim,
                self.dense_width,
                self.optimizer,
                self.learn_rate,
                self.regularization,
                self.verbose,
            )

        self._model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
            verbose=self.verbose,
            class_weight=_set_class_weight(self.class_weight),
        )

    def predict_proba(self, X):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        pos_pred = self._model.predict(X, verbose=self.verbose)
        neg_pred = 1 - pos_pred
        return np.hstack([neg_pred, pos_pred])

    def full_hyper_space(self):
        from hyperopt import hp
        hyper_choices = {
            "mdl_optimizer": ["sgd", "rmsprop", "adagrad", "adam", "nadam"]
        }
        hyper_space = {
            "mdl_dense_width": hp.quniform("mdl_dense_width", 2, 100, 1),
            "mdl_epochs": hp.quniform("mdl_epochs", 20, 60, 1),
            "mdl_optimizer": hp.choice("mdl_optimizer", hyper_choices["mdl_optimizer"]),
            "mdl_learn_rate": hp.lognormal("mdl_learn_rate", 0, 1),
            "mdl_class_weight": hp.lognormal("mdl_class_weight", 3, 1),
            "mdl_regularization": hp.lognormal("mdl_regularization", -4, 2),
        }
        return hyper_space, hyper_choices


def _create_dense_nn_model(vector_size=40,
                           dense_width=128,
                           optimizer='rmsprop',
                           learn_rate_mult=1.0,
                           regularization=0.01,
                           verbose=1):
    """Return callable model.

    Returns
    -------
    callable:
        A function that return the Keras Sklearn model when
        called.

    """

    model = Sequential()

    model.add(
        Dense(
            dense_width,
            input_dim=vector_size,
            kernel_regularizer=regularizers.l2(regularization),
            activity_regularizer=regularizers.l1(regularization),
            activation='relu',
        ))

    # add Dense layer with relu activation
    model.add(
        Dense(
            dense_width,
            kernel_regularizer=regularizers.l2(regularization),
            activity_regularizer=regularizers.l1(regularization),
            activation='relu',
        ))

    # add Dense layer
    model.add(Dense(1, activation='sigmoid'))

    optimizer_fn = _get_optimizer(optimizer, learn_rate_mult)

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer_fn,
        metrics=['acc'])

    if verbose >= 1:
        model.summary()

    return model
