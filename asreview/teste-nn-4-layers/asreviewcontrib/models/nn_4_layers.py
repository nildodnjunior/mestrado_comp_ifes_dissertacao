from asreview.models.classifiers.base import BaseTrainClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy

from tensorflow import keras




class NN4Layers(BaseTrainClassifier): 
    """Rede neural 4 camadas ocultas
    """

    name = "nn-4-layers"

    def __init__(self):

        super().__init__()        
        self._model = None
        self.input_dim = None

    def fit(self, X, y):
        if scipy.sparse.issparse(X):
            X = X.toarray()

        if self._model is None or X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
            self._model = keras.models.Sequential()
            self._model.add(keras.layers.InputLayer(input_shape=(self.input_dim)))
            self._model.add(keras.layers.Dense(128, activation='relu'))
            self._model.add(keras.layers.Dense(128, activation='relu'))
            self._model.add(keras.layers.Dense(128, activation='relu'))
            self._model.add(keras.layers.Dense(1))
            self._model.add(keras.layers.Activation('softmax'))
            self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        
        self._model.fit(X, y, verbose=0)

    def predict_proba(self, X):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        pos_pred = self._model.predict(X, verbose=0)
        neg_pred = 1 - pos_pred
        return np.hstack([neg_pred, pos_pred])
        
        # def predict(self, X):
        #     predictions = self.predict_proba(X)
        #     ypred = self.labels[np.argmax(predictions, axis=1)]
        #     return ypred
