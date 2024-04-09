from asreview.models.classifiers.base import BaseTrainClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from tensorflow import keras




class NN4Layers(BaseTrainClassifier, BaseEstimator, ClassifierMixin):
    """Rede neural 4 camadas ocultas
    """

    name = "nn-4-layers"

    def __init__(self):

        super().__init__()        
#        self._model = None

        def fit(self, X, y):
            self.labels, ids = np.unique(y, return_inverse=True)
#            yhot = keras.utils.to_categorical(ids)
            self._model = keras.models.Sequential()
            self._model.add(keras.layers.InputLayer(input_shape=(X.shape[1])))
            self._model.add(keras.layers.Dense(activation='relu'))
            self._model.add(keras.layers.Dense(activation='relu'))
            self._model.add(keras.layers.Dense(activation='relu'))
            self._model.add(keras.layers.Dense(activation='relu'))
            self._model.add(keras.layers.Activation('softmax'))
            self._model.compile()
            self._model.fit(X, y)

        def predict_proba(self, X):
            """Get the inclusion probability for each sample."""
            return self._model.predict(X)
        
        def predict(self, X):
            predictions = self.predict_proba(X)
            ypred = self.labels[np.argmax(predictions, axis=1)]
            return ypred
