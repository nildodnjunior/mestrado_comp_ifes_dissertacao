from sklearn.naive_bayes import MultinomialNB

from asreview.models.classifiers.base import BaseTrainClassifier

from setfit import SetFitModel, sample_dataset, TrainingArguments, Trainer
from datasets import load_dataset, Dataset

class SetFitClassifier(BaseTrainClassifier):

    name = "setfit"

    def __init__(self):

        super().__init__()
        self._model = SetFitModel.from_pretrained("distilbert/distilbert-base-uncased")
        self._model.labels = [0, 1]
    
    def to_dataset(X, y):
        for i in range(len(X)):
            yield {'abstract': X[i], 'label_included': y[i]}

    def fit(self, X, y):
        self.dataset = Dataset.from_dict({'abstract': X, 'label_included': y})
        args = TrainingArguments(num_epochs=10, max_length=256, batch_size=10)
        trainer = Trainer(model=self._model, args=args, train_dataset=self.dataset, metric='recall', column_mapping={'abstract': 'text', 'label_included':'label'})
        trainer.train()

    def predict_proba(self, X):
        return self._model.predict_proba(X)
