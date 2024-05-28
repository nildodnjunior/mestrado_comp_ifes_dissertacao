import numpy as np
import pandas as pd
from small_text import LeastConfidence, PoolBasedActiveLearner, random_initialization_balanced, TextDataset
from small_text.integrations.transformers.classifiers.setfit import SetFitModelArguments
from small_text.integrations.transformers.classifiers.factories import SetFitClassificationFactory
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from synergy_dataset import Dataset, iter_datasets
from imblearn.over_sampling import SMOTEN
import torch
from transformers import AutoModel
from sklearn.model_selection import train_test_split
import gc

dataset = Dataset('Radjenovic_2013')
dataset = dataset.to_frame()
dataset = dataset.dropna()
X = np.array(dataset['abstract'])
y = np.array(dataset['label_included'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

sampler = SMOTEN(random_state=42)
X_train_os, y_train_os = sampler.fit_resample(X_train.reshape(-1, 1), y_train)

num_classes = 2
target_labels = [0, 1]
train = TextDataset.from_arrays(X_train_os, y_train_os, target_labels=target_labels)
# train = TextDataset.from_arrays(X_train, y_train, target_labels=target_labels)
test = TextDataset.from_arrays(X_test, y_test, target_labels=target_labels)

sentence_transformer_model_name = 'meta-llama/Meta-Llama-3-8B'
setfit_model_args = SetFitModelArguments(sentence_transformer_model_name)
clf_factory = SetFitClassificationFactory(setfit_model_args, num_classes, classification_kwargs={'device': 'cuda', 'max_seq_len': 256, 'mini_batch_size': 32})

query_strategy = LeastConfidence()
setfit_train_kwargs = {'show_progress_bar': False}
active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train, fit_kwargs={'setfit_train_kwargs': setfit_train_kwargs})

indices_initial = random_initialization_balanced(train.y, n_samples=2)
active_learner.initialize_data(indices_initial, train.y[indices_initial])

num_queries = 10
results_setfit = []

for i in range(num_queries):
    indices_queried = active_learner.query(num_samples=2)
    y = train.y[indices_queried]
    active_learner.update(y)
    gc.collect()
    torch.cuda.empty_cache()

    y_pred_train = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    print(f'\nIteration {i+1}')
    print('Train accuracy: {:.2f}'.format(accuracy_score(train.y, y_pred_train)))
    print('Train precision: {:.2f}'.format(precision_score(train.y, y_pred_train, zero_division=0)))
    print('Train recall: {:.2f}'.format(recall_score(train.y, y_pred_train, zero_division=0)))
    print('Train F1 score: {:.2f}'.format(f1_score(train.y, y_pred_train)))
    print('\n')
    print('Test accuracy: {:.2f}'.format(accuracy_score(test.y, y_pred_test)))
    print('Test precision: {:.2f}'.format(precision_score(test.y, y_pred_test, zero_division=0)))
    print('Test recall: {:.2f}'.format(recall_score(test.y, y_pred_test, zero_division=0)))
    print('Test F1 score: {:.2f}'.format(f1_score(test.y, y_pred_test)))
    
    results_setfit.append([accuracy_score(train.y, y_pred_train), 
                           accuracy_score(test.y, y_pred_test), 
                           f1_score(train.y, y_pred_train), 
                           f1_score(test.y, y_pred_test), 
                           recall_score(train.y, y_pred_train), 
                           recall_score(test.y, y_pred_test)])
    
#Plotando resultados
import matplotlib.pyplot as plt

acc_treino = []
acc_teste = []
f1_treino = []
f1_teste = []
recall_treino = []
recall_teste = []

def plota_resultados(treino, teste, metrica):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.plot(np.arange(1, len(treino)+1), treino, label=f'{metrica} treino')
    ax.plot(np.arange(1, len(treino)+1), teste, label=f'{metrica} teste')
    ax.legend(loc='lower right')
    plt.xticks(np.arange(1, len(treino)+1))
    plt.ylim((0.0, 1.0))
    plt.ylabel(metrica)
    plt.xlabel('Número de iterações')
    plt.title(f'{metrica} treino x teste')
    plt.savefig(f'{metrica}.png')


for result in results_setfit:
    acc_treino.append(result[0])
    acc_teste.append(result[1])
    f1_treino.append(result[2])
    f1_teste.append(result[3])
    recall_treino.append(result[4])
    recall_teste.append(result[5])

plota_resultados(acc_treino, acc_teste, 'Accuracy_score')
plota_resultados(f1_treino, f1_teste, 'F1_score')
plota_resultados(recall_treino, recall_teste, 'Recall')