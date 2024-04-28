import numpy as np
import pandas as pd
from small_text import TransformersDataset, TransformerModelArguments, LeastConfidence, TransformerBasedClassificationFactory as TransformerFactory, PoolBasedActiveLearner, random_initialization_balanced as init
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from synergy_dataset import Dataset, iter_datasets
from imblearn.over_sampling import SMOTEN
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# dataset = Dataset('Leenaars_2020')
# dataset = Dataset('Hall_2012')
# dataset = dataset.to_frame()
dataset = pd.read_csv('Leenaars_2020.csv')
dataset = dataset.dropna()

#Divisao treino/teste
X = np.array(dataset['abstract'])
y = np.array(dataset['label_included'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Oversampling
sampler = SMOTEN(random_state=42)
X_train_os, y_train_os = sampler.fit_resample(X_train.reshape(-1, 1), y_train)

#Criação do active learner
transformer_model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(transformer_model)
train = TransformersDataset.from_arrays(X_train_os.flatten(), y_train_os, tokenizer, target_labels=np.array([0, 1]), max_length=256)
test = TransformersDataset.from_arrays(X_test, y_test, tokenizer, target_labels=np.array([0, 1]), max_length=256) #max_length > 256 geralmente ultrapassa a RAM da GPU (8GB)

#Estatísticas do dataset
print('Treino:')
print(0, len(train.y[train.y==0])/len(train.y))
print(1, 1-len(train.y[train.y==0])/len(train.y))
print(f'Número de amostras sem oversampling: {len(X_train)}')
print(f'Número de amostras com oversampling: {len(train)}')

print('\nTeste:')
print(0, len(test.y[test.y==0])/len(test.y))
print(1, 1-len(test.y[test.y==0])/len(test.y))
print(f'Número de amostras: {len(X_test)}')

num_classes = 2
model_args = TransformerModelArguments(transformer_model)
clf_factory = TransformerFactory(model_args, num_classes, kwargs={'device': 'cuda'})
# clf_factory = TransformerFactory(model_args, num_classes)
query_strategy = LeastConfidence()
active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
indices_initial = init(train.y, n_samples=10)
active_learner.initialize_data(indices_initial, train.y[indices_initial])

#Loop do active learner
num_queries = 10
results = []
indices_labeled = []
print("Iniciando active learner...")
for i in range(num_queries):
    indices_queried = active_learner.query(num_samples=20)
    y = train.y[indices_queried]
    active_learner.update(y)

    indices_labeled = np.concatenate([indices_queried, indices_labeled])

    y_pred_train = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)
    print(f'\nIteration {i+1} ({len(indices_labeled)} samples)')
    print('Train accuracy: {:.2f}'.format(accuracy_score(train.y, y_pred_train)))
    print('Train precision: {:.2f}'.format(precision_score(train.y, y_pred_train, zero_division=np.nan)))
    print('Train recall: {:.2f}'.format(recall_score(train.y, y_pred_train, zero_division=np.nan)))
    print('Train F1 score: {:.2f}'.format(f1_score(train.y, y_pred_train)))
    print('\n')
    print('Test accuracy: {:.2f}'.format(accuracy_score(test.y, y_pred_test)))
    print('Test precision: {:.2f}'.format(precision_score(test.y, y_pred_test, zero_division=np.nan)))
    print('Test recall: {:.2f}'.format(recall_score(test.y, y_pred_test, zero_division=np.nan)))
    print('Test F1 score: {:.2f}'.format(f1_score(test.y, y_pred_test)))

    results.append([accuracy_score(train.y, y_pred_train), accuracy_score(test.y, y_pred_test), f1_score(train.y, y_pred_train), f1_score(test.y, y_pred_test)])

#Plotando resultados
acc_treino = []
acc_teste = []
f1_treino = []
f1_teste = []

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


for result in results:
    acc_treino.append(result[0])
    acc_teste.append(result[1])
    f1_treino.append(result[2])
    f1_teste.append(result[3])

plota_resultados(acc_treino, acc_teste, 'Accuracy_score')
plota_resultados(f1_treino, f1_teste, 'F1_score')