import matplotlib.pyplot as plt
import numpy as np

results = [[0.9, 0.8, 0.5, 0.3], [0.92, 0.84, 0.55, 0.42], [0.88, 0.82, 0.57, 0.49], [0.93, 0.89, 0.61, 0.55]]

acc_treino = []
acc_teste = []
f1_treino = []
f1_teste = []

def plota_curvas(treino, teste, metrica):
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

plota_curvas(acc_treino, acc_teste, 'Acurácia')
plota_curvas(f1_treino, f1_teste, 'F1 score')

# fig = plt.figure(figsize=(12, 8))
# ax = plt.axes()

# ax.plot(np.arange(4), acc_treino, label=f'Acurácia treino')
# ax.plot(np.arange(4), acc_teste, label=f'Acurácia teste')
# ax.legend(loc='lower right')

# plt.xticks(np.arange(4))
# plt.ylim((0.25, 1.0))

# plt.ylabel('Acurácia')
# plt.xlabel('Número de iterações')

# plt.title('Acurácia treino x teste')

# plt.savefig("acuracia.png")