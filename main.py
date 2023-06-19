import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_estimator(in_estimator_type, in_estimator_param):
    if in_estimator_type == 'tree':
        return DecisionTreeClassifier(max_depth=in_estimator_param)
    elif in_estimator_type == 'svm':
        return SVC(kernel='linear', C=in_estimator_param)


class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, x, y, in_estimator_param, in_estimator_type):
        n_samples = x.shape[0]
        weights = np.ones(n_samples) / n_samples  # Inicializando os pesos igualmente

        for _ in range(self.n_estimators):
            error = 0
            estimator = get_estimator(in_estimator_type, in_estimator_param)

            estimator.fit(x, y, sample_weight=weights)  # Treinando o estimador com os pesos

            y_pred = estimator.predict(x)

            # Calculando o erro
            for i in range(len(y_pred)):
                if y_pred[i] != y[i]:
                    error = weights[i] + error

            if error >= 0.5:
                print('Erro maior ou igual a 0.5')
                break  # Se o erro for maior ou igual a 0.5, interrompe o loop

            if error == 0:
                error = 1e-5  # Se o erro for 0, adiciona um valor pequeno para evitar divisão por 0

            estimator_weight = 0.5 * np.log((1 / error) - 1)  # Calculando o peso do estimador

            for i in range(len(y_pred)):
                if y_pred[i] != y[i]:
                    weights[i] = weights[i] * np.exp(estimator_weight)
                else:
                    weights[i] = weights[i] * np.exp(-estimator_weight)

            weights /= np.sum(weights)  # Normalizando os pesos

            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)

    def predict(self, x_predict):
        n_samples = x_predict.shape[0]
        y_pred = np.zeros(n_samples)

        for estimator, estimator_weight in zip(self.estimators, self.estimator_weights):
            y_pred += estimator_weight * estimator.predict(x_predict)

        return np.sign(y_pred)


def get_data():
    x, y = np.empty([958, 9], dtype=list), np.empty([958], dtype=int)
    # open file
    with open('tic+tac+toe+endgame/tic-tac-toe.data', 'r') as data_file:
        # read all lines
        lines = data_file.readlines()
        # iterate over lines
        for i in range(len(lines)):
            line_splitted = lines[i].split(',')
            # swaping every x for 1, o for -1 and b for 0
            for j in range(len(line_splitted) - 1):
                if line_splitted[j].strip() == 'x':
                    line_splitted[j] = 1
                elif line_splitted[j].strip() == 'o':
                    line_splitted[j] = -1
                else:
                    line_splitted[j] = 0

            # append line data to data list, but the last element is the class
            x[i] = (line_splitted[:-1])
            # append the last elem to y
            if line_splitted[-1].strip() == 'positive':
                y[i] = 1
            else:
                y[i] = -1

        return x, y


def k_fold_cross_validation(in_features, in_labels, fold_amount, in_estimators, in_tree_depth, in_estimator_type):
    fold_size = len(in_features) // fold_amount
    accuracies = []

    for i in range(fold_amount):
        start = i * fold_size
        end = (i + 1) * fold_size

        features_train = np.concatenate((in_features[:start], in_features[end:]), axis=0)
        labels_train = np.concatenate((in_labels[:start], in_labels[end:]), axis=0)
        features_test = in_features[start:end]
        labels_test = in_labels[start:end]

        boosting = AdaBoost(n_estimators=in_estimators)
        boosting.fit(features_train, labels_train, in_tree_depth, in_estimator_type)
        labels_pred = boosting.predict(features_test)

        in_accuracy = np.sum(labels_pred == labels_test) / len(labels_test)
        accuracies.append(in_accuracy)

    return np.mean(accuracies)


features, labels = get_data()

combined_list = list(zip(features, labels))
np.random.shuffle(combined_list)
features[:], labels[:] = zip(*combined_list)

n_folds = 5

# Testando a acurácia em função do número de estimadores e do tipo
accuracy = []
for estimator_type in ['tree', 'svm']:
    for estimators in range(3, 50):
        tree_depth = 3
        accuracy.append(k_fold_cross_validation(features, labels, n_folds, estimators, tree_depth, estimator_type))
        plt.plot(estimators, accuracy[estimators - 3], 'bo')
    plt.xlabel('Número de estimadores')
    plt.ylabel('Acurácia')
    plt.title('Acurácia x Número de estimadores' + ' (' + estimator_type + ')')
    plt.axis([2, 50, 0.5, 1])
    plt.show()

# Testando a acurácia em função da profundidade da árvore
for tree_depth in range(1, 25):
    estimators = 3
    accuracy.append(k_fold_cross_validation(features, labels, n_folds, estimators, tree_depth, 'tree'))
    plt.plot(tree_depth, accuracy[tree_depth - 1], 'bo')
plt.xlabel('Profundidade da árvore')
plt.ylabel('Acurácia')
plt.title('Acurácia x Profundidade da árvore')
plt.axis([0, 25, 0.5, 1])
plt.show()

# Testando a acurácia em função do número de folds
for n_folds in range(2, 30):
    estimators = 3
    tree_depth = 3
    accuracy.append(k_fold_cross_validation(features, labels, n_folds, estimators, tree_depth, 'tree'))
    plt.plot(n_folds, accuracy[n_folds - 2], 'bo')
plt.xlabel('Número de folds')
plt.ylabel('Acurácia')
plt.title('Acurácia x Número de folds')
plt.axis([1, 30, 0.5, 1])
plt.show()

c_array = [0.1, 0.3, 0.6, 0.9]
# Testando a acurácia em função do C
for c in c_array:
    estimators = 3
    accuracy.append(k_fold_cross_validation(features, labels, n_folds, estimators, c, 'svm'))
    plt.plot(c, accuracy[c_array.index(c)], 'bo')
plt.xlabel('C')
plt.ylabel('Acurácia')
plt.title('Acurácia x C')
plt.axis([0, 1, 0.5, 1])
plt.show()

