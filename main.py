import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, x, y):
        n_samples = x.shape[0]
        weights = np.ones(n_samples) / n_samples  # Inicializando os pesos igualmente

        for _ in range(self.n_estimators):
            error = 0
            estimator = DecisionTreeClassifier(max_depth=3)  # Base Estimator: Decision Tree de profundidade 1
            estimator.fit(x, y, sample_weight=weights)  # Treinando o estimador com os pesos

            y_pred = estimator.predict(x)

            # Calculando o erro
            for i in range(len(y_pred)):
                if y_pred[i] != y[i]:
                    error = weights[i] + error

            if error >= 0.5:
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


features, labels = get_data()

combined_list = list(zip(features, labels))
np.random.shuffle(combined_list)
features[:], labels[:] = zip(*combined_list)

boosting = AdaBoost(n_estimators=3)

features_train = features[:95]
boosting.fit(features_train, labels[:95])

features_test = features[95:]
labels_pred = boosting.predict(features_test)

# TODO: Cross Validation
# Calculando a acurácia
accuracy = 0
for i in range(len(labels_pred)):
    if labels_pred[i] == labels[i + 95]:
        accuracy += 1
print('Acurácia: ', accuracy / len(labels_pred))
