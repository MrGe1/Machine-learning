import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


def plot_classifier(classifier, x, y):
    x_min, x_max = min(x[:, 0]) - 1.0, max(x[:, 0])+1.0
    y_min, y_max = min(x[:, 1]) - 1.0, max(x[:, 1])+1.0
    step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    mesh_output = mesh_output.reshape(x_values.shape)

    plt.figure()
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=80, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    plt.xticks((np.arange(int(min(x[:, 0])-1), int(max(x[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(x[:, 1])-1), int(max(x[:, 1])+1), 1.0)))

    plt.show()

#if __name__=='__main__':
x = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.2], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)
classifier.fit(x, y)
plot_classifier(classifier, x, y)


