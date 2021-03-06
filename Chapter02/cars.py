from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
input_file = 'car.data.txt'

x = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        x.append(data)
x = np.array(x)


label_encoder = []
x_encoded = np.empty(x.shape)
for i, item in enumerate(x[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    x_encoded[:, i] = label_encoder[-1].fit_transform(x[:, i])
x = x_encoded[:, :-1].astype(int)
y = x_encoded[:, -1].astype(int)

params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}

classifier = RandomForestClassifier(**params)
classifier.fit(x, y)

from sklearn import model_selection

accuracy = model_selection.cross_val_score(classifier, x, y, scoring='accuracy', cv=3)

print("Accuracy of the classifier:", str(round(100*accuracy.mean(), 2)), "%")
input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low']
input_data_encoded = [-1] * len(input_data)

for i, item in enumerate(input_data):
    input_data_encoded[i] = int(label_encoder[i].transform(input_data[i]))

input_data_encoded = np.array(input_data_encoded)

out_class = classifier.predict(input_data_encoded)
print("Output class:", label_encoder[-1].inverse_transform(out_class)[0])