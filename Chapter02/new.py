from sklearn.metrics import classification_report
y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]

target_names = ['class-0', 'class-1', 'class-2', 'class-3']
print(classification_report(y_true, y_pred, target_names=target_names))