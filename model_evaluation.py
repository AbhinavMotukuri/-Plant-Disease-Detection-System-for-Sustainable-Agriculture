from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# example predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))
print(confusion_matrix(y_test, y_pred_classes))
