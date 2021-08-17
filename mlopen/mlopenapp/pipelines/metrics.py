from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from . import text_preprocessing as tpp


def plot_confusion(cm):
  plt.figure(figsize = (5,5))
  sn.heatmap(cm, annot=True, cmap="Blues", fmt='.0f')
  plt.xlabel("Prediction")
  plt.ylabel("True value")
  plt.title("Confusion Matrix")
  plt.show()
  return sn


def get_model_metrics(tfidf, model, X_test, y_test, processed=False):
    """
    Calculate accuracy/precision and confusion matrix of a model
    """
    try:
        if not processed:
            for i, x in enumerate(X_test):
                corp = tpp.process_text(x)
                X_test[i] = corp
        X_test = tfidf.transform(X_test)
    except Exception:
        print("Error during transformation")
    try:
        y_pred = model.predict(X_test)
        print("PREDICTIONS ARE:")
        print(y_pred)
        print("LR Model Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred)))
        plot_confusion(confusion_matrix(y_test, y_pred))
        return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)
    except Exception:
        print("Error during prediction")
        return None
