from sklearn.metrics import accuracy_score,confusion_matrix
def predictions(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    con_mat = confusion_matrix(y_test, y_pred)
    print(acc)
    print(f"\n {con_mat} ")