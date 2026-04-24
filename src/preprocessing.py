import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def feature_target_split(data, target_col= "Label"):
    data = pd.DataFrame(data)
    features = data.drop(target_col, axis= 1)
    target = data[target_col]
    return features, target

def splitting(features, target, state):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=state)
    return (X_train, X_test, y_train, y_test)

def scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit(X_test)
    return X_train_scaled, X_test_scaled
    




