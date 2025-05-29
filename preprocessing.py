import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def preprocess(df):
    """
    Normalizes X,Y coordinates for all samples in the dataset:
    - Centers wrist (landmark 0) at (0, 0).
    - Scales by middle finger tip (landmark 12) distance.
    - Preserves Z coordinates if they exist.

    Args:
        df: DataFrame with columns like x1,y1,z1,...,x21,y21,z21.

    Returns:
        DataFrame with normalized X,Y and original Z.
    """

    x_col = [col for col in df.columns if col.startswith('x')]
    y_col = [col for col in df.columns if col.startswith('y')]
    z_col = [col for col in df.columns if col.startswith('z')]
    x_data = np.array(df[x_col])
    y_data = np.array(df[y_col])
    z_data = np.array(df[z_col])
    x_wirst = x_data[:,0].reshape(-1,1)
    y_wirst = y_data[:,0].reshape(-1,1)
    x_centered = x_data-x_wirst
    y_centered = y_data - y_wirst

    dis = np.sqrt(x_centered[:,12]**2+y_centered[:,12]**2).reshape(-1,1)
    x_norm = x_centered / dis
    y_norm = y_centered /dis

    normalized_data = {}
    for i in range(21):
        normalized_data[f"x{i+1}"] = x_norm[:, i]
        normalized_data[f"y{i+1}"] = y_norm[:, i]
        normalized_data[f"z{i+1}"] = z_data[:, i]

    normalized = pd.DataFrame(normalized_data)
    labels = df["label"]
    features = normalized
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=.20,train_size=.8,random_state=100)
    return features_train, features_test, labels_train, labels_test
