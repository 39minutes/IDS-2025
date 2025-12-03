import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np

def load_and_preprocess(data_path, subset_size=0.1):
    df = pd.read_csv(data_path, low_memory=False)
    df = df.sample(frac=subset_size)  # Подмножество для скорости

    # Очистка
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Категориальные
    cat_cols = ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP']  # Удалим IP/ID для обобщения
    df.drop(cat_cols, axis=1, inplace=True, errors='ignore')
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Label':
            df[col] = le.fit_transform(df[col])

    # Лейблы (мультикласс)
    df['Label'] = le.fit_transform(df['Label'])
    num_classes = len(le.classes_)

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Нормализация
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Баланс (SMOTE)
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    # Для LSTM: reshape в последовательности (假设 1 timestep per sample)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, num_classes
