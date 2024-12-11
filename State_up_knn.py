from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef
)
import pandas as pd
import numpy as np
def knn_classification_no_feature_selection(data_path, target_column, n_neighbors=5, random_state=42):
    """
    KNN sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        n_neighbors (int): KNN'deki komşu sayısı. Varsayılan 5.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Veri setini yükle
    df = pd.read_excel(data_path)

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # KNN modelini tanımla
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Stratified 5-katlı çapraz doğrulama
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Tahminleri çapraz doğrulama ile al
    y_pred = cross_val_predict(knn, X, y, cv=cv, method='predict')
    y_proba = cross_val_predict(knn, X, y, cv=cv, method='predict_proba')

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    # Çok sınıflı sınıflandırma için AUC hesapla
    if len(np.unique(y)) > 2:
        auc = roc_auc_score(pd.get_dummies(y).values, y_proba, multi_class='ovr', average='macro')
    else:
        auc = roc_auc_score(y, y_proba[:, 1])

    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y, y_pred)

    # Metrikleri bir sözlükte topla
    metrics = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc,
        'MCC': mcc
    }

    return metrics



def knn_classification_with_feature_selection(data_path, target_column, n_neighbors=5, random_state=42, k=10):
    """
    KNN sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        n_neighbors (int): KNN'deki komşu sayısı. Varsayılan 5.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
        selected_features (list): Seçilen özelliklerin isimleri.
    """
    # Veri setini yükle
    df = pd.read_excel(data_path)

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # Seçilen özellik isimlerini al
    selected_features = list(X.columns[selector.get_support()])

    # KNN modelini tanımla
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Stratified 5-katlı çapraz doğrulama
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Tahminleri çapraz doğrulama ile al
    y_pred = cross_val_predict(knn, X_selected, y, cv=cv, method='predict')
    y_proba = cross_val_predict(knn, X_selected, y, cv=cv, method='predict_proba')

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    # Çok sınıflı sınıflandırma için AUC hesapla
    if len(np.unique(y)) > 2:
        auc = roc_auc_score(pd.get_dummies(y).values, y_proba, multi_class='ovr', average='macro')
    else:
        auc = roc_auc_score(y, y_proba[:, 1])

    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y, y_pred)

    # Metrikleri bir sözlükte topla
    metrics = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc,
        'MCC': mcc
    }

    return metrics, selected_features


metrics, features = knn_classification_with_feature_selection('State_UP.xlsx', 'GrainYield', n_neighbors=5, k=10)
print(metrics)
print("Selected Features:", features)



'''
feature selection olmadan
{'Accuracy': 0.6561014263074485, 'F1 Score': 0.6325793754660793, 'Precision': 0.6378901648031602, 'Recall': 0.6286736379486354, 'AUC': 0.8135814526308512, 'MCC': 0.444475336155501}
'''

'''
feature selection varken
{'Accuracy': 0.5768621236133122, 'F1 Score': 0.545594495956412, 'Precision': 0.5521984518120271, 'Recall': 0.541256194612354, 'AUC': 0.7473030550589693, 'MCC': 0.3128280695646053}
Selected Features: ['Latitude', 'BasalMOP', 'ThirdIrrigationDay', 'SowingMonth', 'SowingWeekNum', 'HerbicideYear', 'HerbicideMonth', 'HerbicideWeekNum', 'DaysFromZerotoSowing', 'DaysFromHerbicideToHarvest']
'''
