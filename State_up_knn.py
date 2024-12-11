from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef
)
import pandas as pd
import numpy as np
def knn_classification_no_feature_selection(data_path, target_column, test_data_path=None, n_neighbors=5, random_state=42):
    """
    KNN sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.
    Test verisi parametre olarak alınır.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        n_neighbors (int): KNN'deki komşu sayısı. Varsayılan 5.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    df = pd.read_excel(data_path)

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # KNN modelini tanımla
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Test veri seti varsa, test verisini yükle
    if test_data_path:
        test_df = pd.read_excel(test_data_path)
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Modeli test verisi üzerinde değerlendir
        knn.fit(X, y)
        y_pred = knn.predict(X_test)
        y_proba = knn.predict_proba(X_test)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        if len(np.unique(y)) > 2:
            auc = roc_auc_score(pd.get_dummies(y_test).values, y_proba, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_test, y_proba[:, 1])

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_test, y_pred)

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



def knn_classification_with_feature_selection(data_path, target_column, test_data_path=None, n_neighbors=5, random_state=42, k=10):
    """
    KNN sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.
    Test verisi parametre olarak alınır.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        n_neighbors (int): KNN'deki komşu sayısı. Varsayılan 5.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
        selected_features (list): Seçilen özelliklerin isimleri.
    """
    # Eğitim veri setini yükle
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

    # Test veri seti varsa, test verisini yükle
    if test_data_path:
        test_df = pd.read_excel(test_data_path)
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Test verisini seçilen özelliklerle daralt
        X_test_selected = selector.transform(X_test)

        # Modeli test verisi üzerinde değerlendir
        knn.fit(X_selected, y)
        y_pred = knn.predict(X_test_selected)
        y_proba = knn.predict_proba(X_test_selected)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        if len(np.unique(y)) > 2:
            auc = roc_auc_score(pd.get_dummies(y_test).values, y_proba, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(y_test, y_proba[:, 1])

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_test, y_pred)

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


metrics = knn_classification_no_feature_selection('State_UP.xlsx', 'GrainYield', test_data_path='test_data.xlsx', n_neighbors=5)
print(metrics)




'''
feature selection olmadan
{'Accuracy': 0.4966532797858099, 'F1 Score': 0.4680623206986328, 'Precision': 0.46466492977243634, 'Recall': 0.4772326878307818, 'AUC': 0.6343956173246447, 'MCC': 0.20397734371261295}
'''

'''
feature selection varken
{'Accuracy': 0.48728246318607765, 'F1 Score': 0.41448753373860797, 'Precision': 0.4312889099242008, 'Recall': 0.4248497758356451, 'AUC': 0.587630896063969, 'MCC': 0.14838130288795473}
Selected Features: ['Latitude', 'BasalMOP', 'ThirdIrrigationDay', 'SowingMonth', 'SowingWeekNum', 'HerbicideYear', 'HerbicideMonth', 'HerbicideWeekNum', 'DaysFromZerotoSowing', 'DaysFromHerbicideToHarvest']
'''
