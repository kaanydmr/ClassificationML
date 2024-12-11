from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
import numpy as np

def naive_bayes_classification(data_path, target_column, random_state=42):
    """
    Naive Bayes sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Veri setini yükle
    df = pd.read_excel(data_path)

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Naive Bayes modelini tanımla
    nb = GaussianNB()

    # Stratified 5-katlı çapraz doğrulama
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Tahminleri çapraz doğrulama ile al
    y_pred = cross_val_predict(nb, X, y, cv=cv, method='predict')
    y_proba = cross_val_predict(nb, X, y, cv=cv, method='predict_proba')

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





def naive_bayes_classification_with_feature_selection(data_path, target_column, test_data_path=None, random_state=42):
    """
    Naive Bayes sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.
    Özellik seçimi eklenmiştir. Ayrıca test verisi de parametre olarak alınır.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    df = pd.read_excel(data_path)

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Naive Bayes modelini tanımla
    nb = GaussianNB()

    # Stratified 5-katlı çapraz doğrulama
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Karşılıklı bilgi tabanlı özellik seçimi
    selector = SelectKBest(mutual_info_classif, k=10)  # En iyi 5 özelliği seç
    X_selected = selector.fit_transform(X, y)

    selected_columns = X.columns[selector.get_support()]
    print("Seçilen Özellikler:", selected_columns.tolist())

    # Test veri seti varsa, test verisini yükle
    if test_data_path:
        test_df = pd.read_excel(test_data_path)
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Test verisini seçilen özelliklerle daralt
        X_test_selected = selector.transform(X_test)

        # Modeli test verisi üzerinde değerlendir
        nb.fit(X_selected, y)
        y_pred = nb.predict(X_test_selected)
        y_proba = nb.predict_proba(X_test_selected)

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

    else:
        # Test verisi yoksa, çapraz doğrulama ile tahminler al
        y_pred = cross_val_predict(nb, X_selected, y, cv=cv, method='predict')
        y_proba = cross_val_predict(nb, X_selected, y, cv=cv, method='predict_proba')

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




metrics = naive_bayes_classification_with_feature_selection('State_UP.xlsx', 'GrainYield',test_data_path='test_data.xlsx')
print(metrics)

'''
no feature selection
{'Accuracy': 0.3629160063391442, 'F1 Score': 0.35661412022633804, 'Precision': 0.5937891816177888, 'Recall': 0.4676707137964411, 'AUC': 0.7382237783660491, 'MCC': 0.24783614105961316}
'''

'''
Seçilen Özellikler: ['Latitude', 'BasalNPK', 'BasalMOP', 'ThirdIrrigationDay', 'SowingMonth', 'SowingWeekNum', 'HerbicideMonth', 'HerbicideWeekNum', 'DaysFromZerotoSowing', 'DaysFromHerbicideToHarvest']
{'Accuracy': 0.5127175368139224, 'F1 Score': 0.39627179856432176, 'Precision': 0.3767326080790892, 'Recall': 0.4636176268282742, 'AUC': 0.6351563804106194, 'MCC': 0.21427690354563592}
'''