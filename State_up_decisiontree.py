import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

def decision_tree_no_feature_selection(train_path, target_column, test_data_path=None, random_state=42, average='macro'):
    """
    Karar ağacı sınıflandırması ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.
    Özellik seçimi yapılmaz.

    Args:
        train_path (str): Eğitim verisinin dosya yolu.
        target_column (str): Hedef sütunun adı.
        test_data_path (str): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        average (str): Çok sınıflı sınıflama için ortalama türü. ('micro', 'macro', 'weighted', None)

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim verisini yükle
    train_df = pd.read_excel(train_path)

    # Özellikler ve hedef değişkeni ayır
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Karar ağacı modelini tanımla
    model = DecisionTreeClassifier(random_state=random_state)

    # Çapraz doğrulama stratejisi
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama ile doğrulama
    y_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
    y_proba = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')

    # Performans metriklerini hesapla
    accuracy = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred, average=average)
    precision = precision_score(y_train, y_pred, average=average)
    recall = recall_score(y_train, y_pred, average=average)

    # Çok sınıflı sınıflandırma için AUC hesapla
    auc = roc_auc_score(y_train, y_proba, multi_class='ovr', average=average)

    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_train, y_pred)

    # Metrikleri bir sözlükte topla
    metrics = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc,
        'MCC': mcc
    }

    # Test verisi varsa, test verisini yükle
    if test_data_path:
        test_df = pd.read_excel(test_data_path)
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Modeli test verisi üzerinde değerlendir
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)

        # Performans metriklerini test verisi için hesapla
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average=average)
        precision_test = precision_score(y_test, y_pred_test, average=average)
        recall_test = recall_score(y_test, y_pred_test, average=average)
        auc_test = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average=average)
        mcc_test = matthews_corrcoef(y_test, y_pred_test)

        # Test metriklerini ekle
        metrics['Test Accuracy'] = accuracy_test
        metrics['Test F1 Score'] = f1_test
        metrics['Test Precision'] = precision_test
        metrics['Test Recall'] = recall_test
        metrics['Test AUC'] = auc_test
        metrics['Test MCC'] = mcc_test

    return metrics



def decision_tree_with_feature_selection(train_path, target_column, test_data_path=None, random_state=42, average='macro'):
    """
    Karar ağacı ile özellik seçimi yapan fonksiyon (RFECV tabanlı çapraz doğrulama ile).

    Args:
        train_path (str): Eğitim verisinin dosya yolu.
        target_column (str): Hedef sütunun adı.
        test_data_path (str): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        average (str): Çok sınıflı sınıflama için ortalama türü. ('micro', 'macro', 'weighted', None)

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
        selected_features (list): Seçilen özelliklerin isimleri.
    """
    # Eğitim verisini yükle
    train_df = pd.read_excel(train_path)

    # Özellikler ve hedef değişkeni ayır
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Karar ağacı modelini tanımla
    model = DecisionTreeClassifier(random_state=random_state)

    # Çapraz doğrulama stratejisi
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # RFECV ile özellik seçimi
    rfecv = RFECV(estimator=model, step=1, cv=cv, scoring="accuracy", n_jobs=-1)
    rfecv.fit(X_train, y_train)

    # Seçilen özellikleri al
    selected_features = list(X_train.columns[rfecv.support_])
    print("Seçilen Özellikler:", selected_features)

    # Test veri seti varsa, test verisini yükle
    if test_data_path:
        test_df = pd.read_excel(test_data_path)
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Test verisini seçilen özelliklerle daralt
        X_test_selected = rfecv.transform(X_test)

        # Modeli test verisi üzerinde değerlendir
        model.fit(rfecv.transform(X_train), y_train)
        y_pred = model.predict(X_test_selected)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_selected), multi_class='ovr')  # Çok sınıflı AUC
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


# Fonksiyonu çağırmak için örnek:
metrics = decision_tree_with_feature_selection(
    train_path="State_UP.xlsx",
    target_column="GrainYield",
    test_data_path="test_data.xlsx",  
    random_state=42
)

# Sonuçları yazdır

print("Performans Metrikleri:", metrics)


'''
feature selection:
Performans Metrikleri: {'Accuracy': 0.46720214190093706, 'F1 Score': 0.457512990205679, 'Precision': 0.4596682007568975, 'Recall': 0.4588035949192702, 'AUC': 0.5903691389255673, 'MCC': 0.17351056216785055}
seçilen:  ['Latitude', 'BasalMOP', 'IrrigationNumber', 'ThirdIrrigationDay', 'SowingYear', 'SowingWeekNum', 'HerbicideDay', 'DaysFromSowingToHerbicide', 'DaysFromHerbicideToHarvest']
'''

'''
feature selection olmadan 
Performans Metrikleri: {'Accuracy': 0.6434231378763867, 'F1 Score': 0.6298654293866593, 'Precision': 0.6291616234572516, 'Recall': 0.6311172168069371, 'AUC': 0.7198383623346216, 'MCC': 0.43387309094673016, 'Test Accuracy': 0.4444444444444444, 'Test F1 Score': 0.41562392810758597, 'Test Precision': 0.4334864870067303, 'Test Recall': 0.40600649943403805, 'Test AUC': 0.5537114130423012, 'Test MCC': 0.11046449636435314}

'''
