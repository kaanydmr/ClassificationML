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

def feature_selection_with_cross_validation(train_path, test_path, target_column, random_state=42, average='macro'):
    """
    Karar ağacı ile özellik seçimi yapan fonksiyon (RFECV tabanlı çapraz doğrulama ile).

    Args:
        train_path (str): Eğitim verisinin dosya yolu.
        test_path (str): Test verisinin dosya yolu.
        target_column (str): Hedef sütunun adı.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        average (str): Çok sınıflı sınıflama için ortalama türü. ('micro', 'macro', 'weighted', None)

    Returns:
        selected_features (list): Seçilen özelliklerin isimleri.
        accuracy (float): Model doğruluğu (seçilen özelliklerle test setinde).
        auc (float): AUC (Area Under Curve).
        f1 (float): F1 skoru.
        precision (float): Precision skoru.
        recall (float): Recall skoru.
        mcc (float): MCC (Matthews Correlation Coefficient).
        cv_scores (list): Çapraz doğrulama sırasında doğruluk skorları.
    """
    # Eğitim ve test verilerini yükle
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    # Hedef ve özellikler
    X_train = train_df.drop(columns=[target_column])  # Eğitim özellikleri
    y_train = train_df[target_column]  # Eğitim hedefi
    X_test = test_df.drop(columns=[target_column])  # Test özellikleri
    y_test = test_df[target_column]  # Test hedefi

    # Karar ağacı modelini tanımla
    model = DecisionTreeClassifier(random_state=random_state)

    # Çapraz doğrulama stratejisi
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # RFECV ile özellik seçimi
    rfecv = RFECV(estimator=model, step=1, cv=cv, scoring="accuracy", n_jobs=-1)
    rfecv.fit(X_train, y_train)

    # Seçilen özellikleri al
    selected_features = list(X_train.columns[rfecv.support_])

    # Yeni veri setini seçilen özelliklerle oluştur
    X_train_selected = rfecv.transform(X_train)
    X_test_selected = rfecv.transform(X_test)

    # Seçilen özelliklerle modeli yeniden eğit ve test seti üzerinde tahmin yap
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    # Model metriklerini hesapla
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_selected), multi_class='ovr')  # Çok sınıflı AUC
    mcc = matthews_corrcoef(y_test, y_pred)

    # Çapraz doğrulama doğruluk skorlarını al
    cv_scores = rfecv.cv_results_['mean_test_score']

    return selected_features, accuracy, auc, f1, precision, recall, mcc, cv_scores



train_path = "LandType_Mediumland.xlsx"
test_path = "test_data.xlsx"



selected_features, accuracy, auc, f1, precision, recall, mcc, cv_scores = feature_selection_with_cross_validation(
    train_path=train_path,
    test_path=test_path,
    target_column="GrainYield",  # Hedef sütunun adı
    random_state=42          # Rastgelelik sabiti
)

# Sonuçları yazdır
print("Seçilen Özellikler:", selected_features)
print(f"Model Doğruluğu (Test Seti): {accuracy:.2f}")
print(f"AUC (Area Under Curve): {auc:.2f}")
print(f"F1 Skoru: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"MCC (Matthews Correlation Coefficient): {mcc:.2f}")
print("Çapraz Doğrulama Skorları (Eğitim Seti):", cv_scores)

'''
Model Doğruluğu (Test Seti): 0.63
AUC (Area Under Curve): 0.72
F1 Skoru: 0.62
Precision: 0.63
Recall: 0.62
MCC (Matthews Correlation Coefficient): 0.42
'''