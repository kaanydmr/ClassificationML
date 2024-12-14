from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

def safe_roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'):
    """
    ROC AUC skorunu güvenli bir şekilde hesaplar. Tek sınıflı durumlarda None döner.

    Args:
        y_true (array-like): Gerçek etiketler.
        y_proba (array-like): Tahmin edilen olasılıklar.
        multi_class (str): Çok sınıflı AUC yöntemi. Varsayılan 'ovr'.
        average (str): Ortalama türü. Varsayılan 'macro'.

    Returns:
        float or None: ROC AUC skoru ya da None (geçersiz durumlarda).
    """
    try:
        if len(np.unique(y_true)) > 2:  # Çok sınıflı durum
            return roc_auc_score(pd.get_dummies(y_true).values, y_proba, multi_class=multi_class, average=average)
        elif len(np.unique(y_true)) == 2:  # İkili sınıflandırma
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Tek sınıflı durum
            print("ROC AUC skoru yalnızca birden fazla sınıf mevcut olduğunda hesaplanabilir.")
            return 0
    except Exception as e:
        print(f"AUC hesaplama hatası: {e}")
        return 0


def decision_tree_classification_with_feature_selection(train_data, target_column, test_data=None, random_state=42, k=10):
    """
    Decision Tree sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.
    Test verisi parametre olarak alınır.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
        selected_features (list): Seçilen özelliklerin isimleri.
    """
    # Eğitim veri setini yükle
    df = train_data

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # Seçilen özellik isimlerini al
    selected_columns = X.columns[selector.get_support()]
    print("Seçilen Özellikler:", selected_columns.tolist())

    # Decision Tree modelini tanımla
    dt = DecisionTreeClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_selected, y):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Modeli eğit
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_val)
        y_proba = dt.predict_proba(X_val)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')

        auc = safe_roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test verisi varsa, model test verisi üzerinde değerlendir

    test_df = test_data
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Modeli test verisi üzerinde değerlendir
    dt.fit(X_selected, y)
    y_pred = dt.predict(X_test_selected)
    y_proba = dt.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def decision_tree_classification_no_feature_selection(train_data, target_column, test_data, random_state=42):
    """
    Decision Tree sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.
    Test verisi parametre olarak alınır.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    df = train_data

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Decision Tree modelini tanımla
    dt = DecisionTreeClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Modeli eğit
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_val)
        y_proba = dt.predict_proba(X_val)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test verisi varsa, model test verisi üzerinde değerlendir

    test_df = test_data
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test setinde performansı değerlendir
    y_pred_test = dt.predict(X_test)
    y_proba_test = dt.predict_proba(X_test)

    # Test seti metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    precision_test = precision_score(y_test, y_pred_test, average='macro')
    recall_test = recall_score(y_test, y_pred_test, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred_test)

    # Test setindeki metrikleri ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics


def knn_classification_with_feature_selection(train_data, target_column, test_data, n_neighbors=5,random_state=42, k=10):
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
    df = train_data

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # Seçilen özellik isimlerini al
    selected_columns = X.columns[selector.get_support()]
    print("Seçilen Özellikler:", selected_columns.tolist())

    # KNN modelini tanımla
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_selected, y):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Modeli eğit
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        y_proba = knn.predict_proba(X_val)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test verisi varsa, model test verisi üzerinde değerlendir

    test_df = test_data
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Modeli test verisi üzerinde değerlendir
    knn.fit(X_selected, y)
    y_pred = knn.predict(X_test_selected)
    y_proba = knn.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def knn_classification_no_feature_selection(train_data, target_column, test_data=None, n_neighbors=5, random_state=42):
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
    df = train_data

    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # KNN modelini tanımla
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Modeli eğit
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        y_proba = knn.predict_proba(X_val)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')

        auc = safe_roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test verisi varsa, model test verisi üzerinde değerlendir

    test_df = test_data
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test setinde performansı değerlendir
    y_pred_test = knn.predict(X_test)
    y_proba_test = knn.predict_proba(X_test)

    # Test seti metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    precision_test = precision_score(y_test, y_pred_test, average='macro')
    recall_test = recall_score(y_test, y_pred_test, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred_test)

    # Test setindeki metrikleri ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def naive_bayes_classification_no_feature_selection(train_data, target_column, test_data=None, random_state=42):
    """
    Naive Bayes sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        train_data (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Naive Bayes modelini tanımla
    nb = GaussianNB()

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        nb.fit(X_train_fold, y_train_fold)
        y_pred = nb.predict(X_val_fold)
        y_proba = nb.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def naive_bayes_classification_with_feature_selection(train_data, target_column, test_data=None, random_state=42, k=10):
    """
    Naive Bayes sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Naive Bayes modelini tanımla
    nb = GaussianNB()

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        nb.fit(X_train_fold, y_train_fold)
        y_pred = nb.predict(X_val_fold)
        y_proba = nb.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    nb.fit(X_train_selected, y_train)
    y_pred = nb.predict(X_test_selected)
    y_proba = nb.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def logistic_regression_classification_no_feature_selection(train_data, target_column, test_data=None, random_state=42):
    """
    Logistic Regression sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Logistic Regression modelini tanımla
    lr = LogisticRegression(random_state=random_state, max_iter=1000)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        lr.fit(X_train_fold, y_train_fold)
        y_pred = lr.predict(X_val_fold)
        y_proba = lr.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def logistic_regression_classification_with_feature_selection(train_data, target_column, test_data=None, random_state=42, k=10):
    """
    Logistic Regression sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Logistic Regression modelini tanımla
    lr = LogisticRegression(random_state=random_state, max_iter=1000)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        lr.fit(X_train_fold, y_train_fold)
        y_pred = lr.predict(X_val_fold)
        y_proba = lr.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    lr.fit(X_train_selected, y_train)
    y_pred = lr.predict(X_test_selected)
    y_proba = lr.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def random_forest_classification_with_feature_selection(train_data, target_column, test_data, random_state=42, k=10):
    """
    Random Forest sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Random Forest modelini tanımla
    rf = RandomForestClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        rf.fit(X_train_fold, y_train_fold)
        y_pred = rf.predict(X_val_fold)
        y_proba = rf.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    rf.fit(X_train_selected, y_train)
    y_pred = rf.predict(X_test_selected)
    y_proba = rf.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def random_forest_classification_no_feature_selection(train_data, target_column, test_data, random_state=42):
    """
    Random Forest sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Random Forest modelini tanımla
    rf = RandomForestClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        rf.fit(X_train_fold, y_train_fold)
        y_pred = rf.predict(X_val_fold)
        y_proba = rf.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def svm_classification_no_feature_selection(train_data, target_column, test_data, random_state=42):
    """
    Support Vector Machine (SVM) sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SVM modelini tanımla
    svm = SVC(random_state=random_state, probability=True)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        svm.fit(X_train_fold, y_train_fold)
        y_pred = svm.predict(X_val_fold)
        y_proba = svm.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_proba = svm.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def svm_classification_with_feature_selection(train_data, target_column, test_data, random_state=42, k=10):
    """
    Support Vector Machine (SVM) sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # SVM modelini tanımla
    svm = SVC(random_state=random_state, probability=True)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        svm.fit(X_train_fold, y_train_fold)
        y_pred = svm.predict(X_val_fold)
        y_proba = svm.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    svm.fit(X_train_selected, y_train)
    y_pred = svm.predict(X_test_selected)
    y_proba = svm.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def gbm_classification_with_feature_selection(train_data, target_column, test_data, random_state=42, k=10):
    """
    Gradient Boosting Machine (GBM) sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # GBM modelini tanımla
    gbm = GradientBoostingClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        gbm.fit(X_train_fold, y_train_fold)
        y_pred = gbm.predict(X_val_fold)
        y_proba = gbm.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    gbm.fit(X_train_selected, y_train)
    y_pred = gbm.predict(X_test_selected)
    y_proba = gbm.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def gbm_classification_no_feature_selection(train_data, target_column, test_data, random_state=42):
    """
    Gradient Boosting Machine (GBM) sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # GBM modelini tanımla
    gbm = GradientBoostingClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        gbm.fit(X_train_fold, y_train_fold)
        y_pred = gbm.predict(X_val_fold)
        y_proba = gbm.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    y_proba = gbm.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics


def adaboost_classification_no_feature_selection(train_data, target_column, test_data, random_state=42):
    """
    AdaBoost sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # AdaBoost modelini tanımla
    adaboost = AdaBoostClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        adaboost.fit(X_train_fold, y_train_fold)
        y_pred = adaboost.predict(X_val_fold)
        y_proba = adaboost.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    adaboost.fit(X_train, y_train)
    y_pred = adaboost.predict(X_test)
    y_proba = adaboost.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def adaboost_classification_with_feature_selection(train_data, target_column, test_data, random_state=42, k=10):
    """
    AdaBoost sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # AdaBoost modelini tanımla
    adaboost = AdaBoostClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        adaboost.fit(X_train_fold, y_train_fold)
        y_pred = adaboost.predict(X_val_fold)
        y_proba = adaboost.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    adaboost.fit(X_train_selected, y_train)
    y_pred = adaboost.predict(X_test_selected)
    y_proba = adaboost.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def lda_classification_with_feature_selection(train_data, target_column, test_data, random_state=42, k=10):
    """
    Linear Discriminant Analysis (LDA) sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # LDA modelini tanımla
    lda = LinearDiscriminantAnalysis()

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        lda.fit(X_train_fold, y_train_fold)
        y_pred = lda.predict(X_val_fold)
        y_proba = lda.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    lda.fit(X_train_selected, y_train)
    y_pred = lda.predict(X_test_selected)
    y_proba = lda.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def lda_classification_no_feature_selection(train_data, target_column, test_data, random_state=42):
    """
    Linear Discriminant Analysis (LDA) sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # LDA modelini tanımla
    lda = LinearDiscriminantAnalysis()

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        lda.fit(X_train_fold, y_train_fold)
        y_pred = lda.predict(X_val_fold)
        y_proba = lda.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    y_proba = lda.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def ann_classification_no_feature_selection(train_data, target_column, test_data, random_state=42):
    """
    Artificial Neural Network (ANN) sınıflandırma ve 5-katlı çapraz doğrulama ile performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # ANN modelini tanımla (MLPClassifier)
    ann = MLPClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        ann.fit(X_train_fold, y_train_fold)
        y_pred = ann.predict(X_val_fold)
        y_proba = ann.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisinde tahmin yap
    ann.fit(X_train, y_train)
    y_pred = ann.predict(X_test)
    y_proba = ann.predict_proba(X_test)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

def ann_classification_with_feature_selection(train_data, target_column, test_data, random_state=42, k=10):
    """
    Artificial Neural Network (ANN) sınıflandırma ve 5-katlı çapraz doğrulama ile özellik seçimi ve performans metriklerini raporlar.

    Args:
        data_path (str): Eğitim veri setinin dosya yolu (Excel formatında).
        target_column (str): Hedef sütunun adı.
        test_data_path (str, optional): Test veri setinin dosya yolu (Excel formatında). Varsayılan None.
        random_state (int): Rastgelelik için sabit değer. Varsayılan 42.
        k (int): Seçilecek en iyi özellik sayısı. Varsayılan 10.

    Returns:
        metrics (dict): Performans metriklerini içeren bir sözlük.
    """
    # Eğitim veri setini yükle
    train_df = train_data

    # Özellikler ve hedef değişkeni ayır (eğitim verisi)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # SelectKBest ile özellik seçimi
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # ANN modelini tanımla (MLPClassifier)
    ann = MLPClassifier(random_state=random_state)

    # 5-katlı çapraz doğrulama (Stratified KFold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Çapraz doğrulama sırasında performans metriklerini depolamak için listeler
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    auc_scores = []
    mcc_scores = []

    # Çapraz doğrulama
    for train_idx, val_idx in cv.split(X_train_selected, y_train):
        X_train_fold, X_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Modeli eğit
        ann.fit(X_train_fold, y_train_fold)
        y_pred = ann.predict(X_val_fold)
        y_proba = ann.predict_proba(X_val_fold)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        precision = precision_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')

        # Çok sınıflı sınıflandırma için AUC hesapla
        auc = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

        # MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        # Metrikleri listelere ekle
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        auc_scores.append(auc)
        mcc_scores.append(mcc)

    # Çapraz doğrulama sonuçlarının ortalamasını al
    metrics = {
        'Accuracy': np.mean(accuracies),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'AUC': np.mean(auc_scores),
        'MCC': np.mean(mcc_scores)
    }

    # Test veri seti varsa, model test verisi üzerinde değerlendir

    test_df = test_data

    # Özellikler ve hedef değişkeni ayır (test verisi)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Test verisini seçilen özelliklerle daralt
    X_test_selected = selector.transform(X_test)

    # Test verisinde tahmin yap
    ann.fit(X_train_selected, y_train)
    y_pred = ann.predict(X_test_selected)
    y_proba = ann.predict_proba(X_test_selected)

    # Test verisi performans metriklerini hesapla
    accuracy_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='macro')
    precision_test = precision_score(y_test, y_pred, average='macro')
    recall_test = recall_score(y_test, y_pred, average='macro')

    auc_test = safe_roc_auc_score(y_val_fold, y_proba, multi_class='ovr', average='macro')

    mcc_test = matthews_corrcoef(y_test, y_pred)

    # Test verisi metriklerini ekle
    metrics.update({
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test Precision': precision_test,
        'Test Recall': recall_test,
        'Test AUC': auc_test,
        'Test MCC': mcc_test
    })

    return metrics

warnings.filterwarnings('ignore')

array = ['CropEstablishment_CT','CropEstablishment_ZT','LandType_Lowland','LandType_MediumLand','LandType_Upland','SoilType_Heavy','SoilType_Low','SoilType_Medium','SowingSchedule_T1','SowingSchedule_T2','SowingSchedule_T3','SowingSchedule_T4','SowingSchedule_T5','State_Bihar','State_UP','VarietyClass_LDV','VarietyClass_SDV']
# 'CropEstablishment_CT_line'
path = 'Data_processed2.xlsx'

for name in array:
    df = pd.read_excel(path)
    df = df[df[name] == 1]



    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    print(name)
    print('Decision Tree fs')
    metrics = decision_tree_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data, k=5)
    print(metrics)
    print('_______________________________________')
    metrics = decision_tree_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('Decision Tree no fs')
    print(metrics)
    print('_______________________________________')


    print('knn fs')
    metrics = knn_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data, k=5)
    print(metrics)
    print('_______________________________________')
    metrics = knn_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('knn no fs')
    print(metrics)
    print('_______________________________________')


    print('naivebayes fs')
    metrics = naive_bayes_classification_with_feature_selection(train_data, 'GrainYield', test_data, k=5)
    print(metrics)
    print('_______________________________________')
    metrics = naive_bayes_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('naive bayes no fs')
    print(metrics)
    print('_______________________________________')


    print('random forest fs')
    metrics = random_forest_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data, k=5)
    print(metrics)
    print('_______________________________________')
    metrics = random_forest_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('random forest no fs')
    print(metrics)
    print('_______________________________________')

    print('logistic regression fs')
    metrics = logistic_regression_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data, k=5)
    print(metrics)
    print('_______________________________________')
    metrics = logistic_regression_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('logistic regression no fs')
    print(metrics)
    print('_______________________________________')

    print('svm fs')
    metrics = svm_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data,k=5)
    print(metrics)
    print('_______________________________________')
    metrics = svm_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('svm no fs')
    print(metrics)
    print('_______________________________________')

    print('gbm fs')
    metrics = gbm_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data,k=5)
    print(metrics)
    print('_______________________________________')
    metrics = gbm_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('gbm no fs')
    print(metrics)
    print('_______________________________________')

    print('lda fs')
    metrics = lda_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data,k=5)
    print(metrics)
    print('_______________________________________')
    metrics = lda_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('lda no fs')
    print(metrics)
    print('_______________________________________')

    print('ann fs')
    metrics = ann_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data,k=5)
    print(metrics)
    print('_______________________________________')
    metrics = ann_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('ann no fs')
    print(metrics)
    print('_______________________________________')

    print('adaboost fs')
    metrics = adaboost_classification_with_feature_selection(train_data, 'GrainYield', test_data=test_data, k=5)
    print(metrics)
    print('_______________________________________')
    metrics = adaboost_classification_no_feature_selection(train_data, 'GrainYield', test_data=test_data)
    print('adaboost no fs')
    print(metrics)
    print('_______________________________________')





