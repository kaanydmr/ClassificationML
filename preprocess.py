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

from sklearn.preprocessing import MinMaxScaler
def histogram(s,df):
    plt.figure(figsize=(10, 6))
    plt.hist(df[s], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(s)
    plt.ylabel('Frequency')
    plt.title('Attribute Distribution')
    plt.show()



def modemedianmean(s,df):
    mean_value = df[s].mean()
    median_value = df[s].median()
    mode_value = df[s].mode()[0]  # Mode birden fazla olabilir, ilk değeri alıyoruz

    # Sonuçları yazdır
    print(f"{s} sütununun ortalaması (Mean): {mean_value}")
    print(f"{s} sütununun medyanı (Median): {median_value}")
    print(f"{s} sütununun modu (Mode): {mode_value}")

def zscore(s,df):
    z_scores = (df[s] - df[s].mean()) / df[s].std()

    # Outlier'ları bul
    outliers = df[np.abs(z_scores) > 3]
    for index, value in outliers[s].items():
        print(f"Satır: {index}, {s} Değeri: {value}")

    df.loc[np.abs(z_scores) > 3, s] = np.nan


def iqr_outliers(s, df):
    Q1 = df[s].quantile(0.25)  # Birinci çeyrek (Q1)
    Q3 = df[s].quantile(0.75)  # Üçüncü çeyrek (Q3)
    IQR = Q3 - Q1  # Çeyrekler arası açıklık (IQR)

    # Alt ve üst sınırları hesapla
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outlier'ları bul
    outliers = df[(df[s] < lower_bound) | (df[s] > upper_bound)]
    for index, value in outliers[s].items():
        print(f"Satır: {index}, {s} Değeri: {value}")

    # Outlier'ları NaN yap
    df.loc[(df[s] < lower_bound) | (df[s] > upper_bound), s] = np.nan

def checknumeric(s,df):
    non_numeric_rows = df[~df[s].apply(pd.to_numeric, errors='coerce').notna()]

    # Sonuçları yazdır
    print(f"{s} sütununda numeric olmayan ve NaN olmayan değerler ve satır numaraları:")
    for index, row in non_numeric_rows.iterrows():
        if pd.notna(row[s]):  # NaN olmayanları yazdır
            print(f"Satır: {index}, Değer: {row[s]}")



def knn(s, df, path):

    # Sadece sayısal sütunları seç
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Veriyi ölçekle (Min-Max veya Z-Score ile)
    scaler = MinMaxScaler()  # Alternatif: StandardScaler()
    numeric_df_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

    # KNN Imputer oluştur ve veriyi doldur
    imputer = KNNImputer(n_neighbors=3)
    imputed_data = imputer.fit_transform(numeric_df_scaled)

    # Doldurulan veriyi geri ölçekle
    numeric_df_imputed = pd.DataFrame(scaler.inverse_transform(imputed_data), columns=numeric_df.columns)

    # HerbicideMonth sütununu orijinal veri setine geri koy
    df[s] = numeric_df_imputed[s]

    # Sonuçları Excel'e kaydet
    df.to_excel(path, index=False)


def handle_outliers_and_missing(s, df, path,neighbors=3):
    # 1. Z-Score ile outlier tespiti
    Q1 = df[s].quantile(0.25)  # Birinci çeyrek (Q1)
    Q3 = df[s].quantile(0.75)  # Üçüncü çeyrek (Q3)
    IQR = Q3 - Q1  # Çeyrekler arası açıklık (IQR)

    # Alt ve üst sınırları hesapla
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outlier'ları bul
    outliers = df[(df[s] < lower_bound) | (df[s] > upper_bound)]
    for index, value in outliers[s].items():
        print(f"Satır: {index}, {s} Değeri: {value}")

    # Outlier'ları NaN yap
    df.loc[(df[s] < lower_bound) | (df[s] > upper_bound), s] = np.nan

    # 2. KNN ile doldurma işlemi
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Sadece sayısal sütunları seç
    imputer = KNNImputer(n_neighbors=neighbors)
    imputed_data = imputer.fit_transform(numeric_df)

    # Doldurulan veriyi geri DataFrame'e dönüştür
    numeric_df_imputed = pd.DataFrame(imputed_data, columns=numeric_df.columns)

    # Orijinal veri setini güncelle
    for col in numeric_df_imputed.columns:
        df[col] = numeric_df_imputed[col]

    df.to_excel(path, index=False)


def floor(s,df,path):
    if s in df.columns:
        df[s] = df[s].apply(lambda x: np.floor(x) if pd.notna(x) else x)

    df.to_excel(path, index=False)


def medyaniledoldur(s,df,path):
    df[s] = df[s].fillna(df[s].median())
    df.to_excel(path, index=False)
# zscore-numeric-
def outlierdoldurma(s,df,path):
    z_scores = (df[s] - df[s].mean()) / df[s].std()
    outliers = np.abs(z_scores) > 3
    df.loc[outliers, s] = df[s].median()
    df.to_excel(path, index=False)


def meaniledoldur(s,df,path):
    df[s] = df[s].fillna(df[s].mean())
    df.to_excel(path, index=False)

def check_missing_values(df):
    if df.isnull().values.any():
        print("Veri setinde eksik (NaN) değerler var.")
        print("\nEksik değerlerin sütun bazında sayısı:")
        print(df.isnull().sum())
    else:
        print("Veri setinde eksik (NaN) değer bulunmuyor.")

def detect_outliers_iqr(df):
    for column in df.select_dtypes(include=[np.number]).columns:  # Sadece sayısal sütunları kontrol et
        Q1 = df[column].quantile(0.25)  # Birinci çeyrek (Q1)
        Q3 = df[column].quantile(0.75)  # Üçüncü çeyrek (Q3)
        IQR = Q3 - Q1  # Çeyrekler arası açıklık (IQR)

        # Alt ve üst sınırları hesapla
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Outlier'ları bul
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        # Outlier'ları yazdır
        for index, value in outliers[column].items():
            print(f"Satır: {index}, {column} Değeri: {value}")


def stratified_5fold_logistic_regression_df_with_metrics(df, target_column='GrainYield'):
    """
    DataFrame kullanarak 5-Fold Stratified Cross Validation ile Logistic Regression uygular
    ve AUC, Accuracy, F1-score, Precision, Recall, Matthews Correlation Coefficient (MCC) metriklerini hesaplar.

    Args:
        df (pd.DataFrame): Veri çerçevesi (features ve target).
        target_column (str): Hedef değişkenin sütun adı.

    Returns:
        dict: Metriklerin her fold için değerleri ve ortalamaları.
    """
    # Özellikler ve hedef değişkeni ayır
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Stratified K-Fold tanımla
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model tanımla (max_iter artırıldı)
    model = LogisticRegression(max_iter=5000)

    # Metrikleri saklamak için listeler
    auc_scores = []
    accuracies = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    mcc_scores = []

    # Cross-validation döngüsü
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Modeli eğit
        model.fit(X_train, y_train)

        # Tahmin yap
        y_pred = model.predict(X_test)

        # Pozitif sınıfların olasılıklarını al
        y_prob = model.predict_proba(X_test)  # Tüm sınıflar için olasılık

        # AUC hesapla (multi_class='ovr' çok sınıflı sınıflandırma için uygun)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        accuracy = accuracy_score(y_test, y_pred)

        # Çok sınıflı metrik hesaplamaları için average='macro' veya 'weighted' kullanabilirsin
        f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' ile F1 hesapla
        precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' ile Precision hesapla
        recall = recall_score(y_test, y_pred, average='weighted')  # 'weighted' ile Recall hesapla
        mcc = matthews_corrcoef(y_test, y_pred)

        # Metrikleri listelere ekle
        auc_scores.append(auc)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        mcc_scores.append(mcc)

    # Ortalama metrikleri hesapla
    metrics = {
        'AUC': {'scores': auc_scores, 'mean': np.mean(auc_scores)},
        'Accuracy': {'scores': accuracies, 'mean': np.mean(accuracies)},
        'F1 Score': {'scores': f1_scores, 'mean': np.mean(f1_scores)},
        'Precision': {'scores': precision_scores, 'mean': np.mean(precision_scores)},
        'Recall': {'scores': recall_scores, 'mean': np.mean(recall_scores)},
        'MCC': {'scores': mcc_scores, 'mean': np.mean(mcc_scores)}
    }

    return metrics


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
'''
file_path = "Data_processed1.xlsx"
train_path = "train_data.xlsx"

df = pd.read_excel(train_path)
'''
'''
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Eğitim ve test veri setlerini farklı Excel dosyalarına kaydet
train_file_path = "train_data.xlsx"
test_file_path = "test_data.xlsx"

train_data.to_excel(train_file_path, index=False)
test_data.to_excel(test_file_path, index=False)

'''
'''
a='DaysFromHerbicideToHarvest'
b='Longitude'
c='HerbicideYear'
d='HerbicideMonth'
e='HerbicideDay'
f='HerbicideWeekNum'
g='DaysFromSowingToHerbicide'

arr = ['Split3Urea','IrrigationNumber','FirstIrrigationDay','SecondIrrigationDay','HerbicideDose','HarvestMonth','HarvestWeekNum','DaysFromSowingToHerbicide']

'''
'''
knn(a,df,file_path)
floor(a,df,file_path)

knn(b,df,file_path)
floor(b,df,file_path)

knn(c,df,file_path)
floor(c,df,file_path)

knn(d,df,file_path)
floor(d,df,file_path)

knn(e,df,file_path)
floor(e,df,file_path)

knn(f,df,file_path)
floor(f,df,file_path)

knn(g,df,file_path)
floor(g,df,file_path)


handle_outliers_and_missing(g,df,file_path)

for x in arr:
    handle_outliers_and_missing(x,df,file_path)
    floor(x,df,file_path)

'''


'''

metrics = stratified_5fold_logistic_regression_df_with_metrics(df)

# Metrikleri yazdır
for metric, values in metrics.items():
    print(f"{metric}:")
    print(f"  Scores for each fold: {values['scores']}")
    print(f"  Mean {metric}: {values['mean']:.4f}")
    print()
'''
'''
train_data = df[df["VarietyClass_LDV"] == 1]
train_data.to_excel("VarietyClass_LDV.xlsx", index=False)

'''
'''
train_path = "State_UP.xlsx"
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
'''
Model Doğruluğu (Test Seti): 0.47
AUC (Area Under Curve): 0.59
F1 Skoru: 0.46
Precision: 0.46
Recall: 0.46
MCC (Matthews Correlation Coefficient): 0.17
'''


def minmax_normalize_excel(input_file, output_file, target_column):
    """
    Verilen Excel dosyasındaki verileri 0-1 aralığında normalize eder ve sonucu yeni bir dosyaya kaydeder.

    Args:
        input_file (str): Giriş Excel dosyasının yolu.
        output_file (str): Çıkış Excel dosyasının yolu.
        target_column (str): Hedef sütunun (target attribute) adı. Bu sütun normalize edilmez.

    Returns:
        None
    """
    # Excel dosyasını oku
    data = pd.read_excel(input_file)

    # Hedef sütunu ayır
    target = data[target_column]

    # Binary sütunları tespit et (sadece 0 ve 1 değerleri içeren sütunlar)
    binary_columns = data.columns[(data.nunique() == 2) & (data.isin([0, 1]).all())]

    # Binary sütunlar ve hedef sütunu dışındaki numeric verileri seç
    features_to_normalize = data.drop(columns=[target_column] + list(binary_columns))

    # Min-Max normalizasyonu uygula
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features_to_normalize)

    # Normalize edilmiş veriyi DataFrame'e dönüştür
    normalized_df = pd.DataFrame(normalized_features, columns=features_to_normalize.columns)

    # Virgülden sonra 3 basamağa yuvarla
    normalized_df = normalized_df.round(3)

    # Orijinal binary sütunları ve hedef sütunu ekle
    for col in binary_columns:
        normalized_df[col] = data[col]
    normalized_df[target_column] = target.reset_index(drop=True)

    # Sütun sırasını koru
    final_df = normalized_df[data.columns]

    # Yeni dosyayı kaydet
    final_df.to_excel(output_file, index=False)

    print(f"Normalize edilmiş veriler {output_file} dosyasına kaydedildi.")

'''
minmax_normalize_excel('Data_processed1.xlsx','Data_processed2.xlsx','GrainYield')
'''
from imblearn.over_sampling import SMOTE

path = 'Data_processed2.xlsx'
df = pd.read_excel(path)

X = df.drop(columns=['GrainYield'])
y = df['GrainYield']

# Sınıfların sayısını kontrol et
class_counts = y.value_counts()


if all(class_counts > 1):  # Tüm sınıflar için yeterli örnek olmalı
    # SMOTE işlemi
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Yeni veri çerçevesi oluşturuluyor
    df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                    pd.DataFrame(y_resampled, columns=['GrainYield'])], axis=1)
else:
    print(f"Yeterli sınıf örneği yok, SMOTE uygulanmadı: ")

value_counts = df['GrainYield'].value_counts()
# Daire grafiği çizme
plt.figure(figsize=(8, 8))
value_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])

# Grafik detaylarını ayarlama
plt.title('GrainYield Distribution')
plt.ylabel('')  # Y ekseni etiketini kaldırır
plt.show()

