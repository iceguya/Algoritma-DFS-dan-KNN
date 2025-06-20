import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Memulai Eksekusi Script ---")

print("Mengunduh dataset dari Kaggle...")
try:
    path = kagglehub.dataset_download("bismasajjad/global-ai-job-market-and-salary-trends-2025")
    print(f"Dataset berhasil diunduh ke: {path}")
except Exception as e:
    print(f"Gagal mengunduh dataset: {e}")
    print("Mencoba menggunakan path lokal jika sudah diunduh sebelumnya...")
    path = os.path.expanduser("~/.kaggle/kagglehub/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025/current")
    if not os.path.exists(path):
        print("Path lokal tidak ditemukan. Tidak dapat melanjutkan tanpa dataset.")
        exit()

print("Mencari file CSV di folder yang diunduh...")
csv_file_name = None
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.csv'):
            csv_file_name = file
            break
    if csv_file_name:
        break

if not csv_file_name:
    print(f"Tidak ditemukan file CSV di {path}. Harap verifikasi struktur dataset.")
    exit()

full_csv_path = os.path.join(path, csv_file_name)
print(f"Membaca file CSV: {full_csv_path}")

try:
    df = pd.read_csv(full_csv_path)
    print("Data berhasil dimuat. Melanjutkan pra-pemrosesan...")
except Exception as e:
    print(f"Gagal membaca file CSV dari {full_csv_path}: {e}")
    exit()

print("Menampilkan 5 baris awal:")
print(df.head())

required_columns = ['job_title', 'company_location', 'salary_usd', 'experience_level', 'industry']
if not all(col in df.columns for col in required_columns):
    print(f"Tidak semua kolom yang dibutuhkan tersedia dalam dataset. Kolom yang dibutuhkan: {required_columns}")
    exit()

df.dropna(subset=required_columns, inplace=True)

label_enc = LabelEncoder()
df['job_title'] = label_enc.fit_transform(df['job_title'])
df['company_location'] = label_enc.fit_transform(df['company_location'])
df['experience_level'] = label_enc.fit_transform(df['experience_level'])
df['industry'] = label_enc.fit_transform(df['industry'])

X = df[['company_location', 'salary_usd', 'experience_level', 'industry']]
y = df['job_title']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Evaluasi Model ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("--- Eksekusi Script Selesai ---")
