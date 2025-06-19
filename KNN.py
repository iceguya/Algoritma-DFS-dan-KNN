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

print("--- Memulai Eksekusi Script ---")  # DEBUG POINT 1

# --- 1. Unduh Data dari Kaggle ---
print("Mengunduh dataset dari Kaggle...")  # DEBUG POINT 2
try:
    path = kagglehub.dataset_download("bismasajjad/global-ai-job-market-and-salary-trends-2025")
    print(f"Dataset berhasil diunduh ke: {path}")  # DEBUG POINT 3
except Exception as e:
    print(f"Gagal mengunduh dataset: {e}")  # DEBUG POINT 4
    print("Mencoba menggunakan path lokal jika sudah diunduh sebelumnya...")
    path = os.path.expanduser("~/.kaggle/kagglehub/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025/current")
    if not os.path.exists(path):
        print("Path lokal tidak ditemukan. Tidak dapat melanjutkan tanpa dataset.")  # DEBUG POINT 5
        exit()

# --- 2. Cari File CSV ---
print("Mencari file CSV di folder yang diunduh...")  # DEBUG POINT 6
csv_file_name = None
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.csv'):
            csv_file_name = file
            break
    if csv_file_name:
        break

if not csv_file_name:
    print(f"Tidak ditemukan file CSV di {path}. Harap verifikasi struktur dataset.")  # DEBUG POINT 7
    exit()

full_csv_path = os.path.join(path, csv_file_name)
print(f"Membaca file CSV: {full_csv_path}")  # DEBUG POINT 8

try:
    df = pd.read_csv(full_csv_path)
    print("Data berhasil dimuat. Melanjutkan pra-pemrosesan...")  # DEBUG POINT 9
except Exception as e:
    print(f"Gagal membaca file CSV dari {full_csv_path}: {e}")  # DEBUG POINT 10
    exit()

# --- 3. Pra-pemrosesan Data ---
print("Menampilkan 5 baris awal:")
print(df.head())

# Pilih kolom yang relevan
required_columns = ['Job Title', 'Location', 'Salary', 'Experience Level', 'Industry']
if not all(col in df.columns for col in required_columns):
    print(f"Tidak semua kolom yang dibutuhkan tersedia dalam dataset. Kolom yang dibutuhkan: {required_columns}")
    exit()

# Bersihkan data gaji jika perlu (hilangkan simbol, ubah ke float)
df['Salary'] = df['Salary'].astype(str).replace('[^0-9.]', '', regex=True).astype(float)

# Hilangkan baris dengan NaN
df.dropna(subset=required_columns, inplace=True)

# Label Encoding untuk kolom kategorikal
label_enc = LabelEncoder()
df['Job Title'] = label_enc.fit_transform(df['Job Title'])
df['Location'] = label_enc.fit_transform(df['Location'])
df['Experience Level'] = label_enc.fit_transform(df['Experience Level'])
df['Industry'] = label_enc.fit_transform(df['Industry'])

# Feature & Target
X = df[['Location', 'Salary', 'Experience Level', 'Industry']]
y = df['Job Title']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 4. Pelatihan Model ---
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# --- 5. Evaluasi Model ---
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
