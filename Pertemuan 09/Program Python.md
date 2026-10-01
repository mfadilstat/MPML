# Mata kuliah: Modern Prediksi dan Machine Learning

Author: Muhammad Fadil

Update: 2026-10-01

# Klasifikasi Bunga Iris dengan Random Forest (Python / scikit-learn)

Proyek ini menerapkan algoritma **Random Forest** untuk mengklasifikasikan spesies bunga _Iris_ (`setosa`, `versicolor`, `virginica`) berdasarkan empat ukuran morfologi bunga. Seluruh alur kerja ada di notebook `Program Python.ipynb`: eksplorasi data, pembagian data berstrata, pelatihan model, evaluasi, pentingnya variabel, validasi silang 10-fold, _tuning_ dengan `GridSearchCV`, visualisasi, dan prediksi data baru.

> Semua angka di dokumen ini diambil dari output yang tersimpan di notebook (`random_state=42`). Pada versi scikit-learn yang berbeda, angka bisa sedikit berbeda.

---

## Daftar Isi

1. [Tujuan](#1-tujuan)
2. [Dataset](#2-dataset)
3. [Persyaratan & Cara Menjalankan](#3-persyaratan--cara-menjalankan)
4. [Konsep Singkat](#4-konsep-singkat)
5. [Penjelasan Kode & Interpretasi Hasil](#5-penjelasan-kode--interpretasi-hasil)
6. [Ringkasan Hasil](#6-ringkasan-hasil)
7. [Kesimpulan](#7-kesimpulan)

---

## 1. Tujuan

- Membangun model klasifikasi spesies Iris dengan Random Forest.
- Mengukur kinerja dengan **akurasi OOB**, **akurasi data uji**, **confusion matrix**, dan **10-fold cross-validation**.
- Mengetahui variabel yang paling berpengaruh, memakai dua cara: Gini importance dan permutation importance.
- Mencari kombinasi hiperparameter terbaik dengan _grid search_.
- Memakai model untuk memprediksi data baru.

## 2. Dataset

Dataset `iris` dari `sklearn.datasets` (Fisher, 1936): **150 observasi**, **4 fitur numerik**, dan **3 kelas seimbang** (masing-masing 50 sampel).

| Fitur               | Keterangan           |
| ------------------- | -------------------- |
| `sepal length (cm)` | Panjang kelopak luar |
| `sepal width (cm)`  | Lebar kelopak luar   |
| `petal length (cm)` | Panjang mahkota      |
| `petal width (cm)`  | Lebar mahkota        |

Target `y` berupa kode numerik: `0 = setosa`, `1 = versicolor`, `2 = virginica`.

## 3. Persyaratan & Cara Menjalankan

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook Program_Python.ipynb
```

Jalankan seluruh sel dari atas ke bawah (**Run All**). Notebook menyimpan dua gambar ke folder kerja: `cm.png` (confusion matrix) dan `imp.png` (pentingnya variabel).

## 4. Konsep Singkat

**Random Forest** menggabungkan banyak pohon keputusan (_ensemble_):

1. **Bagging:** tiap pohon dilatih pada sampel bootstrap dari data latih.
2. **Subset fitur acak:** tiap percabangan hanya mempertimbangkan `max_features` fitur acak.

Prediksi akhir ditentukan lewat **suara terbanyak** semua pohon. Data yang tidak terpilih dalam bootstrap suatu pohon (_out-of-bag_, ± 37%) dipakai untuk menguji pohon itu, sehingga muncul **akurasi OOB** sebagai estimasi kinerja tanpa data uji terpisah.

## 5. Penjelasan Kode & Interpretasi Hasil

### 5.1 Import library (Sel 1)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
```

Memuat library untuk manipulasi data (`pandas`), visualisasi (`matplotlib`), model (`RandomForestClassifier`), pembagian data dan validasi (`train_test_split`, `cross_val_score`, `GridSearchCV`), metrik evaluasi, dan permutation importance. `numpy` diimpor tetapi belum dipakai di notebook ini.

### 5.2 Memuat dan mengeksplorasi data (Sel 2)

```python
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
names = iris.target_names
print(X.describe().round(2))
print(y.value_counts())
```

`as_frame=True` membuat `X` berupa DataFrame sehingga nama kolom terbawa. `names` menyimpan nama kelas untuk label laporan dan grafik.

**Hasil:**

| Statistik    | sepal length | sepal width | petal length | petal width |
| ------------ | ------------ | ----------- | ------------ | ----------- |
| mean         | 5,84         | 3,06        | 3,76         | 1,20        |
| std          | 0,83         | 0,44        | 1,77         | 0,76        |
| min          | 4,30         | 2,00        | 1,00         | 0,10        |
| median (50%) | 5,80         | 3,00        | 4,35         | 1,30        |
| max          | 7,90         | 4,40        | 6,90         | 2,50        |

Setiap kelas (0, 1, 2) berjumlah 50 sampel.

**Interpretasi:**

- Data lengkap (150 baris untuk semua fitur) dan sudah numerik, jadi tidak perlu pembersihan.
- Kelas **seimbang sempurna**, sehingga akurasi adalah metrik yang adil.
- Fitur petal punya simpangan baku lebih besar dibanding sepal (1,77 untuk petal length vs 0,44 untuk sepal width), dan rata-rata petal length (3,76) berbeda dari mediannya (4,35). Ini menandakan petal menyebar luas antar spesies, tanda awal bahwa fitur petal akan membantu pemisahan kelas.

> Sel ke-3 di notebook kosong dan bisa dihapus.

### 5.3 Pembagian data dan pelatihan model (Sel 4)

```python
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
rf = RandomForestClassifier(
    n_estimators=500,      # jumlah pohon (B)
    max_features="sqrt",   # mtry = sqrt(p)
    oob_score=True,        # hitung akurasi Out-of-Bag
    random_state=42)
rf.fit(Xtr, ytr)
print("Akurasi OOB:", round(rf.oob_score_, 4))
```

| Bagian                | Penjelasan                                                               |
| --------------------- | ------------------------------------------------------------------------ |
| `test_size=0.3`       | 30% data (45 sampel) untuk uji, 70% (105 sampel) untuk latih             |
| `stratify=y`          | Proporsi kelas dijaga sama di data latih dan uji (35/35/35 dan 15/15/15) |
| `n_estimators=500`    | Jumlah pohon                                                             |
| `max_features="sqrt"` | Tiap split memakai √4 = 2 fitur acak                                     |
| `oob_score=True`      | Hitung akurasi OOB                                                       |
| `random_state=42`     | Hasil dapat direproduksi                                                 |

**Hasil:** `Akurasi OOB: 0.9524`

**Interpretasi:** sekitar **95,24%** data latih diprediksi benar oleh pohon-pohon yang tidak melihat data tersebut; setara dengan 5 kesalahan dari 105 sampel (OOB error 4,76%). Ini estimasi awal kinerja yang cukup baik.

### 5.4 Evaluasi pada data uji (Sel 5)

```python
pred = rf.predict(Xte)
print("Akurasi uji:", round(accuracy_score(yte, pred), 4))
cm = confusion_matrix(yte, pred)
print(cm)
print(classification_report(yte, pred, target_names=names, digits=3))
```

**Hasil:** `Akurasi uji: 0.9111`

**Confusion matrix (baris = aktual, kolom = prediksi):**

| Aktual \ Prediksi | setosa | versicolor | virginica |
| ----------------- | ------ | ---------- | --------- |
| **setosa**        | 15     | 0          | 0         |
| **versicolor**    | 0      | 14         | 1         |
| **virginica**     | 0      | 3          | 12        |

**Classification report:**

| Kelas        | Precision | Recall | F1-score  | Support |
| ------------ | --------- | ------ | --------- | ------- |
| setosa       | 1,000     | 1,000  | 1,000     | 15      |
| versicolor   | 0,824     | 0,933  | 0,875     | 15      |
| virginica    | 0,923     | 0,800  | 0,857     | 15      |
| **accuracy** |           |        | **0,911** | 45      |
| macro avg    | 0,916     | 0,911  | 0,911     | 45      |

**Interpretasi:**

- Dari 45 data uji, **41 benar** dan **4 salah** (akurasi 91,11%).
- **Setosa sempurna** (15/15). Semua kesalahan terjadi antara versicolor dan virginica, karena ukuran kedua spesies itu tumpang tindih.
- Kesalahan lebih sering mengarah ke satu sisi: 3 virginica ditebak versicolor, sedangkan 1 versicolor ditebak virginica.
- **Precision vs recall:** versicolor punya recall tinggi (0,933) tetapi precision lebih rendah (0,824), karena model "menampung" tiga virginica ke kelas ini. Sebaliknya virginica punya precision tinggi (0,923) tetapi recall lebih rendah (0,800), artinya ketika model menyebut virginica biasanya benar, tetapi beberapa virginica terlewat.
- Akurasi uji (91,1%) sedikit di bawah OOB (95,2%). Selisih ini setara dengan 2 sampel pada data uji sekecil 45, jadi masih dalam kewajaran dan tidak menunjukkan overfitting yang serius.

### 5.5 Pentingnya variabel (Sel 6)

```python
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.round(4))   # Gini importance (MDI)
pi = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=42)
print(pd.Series(pi.importances_mean, index=X.columns).round(4))
```

Dua ukuran dipakai:

- **Gini importance (MDI):** total penurunan impuritas Gini yang disumbang tiap fitur, dihitung dari data latih. Jumlah seluruh fitur = 1.
- **Permutation importance:** penurunan akurasi pada **data uji** ketika nilai suatu fitur diacak (diulang 30 kali).

**Hasil:**

| Fitur             | Gini importance | Permutation importance |
| ----------------- | --------------- | ---------------------- |
| petal width (cm)  | **0,4334**      | 0,1763                 |
| petal length (cm) | 0,4179          | **0,1778**             |
| sepal length (cm) | 0,1208          | 0,0015                 |
| sepal width (cm)  | 0,0279          | 0,0081                 |

**Interpretasi:**

- Kedua ukuran sepakat bahwa **`petal width` dan `petal length` adalah fitur terpenting**. Gini importance keduanya sekitar 0,84 dari total, dan jika salah satunya diacak, akurasi uji turun sekitar 17–18 poin persentase.
- Selisih antara `petal width` dan `petal length` sangat kecil dan urutannya bahkan terbalik pada dua ukuran. Keduanya berkorelasi tinggi, jadi sulit dikatakan mana yang lebih unggul.
- **Fitur sepal hampir tidak berkontribusi.** `sepal length` punya Gini importance 0,12, tetapi permutation importance-nya hanya 0,0015, artinya mengacak fitur ini nyaris tidak mengubah akurasi uji. Gini importance cenderung menaksir terlalu tinggi fitur kontinu dan fitur yang berkorelasi dengan fitur penting, sehingga permutation importance pada data uji umumnya lebih dapat dipercaya.

### 5.6 Cross-validation 10-fold (Sel 7)

```python
cv = cross_val_score(RandomForestClassifier(n_estimators=500, random_state=42), X, y, cv=10)
print("CV 10-fold:", cv.mean().round(4), "+/-", cv.std().round(4))
```

Data **seluruhnya** (150 sampel) dibagi 10 lipatan; model dilatih pada 9 lipatan dan diuji pada 1 lipatan, diulang 10 kali. Untuk klasifikasi, `cv=10` otomatis memakai lipatan berstrata.

**Hasil:** `CV 10-fold: 0.9667 +/- 0.0333`

**Interpretasi:** akurasi rata-rata **96,67%** dengan simpangan baku 3,33 poin antar lipatan. Estimasi ini lebih stabil daripada satu kali pembagian 70/30 karena dirata-ratakan dari 10 uji. Angkanya sedikit lebih tinggi dari akurasi uji (91,1%), selisih yang sebagian besar disebabkan oleh sedikitnya data uji pada satu kali split dan oleh model CV yang dilatih dengan data lebih banyak (135 sampel per lipatan vs 105). Secara keseluruhan, kinerja sebenarnya diperkirakan di kisaran **± 91–97%**.

### 5.7 Tuning dengan GridSearchCV (Sel 8)

```python
param = {"n_estimators": [100, 300, 500],
         "max_depth": [None, 3, 5],
         "max_features": ["sqrt", 2, 4]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param, cv=5)
gs.fit(Xtr, ytr)
print(gs.best_params_, round(gs.best_score_, 4))
print("Akurasi uji model terbaik:", accuracy_score(yte, gs.best_estimator_.predict(Xte)))
```

Grid search menguji 3 × 3 × 3 = **27 kombinasi** hiperparameter, masing-masing dengan 5-fold CV (total 135 kali pelatihan) pada **data latih saja**, sehingga data uji tetap bersih.

**Hasil:**

```
{'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 100}  0.9524
Akurasi uji model terbaik: 0.8888888888888888
```

**Interpretasi:**

- Kombinasi terbaik: 100 pohon, tanpa batas kedalaman, `max_features="sqrt"`, dengan akurasi CV **95,24%**. Nilai ini sama dengan akurasi OOB model awal (0,9524), dan konfigurasinya pun pada dasarnya sama dengan model awal (hanya jumlah pohon yang lebih sedikit).
- Akurasi uji model terbaik **88,89%** (40/45), bahkan sedikit di bawah model awal (91,11%, atau 41/45). Selisihnya hanya **1 sampel**, jadi ini bukan bukti bahwa model hasil tuning lebih buruk. Hasil ini menunjukkan bahwa **tuning tidak memberi perbaikan nyata** pada dataset yang sudah mudah ini.
- Pelajaran praktis: memilih model terbaik berdasarkan CV pada data latih belum tentu menang pada satu kali data uji kecil. Pada data seperti ini, banyak kombinasi hiperparameter yang hasilnya hampir sama.

### 5.8 Visualisasi dan prediksi data baru (Sel 9)

```python
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout(); plt.savefig("cm.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.5))
imp[::-1].plot.barh(ax=ax, color="#2E75B6")
ax.set_title("Pentingnya Variabel (Gini Importance)")
plt.tight_layout(); plt.savefig("imp.png", dpi=150)

new = pd.DataFrame([[5.1, 3.5, 1.4, 0.2],
                    [6.0, 2.9, 4.5, 1.5],
                    [6.9, 3.1, 5.4, 2.1]], columns=X.columns)
print(rf.predict(new))
print(rf.predict_proba(new).round(3))
```

- Bagian pertama menggambar confusion matrix dan menyimpannya sebagai `cm.png`.
- Bagian kedua menggambar diagram batang horizontal Gini importance (`imp[::-1]` membalik urutan agar fitur terpenting tampil di atas) dan menyimpannya sebagai `imp.png`.
- Bagian ketiga memprediksi tiga bunga hipotetis dengan model `rf`. Nama kolom harus sama dengan data latih.

**Hasil prediksi:**

| Bunga | Ukuran (SL, SW, PL, PW) | Prediksi           | P(setosa) | P(versicolor) | P(virginica) |
| ----- | ----------------------- | ------------------ | --------- | ------------- | ------------ |
| 1     | 5,1 / 3,5 / 1,4 / 0,2   | **0 = setosa**     | 1,000     | 0,000         | 0,000        |
| 2     | 6,0 / 2,9 / 4,5 / 1,5   | **1 = versicolor** | 0,000     | 1,000         | 0,000        |
| 3     | 6,9 / 3,1 / 5,4 / 2,1   | **2 = virginica**  | 0,000     | 0,006         | 0,994        |

**Interpretasi:**

- `predict()` mengembalikan kode kelas (0, 1, 2). Petakan ke nama dengan `names[...]`.
- `predict_proba()` adalah rata-rata probabilitas kelas dari seluruh pohon, bukan probabilitas terkalibrasi.
- Bunga 1 dan 2 diprediksi dengan keyakinan penuh. Bunga 3 diprediksi virginica dengan keyakinan 99,4%, dan sisa 0,6% ke versicolor mencerminkan area tumpang tindih kedua spesies.

## 6. Ringkasan Hasil

| Ukuran                            | Nilai                            |
| --------------------------------- | -------------------------------- |
| Data latih / uji                  | 105 / 45 (berstrata)             |
| Akurasi OOB                       | 95,24%                           |
| Akurasi data uji                  | 91,11% (41/45)                   |
| Akurasi CV 10-fold (seluruh data) | 96,67% ± 3,33                    |
| Best CV Grid Search (data latih)  | 95,24%                           |
| Akurasi uji model hasil tuning    | 88,89% (40/45)                   |
| Fitur terpenting                  | `petal width`, `petal length`    |
| Fitur paling lemah                | `sepal length` dan `sepal width` |

## 7. Kesimpulan

1. Random Forest mengklasifikasikan spesies Iris dengan akurasi sekitar **91–97%** tergantung metode evaluasi. Hasil OOB, data uji, dan CV konsisten dalam rentang itu.
2. **Setosa** terpisah sempurna; kesalahan hanya terjadi antara **versicolor dan virginica**.
3. **`petal width` dan `petal length`** adalah penentu utama klasifikasi, sedangkan fitur sepal hampir tidak berkontribusi (terutama menurut permutation importance).
4. Hyperparameter tuning tidak memberi peningkatan berarti; konfigurasi bawaan sudah memadai untuk dataset ini.
5. Perbedaan kecil antar model (1–2 sampel) pada data uji 45 sampel sebaiknya tidak ditafsirkan sebagai perbedaan kinerja yang nyata.

## Referensi

- Breiman, L. (2001). _Random Forests_. Machine Learning, 45(1), 5–32.
- Fisher, R. A. (1936). _The use of multiple measurements in taxonomic problems_. Annals of Eugenics, 7(2), 179–188.
- Pedregosa, F. et al. (2011). _Scikit-learn: Machine Learning in Python_. JMLR, 12, 2825–2830.
