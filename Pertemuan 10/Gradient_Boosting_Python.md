<h1>Mata kuliah : Modern Prediksi dan Machine Learning</h1>
<h3>Pertemuan 10: Gradient Boosting</h3>

<tt>Author: Muhammad Fadil</tt><br>
<tt>Update: 2026-10-01</tt>

# Prediksi Perkembangan Diabetes dengan Gradient Boosting (Python)

Program **Gradient Boosting Regressor** untuk memprediksi ukuran perkembangan penyakit diabetes pada pasien (masalah **regresi**). Notebook `Gradient_Boosting.ipynb` membandingkan Gradient Boosting dengan dua model pembanding (Regresi Linear dan Random Forest), menganalisis pengaruh jumlah iterasi dan _learning rate_, memakai _early stopping_, melakukan validasi silang dan _grid search_, lalu menafsirkan pentingnya variabel.

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

- Memprediksi skor perkembangan diabetes (variabel kontinu) dari 10 variabel klinis.
- Membandingkan Gradient Boosting dengan Regresi Linear dan Random Forest.
- Memahami risiko _overfitting_ pada boosting dan cara mengendalikannya (jumlah iterasi, _learning rate_, _early stopping_).
- Mencari hiperparameter terbaik dengan _grid search_ dan mengevaluasi dengan validasi silang.
- Mengidentifikasi variabel yang paling berpengaruh.

## 2. Dataset

Dataset `diabetes` dari `sklearn.datasets`: **442 pasien**, **10 fitur**, dan satu target numerik.

| Variabel    | Keterangan                                                          |
| ----------- | ------------------------------------------------------------------- |
| `age`       | Usia                                                                |
| `sex`       | Jenis kelamin                                                       |
| `bmi`       | Indeks massa tubuh                                                  |
| `bp`        | Tekanan darah rata-rata                                             |
| `s1` – `s6` | Enam ukuran serum darah (mis. kolesterol, trigliserida, gula darah) |
| `target`    | Ukuran perkembangan penyakit satu tahun setelah pengukuran awal     |

Fitur pada dataset ini sudah **dipusatkan dan diskalakan** (rata-rata 0, simpangan baku ≈ 0,048), sehingga tidak perlu normalisasi tambahan.

## 3. Persyaratan & Cara Menjalankan

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Jalankan seluruh sel dari atas ke bawah (**Run All**). Notebook menghasilkan berkas:

| Berkas         | Isi                                                                  |
| -------------- | -------------------------------------------------------------------- |
| `diabetes.csv` | Data lengkap (fitur + `y`), diekspor agar dapat dipakai di program R |
| `curve.png`    | Kurva RMSE terhadap jumlah iterasi                                   |
| `imp.png`      | Pentingnya variabel                                                  |
| `avp.png`      | Aktual vs prediksi pada data uji                                     |

## 4. Konsep Singkat

**Gradient Boosting** membangun model secara **berurutan**. Pohon pertama memprediksi target; pohon-pohon berikutnya masing-masing dilatih untuk memperbaiki **sisa kesalahan (residual)** model sebelumnya. Prediksi akhir adalah penjumlahan semua pohon, tiap pohon dikalikan _learning rate_ (ν):

$$F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$$

Perbedaan dengan Random Forest: Random Forest membangun pohon-pohon dalam, **paralel** dan independen, lalu merata-ratakannya. Boosting memakai pohon-pohon **dangkal** (_weak learner_) yang dibangun **berurutan**. Konsekuensinya, boosting sensitif terhadap jumlah iterasi: terlalu banyak pohon membuat model menghafal data latih (_overfitting_).

Hiperparameter utama:

| Hiperparameter  | Fungsi                                                                              |
| --------------- | ----------------------------------------------------------------------------------- |
| `n_estimators`  | Jumlah pohon / iterasi (M)                                                          |
| `learning_rate` | _Shrinkage_ (ν); semakin kecil, semakin lambat belajar dan butuh lebih banyak pohon |
| `max_depth`     | Kedalaman tiap pohon                                                                |
| `subsample`     | Proporsi data per pohon (<1 = _stochastic gradient boosting_)                       |

Metrik regresi yang dipakai: **RMSE** (akar rata-rata kuadrat galat, satuan sama dengan target, menghukum galat besar), **MAE** (rata-rata galat absolut), dan **R²** (proporsi variasi target yang dijelaskan model; 0 = setara menebak rata-rata, 1 = sempurna).

## 5. Penjelasan Kode & Interpretasi Hasil

### 5.1 Import library (Sel 1)

Memuat `numpy`, `pandas`, `matplotlib`, dataset `load_diabetes`, alat pembagian data dan validasi (`train_test_split`, `cross_val_score`, `GridSearchCV`, `KFold`), tiga model (`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`), metrik (`mean_squared_error`, `mean_absolute_error`, `r2_score`), dan `permutation_importance`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
```

### 5.2 Memuat dan mengeksplorasi data (Sel 2)

```python
d = load_diabetes(as_frame=True)
X, y = d.data, d.target
print(X.shape)
print(pd.concat([X, y], axis=1).describe().round(3).T[["mean", "std", "min", "50%", "max"]])
X.assign(y=y).to_csv("diabetes.csv", index=False)
```

**Hasil:** bentuk data `(442, 10)`. Statistik target:

| Variabel | Mean    | Std    | Min      | Median | Max     |
| -------- | ------- | ------ | -------- | ------ | ------- |
| target   | 152,133 | 77,093 | 25       | 140,5  | 346     |
| 10 fitur | ≈ 0     | 0,048  | −0,138 … |        | … 0,199 |

**Interpretasi:**

- Semua fitur ber-rata-rata ≈ 0 dengan simpangan baku sama (0,048), tanda bahwa data sudah distandardisasi. Akibatnya skala fitur tidak memengaruhi perbandingan antar model.
- Target menyebar luas (25–346) dengan simpangan baku 77,09. Angka ini patokan penting: model yang hanya menebak rata-rata akan memiliki RMSE ≈ 77. Model yang baik harus berada **jauh di bawah** nilai itu.
- Median (140,5) sedikit di bawah rata-rata (152,1), menandakan sebaran sedikit miring ke kanan.

### 5.3 Pembagian data dan fungsi evaluasi (Sel 3)

```python
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluasi(model, nama):
    p = model.predict(Xte)
    print(nama, "| RMSE", ..., "| MAE", ..., "| R2", ...)
    return p
```

Data dibagi **80% latih (353 pasien)** dan **20% uji (89 pasien)**. Fungsi `evaluasi()` menghitung RMSE, MAE, dan R² pada data uji supaya semua model dinilai dengan cara yang sama.

### 5.4 Model pembanding / baseline (Sel 4)

```python
lin = LinearRegression().fit(Xtr, ytr)
rf  = RandomForestRegressor(n_estimators=500, random_state=42).fit(Xtr, ytr)
```

**Hasil (data uji):**

| Model                     | RMSE   | MAE    | R²     |
| ------------------------- | ------ | ------ | ------ |
| Regresi Linear            | 53,853 | 42,794 | 0,4526 |
| Random Forest (500 pohon) | 54,711 | 44,425 | 0,4350 |

**Interpretasi:**

- Dua model dasar ini sudah menjelaskan sekitar 44–45% variasi target dan menurunkan RMSE dari 77 (tebakan rata-rata) menjadi ± 54.
- **Regresi Linear justru sedikit lebih baik** dari Random Forest. Hal ini menandakan hubungan fitur dengan target pada dataset ini sebagian besar linear dan sinyalnya terbatas (banyak _noise_), sehingga model kompleks tidak otomatis lebih unggul. Model ini menjadi tolok ukur yang harus dilampaui oleh Gradient Boosting.

### 5.5 Gradient Boosting 500 pohon tanpa early stopping (Sel 5)

```python
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                               max_depth=3, subsample=0.8, random_state=42)
gb.fit(Xtr, ytr)
```

**Hasil:** RMSE 56,555 | MAE 45,805 | R² 0,3963

**Interpretasi:** dengan 500 pohon, Gradient Boosting justru **lebih buruk** daripada Regresi Linear dan Random Forest (R² 0,396 vs 0,453 dan 0,435). Ini gejala awal _overfitting_, dan dibuktikan pada sel berikutnya.

### 5.6 Kurva RMSE terhadap jumlah iterasi (Sel 6)

```python
rmse_tr = [np.sqrt(mean_squared_error(ytr, q)) for q in gb.staged_predict(Xtr)]
rmse_te = [np.sqrt(mean_squared_error(yte, q)) for q in gb.staged_predict(Xte)]
```

`staged_predict()` memberi prediksi setelah tiap iterasi, sehingga RMSE dapat dilacak sepanjang proses boosting.

**Hasil:**

| Iterasi | RMSE latih | RMSE uji  |
| ------- | ---------- | --------- |
| 1       | 75,94      | 71,73     |
| 50      | 44,80      | **52,23** |
| 100     | 37,22      | 52,31     |
| 200     | 28,91      | 53,49     |
| 500     | 14,92      | 56,56     |

**Interpretasi:** ini hasil terpenting notebook.

- **RMSE latih turun terus** (75,9 → 14,9) karena setiap pohon baru menghafal lebih banyak detail data latih.
- **RMSE uji turun sampai sekitar iterasi 50, lalu naik kembali** (52,2 → 56,6). Inilah titik balik _overfitting_: setelah itu model menyesuaikan _noise_ alih-alih pola umum.
- Pada iterasi 500, jurang antara galat latih (14,9) dan uji (56,6) hampir **4 kali lipat**. Model tampak hampir sempurna pada data yang dilihatnya, tetapi tidak lebih baik pada data baru.
- Pada iterasi 1, RMSE latih (75,94) hampir sama dengan simpangan baku target (≈ 77), karena langkah pertama dengan _learning rate_ kecil baru sedikit bergeser dari tebakan rata-rata.

### 5.7 Gradient Boosting dengan early stopping, model final (Sel 7)

```python
gbe = GradientBoostingRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.8,
    validation_fraction=0.15,   # 15% data latih untuk validasi internal
    n_iter_no_change=20,        # berhenti bila 20 iterasi tidak membaik
    random_state=42)
```

_Early stopping_ menyisihkan 15% data latih sebagai validasi internal dan menghentikan pelatihan jika galat validasi tidak membaik selama 20 iterasi berturut-turut. Dengan begitu jumlah pohon ditentukan otomatis tanpa menyentuh data uji.

**Hasil:** pohon terpilih **57** dari maksimum 1000 | RMSE 52,614 | MAE 42,548 | R² **0,4775**

**Interpretasi:**

- Pelatihan berhenti di 57 pohon, sangat dekat dengan titik terbaik pada kurva sebelumnya (sekitar iterasi 50–100). Ini menunjukkan _early stopping_ bekerja baik tanpa melihat data uji.
- Dibanding 500 pohon, R² naik dari 0,396 menjadi 0,478 dan RMSE turun dari 56,6 menjadi 52,6, **perbaikan nyata hanya dengan menghentikan lebih awal**.
- Model ini kini mengungguli kedua baseline (R² 0,453 dan 0,435), meski selisih dengan Regresi Linear kecil (±0,025 R²).

### 5.8 Pengaruh learning rate (Sel 8)

```python
for lr in [0.5, 0.1, 0.05, 0.01]:
    m = GradientBoostingRegressor(n_estimators=500, learning_rate=lr, max_depth=3,
                                  subsample=0.8, random_state=42).fit(Xtr, ytr)
    q = [np.sqrt(mean_squared_error(yte, z)) for z in m.staged_predict(Xte)]
```

**Hasil:**

| Learning rate | Iterasi terbaik | RMSE terbaik | RMSE pada 500 pohon |
| ------------- | --------------- | ------------ | ------------------- |
| 0,5           | 3               | 53,45        | 65,44               |
| 0,1           | 22              | 53,16        | 62,44               |
| 0,05          | 61              | **51,91**    | 56,56               |
| 0,01          | 380             | 52,81        | 53,35               |

**Interpretasi:**

- Ada pola jelas **trade-off antara learning rate dan jumlah pohon**: semakin kecil `learning_rate`, semakin banyak iterasi yang dibutuhkan (3 → 22 → 61 → 380).
- `learning_rate` besar (0,5 dan 0,1) cepat mencapai titik terbaik, tetapi kemudian **overfitting parah** (RMSE membengkak ke 65,4 dan 62,4 pada 500 pohon).
- `learning_rate` kecil (0,01) paling **tahan terhadap kelebihan pohon** (RMSE hanya 53,35 pada 500 pohon, hampir sama dengan terbaiknya), tetapi perlu banyak pohon sehingga komputasi lebih lama.
- RMSE terbaik dari keempatnya hanya berselisih sekitar 1,5 poin (51,9 – 53,5). Pilihan `learning_rate` lebih menentukan **seberapa mudah model dilatih** daripada seberapa baik hasil terbaiknya.
- Catatan: "RMSE terbaik" dipilih dengan melihat data uji, jadi angkanya **optimistis** dan tidak boleh dianggap estimasi kinerja yang jujur. Pola umumnya yang penting, bukan angka persisnya.

### 5.9 Validasi silang 5-fold (Sel 9)

```python
kf = KFold(5, shuffle=True, random_state=42)
cross_val_score(m, X, y, cv=kf, scoring="r2")   # untuk 3 model
```

Ketiga model dievaluasi dengan R² pada 5 lipatan di **seluruh data (442 pasien)**. Gradient Boosting memakai `n_estimators=100`.

**Hasil:**

| Model             | CV R² (rata-rata ± simpangan baku) |
| ----------------- | ---------------------------------- |
| Regresi Linear    | **0,4785 ± 0,0850**                |
| Random Forest     | 0,4268 ± 0,0939                    |
| Gradient Boosting | 0,4471 ± 0,0848                    |

**Interpretasi:**

- Urutannya sama dengan data uji: Regresi Linear > Gradient Boosting > Random Forest.
- Namun selisih antar model (0,03–0,05) **lebih kecil daripada simpangan baku antar lipatan (±0,085–0,094)**. Artinya perbedaan ketiga model **tidak cukup meyakinkan secara statistik**; ketiganya pada dasarnya setara pada data ini.
- Gradient Boosting memang menguasai Random Forest, tetapi tidak mengungguli Regresi Linear. Kemenangan model kompleks tidak dijamin ketika sinyal pada data terbatas.

### 5.10 Tuning hiperparameter dengan GridSearchCV (Sel 10)

```python
grid = {"n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4]}
gs = GridSearchCV(GradientBoostingRegressor(subsample=0.8, random_state=42),
                  grid, cv=5, scoring="neg_root_mean_squared_error")
gs.fit(Xtr, ytr)
```

Menguji 3 × 3 × 3 = **27 kombinasi**, masing-masing dengan 5-fold CV (135 pelatihan), pada data latih saja. Skornya `neg_root_mean_squared_error` (negatif RMSE, karena scikit-learn selalu memaksimalkan skor).

**Hasil:**

```
{'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 100} | RMSE CV 57.469
GB hasil tuning | RMSE 52.369 | MAE 42.345 | R2 0.4824
```

**Interpretasi:**

- Kombinasi terbaik: `learning_rate=0.05`, `max_depth=2`, `n_estimators=100`. Pohon **dangkal** (kedalaman 2) dipilih, sesuai dengan sifat data yang cenderung linear dengan sinyal lemah; model sederhana lebih aman dari _overfitting_.
- Pada data uji model ini mencapai **R² 0,4824 dan RMSE 52,369, terbaik di antara semua model** di notebook. Tetapi keunggulannya sangat tipis dibanding early stopping (0,4775) dan Regresi Linear (0,4526).
- RMSE CV (57,47) lebih tinggi daripada RMSE uji (52,37). Perbedaan ini wajar: CV dihitung dari lipatan di data latih, sedangkan data uji hanya 89 pasien sehingga estimasinya lebih bervariasi. Jadi **RMSE uji bisa jadi sedikit terlalu optimistis**, dan estimasi lebih jujur ada di kisaran 52–57.
- `max_depth=2` berada di **batas bawah grid**. Kemungkinan nilai yang lebih rendah (kedalaman 1, yaitu _decision stump_) belum teruji dan layak dicoba.

### 5.11 Pentingnya variabel (Sel 11)

```python
imp = pd.Series(gbe.feature_importances_, index=X.columns).sort_values(ascending=False)
pi = permutation_importance(gbe, Xte, yte, n_repeats=30, random_state=42)
```

Dua ukuran untuk model final (`gbe`):

- **MDI** (_mean decrease in impurity_): kontribusi tiap fitur dalam mengurangi galat pada data latih. Totalnya = 1.
- **Permutation importance:** penurunan **R²** pada data uji ketika nilai satu fitur diacak (30 kali).

**Hasil:**

| Fitur   | MDI        | Permutation (penurunan R²) |
| ------- | ---------- | -------------------------- |
| **bmi** | **0,4200** | 0,2501                     |
| **s5**  | 0,2449     | **0,2714**                 |
| bp      | 0,0902     | 0,0466                     |
| s6      | 0,0566     | 0,0279                     |
| s3      | 0,0481     | 0,0146                     |
| s2      | 0,0464     | −0,0097                    |
| age     | 0,0386     | 0,0107                     |
| s1      | 0,0324     | −0,0150                    |
| sex     | 0,0134     | 0,0268                     |
| s4      | 0,0094     | −0,0022                    |

**Interpretasi:**

- Kedua ukuran sepakat bahwa **`bmi` dan `s5` adalah dua fitur terpenting**. Jika salah satunya diacak, R² uji turun sekitar 0,25–0,27, padahal R² model sendiri hanya sekitar 0,48. Artinya kedua fitur ini menyumbang bagian terbesar dari kemampuan prediksi model. Secara klinis masuk akal bahwa indeks massa tubuh berkaitan dengan perkembangan diabetes.
- **`bp` (tekanan darah)** berada di urutan ketiga, dengan kontribusi jauh lebih kecil.
- Fitur `s1`, `s2`, dan `s4` memiliki permutation importance **negatif atau mendekati nol**: mengacaknya tidak menurunkan, bahkan sedikit menaikkan kinerja. Artinya fitur ini tidak membantu prediksi pada data uji (hanya _noise_). `s1` dan `s2` dikenal berkorelasi tinggi satu sama lain, sehingga informasi keduanya tumpang tindih.
- **`sex` menarik:** MDI-nya terendah kedua (0,0134), tetapi permutation importance-nya (0,0268) justru menengah. MDI cenderung meremehkan fitur biner atau berkategori sedikit, sehingga permutation importance di data uji lebih dapat dipercaya untuk kasus ini.
- Permutation importance dihitung pada hanya 89 data uji, jadi nilai kecil (mis. `age`, `s4`) sebaiknya tidak ditafsirkan terlalu rinci.

### 5.12 Visualisasi (Sel 12)

Sel ini menyimpan tiga gambar:

1. **`curve.png`**: RMSE latih (biru) dan uji (oranye) terhadap jumlah iterasi, dengan garis putus-putus abu-abu di 57 pohon (hasil _early stopping_). Menampilkan temuan di 5.6: garis latih terus turun, garis uji berbentuk "U" dan mulai naik setelah titik optimum.
2. **`imp.png`**: diagram batang horizontal MDI, memperlihatkan dominasi `bmi` dan `s5`.
3. **`avp.png`**: sebaran nilai aktual vs prediksi pada data uji, dengan garis diagonal merah putus-putus sebagai prediksi sempurna. Titik yang dekat garis berarti prediksi akurat. Dengan R² ≈ 0,48, titik-titik akan tersebar cukup lebar di sekitar garis.

### 5.13 Prediksi pasien (Sel 13)

```python
baru = Xte.iloc[:3]
print(pd.DataFrame({"aktual": yte.iloc[:3].values,
                    "prediksi": gbe.predict(baru).round(1)}))
```

**Hasil:**

| Pasien | Aktual | Prediksi | Selisih |
| ------ | ------ | -------- | ------- |
| 1      | 219,0  | 133,0    | −86,0   |
| 2      | 70,0   | 179,1    | +109,1  |
| 3      | 202,0  | 144,1    | −57,9   |

**Interpretasi:**

- Ketiga prediksi meleset cukup jauh (58–109 poin), lebih besar dari RMSE rata-rata (≈ 52). Ini wajar bagi model dengan R² ≈ 0,48.
- Pola yang terlihat: nilai tinggi (219 dan 202) **diprediksi terlalu rendah**, nilai rendah (70) **diprediksi terlalu tinggi**. Model cenderung menarik prediksi ke arah rata-rata (≈ 152), ciri umum model dengan sinyal lemah.
- Model ini cukup informatif untuk melihat kecenderungan secara kelompok, tetapi **belum cukup andal untuk memprediksi individu secara tepat**.
- Perlu diingat bahwa "pasien baru" di sini sebenarnya tiga baris pertama dari data uji, bukan data yang benar-benar baru di luar dataset.

## 6. Ringkasan Hasil

| Model                                         | RMSE uji   | MAE uji    | R² uji     |
| --------------------------------------------- | ---------- | ---------- | ---------- |
| Regresi Linear                                | 53,853     | 42,794     | 0,4526     |
| Random Forest (500 pohon)                     | 54,711     | 44,425     | 0,4350     |
| Gradient Boosting 500 pohon (overfit)         | 56,555     | 45,805     | 0,3963     |
| Gradient Boosting + early stopping (57 pohon) | 52,614     | 42,548     | 0,4775     |
| **Gradient Boosting hasil tuning**            | **52,369** | **42,345** | **0,4824** |

| Ukuran lain                 | Nilai                                                   |
| --------------------------- | ------------------------------------------------------- |
| Data latih / uji            | 353 / 89                                                |
| CV R² 5-fold (LR / RF / GB) | 0,4785 / 0,4268 / 0,4471                                |
| Hiperparameter terbaik      | `learning_rate=0.05`, `max_depth=2`, `n_estimators=100` |
| RMSE CV model tuning        | 57,469                                                  |
| Fitur terpenting            | `bmi`, `s5` (lalu `bp`)                                 |
| Fitur tidak berkontribusi   | `s1`, `s2`, `s4`                                        |

## 7. Kesimpulan

1. **Gradient Boosting sangat rentan overfitting bila jumlah iterasi tidak dikendalikan.** Dengan 500 pohon, RMSE latih 14,9 tetapi RMSE uji 56,6; setelah _early stopping_ (57 pohon) RMSE uji turun ke 52,6.
2. **Learning rate dan jumlah pohon saling menggantikan:** _learning rate_ kecil butuh lebih banyak pohon tetapi jauh lebih tahan terhadap overfitting.
3. Model terbaik adalah Gradient Boosting hasil tuning (R² uji 0,482), tetapi **keunggulannya atas Regresi Linear sangat tipis** dan di bawah ketidakpastian validasi silang (±0,085). Pada dataset ini, ketiga model pada dasarnya setara.
4. **`bmi` dan `s5`** adalah prediktor utama perkembangan diabetes; `s1`, `s2`, dan `s4` tidak menambah kemampuan prediksi.
5. Dengan R² ≈ 0,48, model menjelaskan kurang dari separuh variasi target. Ia cocok untuk gambaran umum, tidak untuk prediksi individu yang presisi.

## Referensi

- Fadil, M., Islamiyati, A., & Thamrin, S. A. (2025). Classification of nutritional status in toddlers using the support vector machine method. _Communications in Mathematical Biology and Neuroscience_, 2025, Article ID 57. https://doi.org/10.28919/cmbn/9126
- Friedman, J. H. (2001). _Greedy Function Approximation: A Gradient Boosting Machine_. Annals of Statistics, 29(5), 1189–1232.
- Friedman, J. H. (2002). _Stochastic Gradient Boosting_. Computational Statistics & Data Analysis, 38(4), 367–378.
- Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. (2004). _Least Angle Regression_. Annals of Statistics, 32(2), 407–499 (sumber dataset diabetes).
- Pedregosa, F. et al. (2011). _Scikit-learn: Machine Learning in Python_. JMLR, 12, 2825–2830.
