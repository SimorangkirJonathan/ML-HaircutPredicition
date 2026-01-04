# Real-Time Hairstyle Recommendation System

Sistem rekomendasi gaya rambut real-time menggunakan kamera yang mendeteksi bentuk wajah dan tipe rambut secara otomatis, kemudian memberikan rekomendasi potongan rambut yang cocok.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green)

---

## Daftar Isi

- [Fitur](#fitur)
- [Teknologi](#teknologi)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Dataset & Training](#dataset--training)
- [Struktur Proyek](#struktur-proyek)
- [Kontrol Aplikasi](#kontrol-aplikasi)
- [Troubleshooting](#troubleshooting)

---

## Fitur

### Deteksi Real-Time
- **Deteksi Wajah** - Menggunakan OpenCV Haar Cascade
- **68 Facial Landmarks** - OpenCV Facemark LBF untuk pengukuran akurat
- **Temporal Smoothing** - Prediksi stabil (rata-rata 15 frame)

### Klasifikasi
- **Face Shape Detection** (6 kategori):
  - Oval, Round, Square, Heart, Oblong, Diamond
  
- **Hair Type Detection** (5 kategori):
  - Straight, Wavy, Curly, Kinky, Dreadlocks

### Rekomendasi
- **30 Kombinasi Gaya Rambut** (6 face shapes × 5 hair types)
- Tips styling dalam Bahasa Indonesia
- UI informatif dengan hasil real-time

---

## Teknologi

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3.10 |
| Deep Learning | TensorFlow/Keras |
| Computer Vision | OpenCV 4.12 |
| Face Detection | Haar Cascade |
| Facial Landmarks | OpenCV Facemark LBF (68 points) |
| CNN Architecture | MobileNetV2 (Transfer Learning) |

---

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/hairstyle-recommendation.git
cd hairstyle-recommendation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
```
opencv-python
opencv-contrib-python
tensorflow
numpy
Pillow
scikit-learn
```

### 3. Download Facial Landmark Model
Model LBF akan otomatis download, atau manual:
```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml" -OutFile "models/lbfmodel.yaml"
```

### 4. Training Model (Opsional)
Jika ingin melatih ulang model hair type:
```bash
python train_hair_model.py
```

---

## Penggunaan

### Web Interface (Streamlit) - Rekomendasi
UI modern berbasis web dengan tampilan profesional:
```bash
pip install streamlit streamlit-webrtc
streamlit run app_streamlit.py
```

Browser akan otomatis terbuka. Klik **START** untuk memulai kamera.

### Classic Mode (OpenCV)
```bash
python main.py
```

### Kontrol Keyboard (Classic Mode)
| Tombol | Fungsi |
|--------|--------|
| `Q` | Keluar dari aplikasi |
| `S` | Simpan screenshot |
| `L` | Toggle tampilan landmarks |
| `H` | Toggle help overlay |


---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                      CAMERA INPUT                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FACE DETECTION                           │
│              (Haar Cascade + Facemark LBF)                  │
│                                                             │
│   ┌─────────────────────┐    ┌─────────────────────┐       │
│   │   Face Bounding Box │    │  68 Facial Landmarks │       │
│   └─────────────────────┘    └─────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   FACE SHAPE CLASSIFIER │     │   HAIR TYPE CLASSIFIER  │
│     (Rule-based)        │     │   (CNN - MobileNetV2)   │
│                         │     │                         │
│  Geometric Ratios:      │     │  Input: Hair Region     │
│  • Length/Width         │     │  Output: 5 classes      │
│  • Forehead/Jaw         │     │  Accuracy: 85%          │
│  • Jaw/Face             │     │                         │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   TEMPORAL SMOOTHING                        │
│              (15-frame voting window)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                HAIRSTYLE RECOMMENDER                        │
│           30 kombinasi gaya rambut + tips                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      UI DISPLAY                             │
│    Face Shape │ Hair Type │ Recommended Styles │ Tips      │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset & Training

### Dataset Hair Type
```
data/
├── Straight/     (XXX images)
├── Wavy/         (XXX images)
├── Curly/        (XXX images)
├── Kinky/        (XXX images)
└── Dreadlocks/   (XXX images)
```
Total: ~2000 images

### Training Configuration
```python
# Anti-Overfitting Techniques:
- Dropout: 50% (decreasing per layer)
- L2 Regularization: 0.01
- Gaussian Noise: 0.1
- Data Augmentation:
  • Rotation: ±30°
  • Shift: ±25%
  • Zoom: ±25%
  • Brightness: 0.7-1.3
  • Horizontal Flip: Yes

# Training Strategy:
- Phase 1: Frozen base (15 epochs)
- Phase 2: Fine-tuning last 20 layers (40 epochs)
- Early Stopping: patience=10
- Learning Rate Reduction: factor=0.5, patience=5
```

### Model Performance
| Metric | Value |
|--------|-------|
| Validation Accuracy | 85.06% |
| Architecture | MobileNetV2 + Custom Head |
| Input Size | 224 × 224 × 3 |

---

## Struktur Proyek

```
hairstyle-recommendation/
│
├── main.py                 # Entry point aplikasi
├── train_hair_model.py     # Script training CNN
├── requirements.txt        # Dependencies
├── README.md               # Dokumentasi
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── camera.py           # Camera capture
│   ├── face_detector.py    # Face & landmark detection
│   ├── face_shape_classifier.py  # Face shape rules
│   ├── hair_classifier.py  # CNN hair classifier
│   └── recommender.py      # Recommendation engine
│
├── models/                 # Trained models
│   ├── hair_type_model.keras   # Hair classifier
│   ├── lbfmodel.yaml           # Facial landmarks
│   └── class_mapping.txt       # Class labels
│
└── data/                   # Training dataset
    ├── Straight/
    ├── Wavy/
    ├── Curly/
    ├── Kinky/
    └── Dreadlocks/
```

---

## Kontrol Aplikasi

### Tampilan UI
```
┌─────────────────────────────────────────┬──────────────────┐
│                                         │    RESULTS       │
│                                         │                  │
│         [HAIR REGION]                   │  Face: Oval 80%  │
│         ┌─────────────┐                 │  Hair: Wavy 75%  │
│         │             │                 │                  │
│         │   68 MARKS  │                 │  Styles:         │
│         │             │                 │  1. Textured Crop│
│         │             │                 │  2. Messy Fringe │
│         │             │                 │  3. Natural Waves│
│         └─────────────┘                 │                  │
│                                         │                  │
├─────────────────────────────────────────┴──────────────────┤
│  Q: Quit | S: Screenshot | L: Landmarks | H: Help          │
└────────────────────────────────────────────────────────────┘
```

### Landmarks Legend
- **Merah** - Jawline (rahang)
- **Kuning** - Alis
- **Biru** - Mata
- **Hijau** - Hidung
- **Pink** - Mulut

---

## Troubleshooting

### Model tidak terload
```bash
# Retrain model
python train_hair_model.py
```

### Camera tidak terbuka
- Pastikan tidak ada aplikasi lain yang menggunakan camera
- Coba restart aplikasi

### Facial landmarks tidak muncul
```bash
# Download ulang model LBF
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml" -OutFile "models/lbfmodel.yaml"
```

### Deteksi tidak stabil
- Pastikan pencahayaan cukup
- Posisikan wajah di tengah frame
- Jangan terlalu jauh dari kamera

---

## Author

**Jonathan Simorangkir**

---

## Acknowledgments

- MobileNetV2 - Google Research
- OpenCV Facemark LBF - Kok Wei Chee
- TensorFlow/Keras Team