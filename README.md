# 🍅 Tomato Leaf Disease Classifier

A deep learning web app that detects tomato leaf diseases using **MobileNetV2** transfer learning.

## 🎯 Classes

| Class | Description |
|-------|-------------|
| ✅ Healthy | No disease detected |
| ⚠️ Early Blight | *Alternaria solani* fungal infection |
| 🔴 Septoria Leaf Spot | *Septoria lycopersici* fungal infection |

## 🛠️ Tech Stack

- **Model:** MobileNetV2 (ImageNet pretrained, fine-tuned)
- **Dataset:** PlantVillage — 500 images/class (1500 total)
- **Framework:** TensorFlow / Keras
- **App:** Streamlit
- **Augmentation:** Random flip, brightness, contrast, saturation + Custom Black Patch Masking

## 📁 Project Structure

```
tomato-disease-app/
├── app.py                  # Streamlit application
├── best_model.keras        # Trained model (add manually — see below)
├── requirements.txt        # Python dependencies
├── notebook.ipynb          # Training notebook (Colab)
└── README.md
```

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/tomato-disease-app.git
cd tomato-disease-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your trained model
# Place best_model.keras in the root folder

# 4. Run the app
streamlit run app.py
```

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 |
| Input Size | 224 × 224 × 3 |
| Head | GAP → BN → Dense(128) → Dropout(0.3) → Softmax(3) |
| Optimizer | Adam (lr=1e-3) |
| Loss | Sparse Categorical Crossentropy |
| Seed | 7282 |

## ⚙️ Adding the Model

The `.keras` model file is **not** tracked by Git (too large).  
Options to add it:

1. **Git LFS** — `git lfs track "*.keras"`
2. **Google Drive** — upload and link in README
3. **Hugging Face Hub** — host for free at `hf.co`

## 📊 Training Results

- Train/Val/Test split: 70% / 15% / 15%
- Augmentation: 5 standard + 1 custom (black patch masking)
- TTA: 6-fold at inference

## 🙏 Credits

Dataset: [PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) via TensorFlow Datasets
