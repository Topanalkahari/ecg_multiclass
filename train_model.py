# train_model.py

import joblib
import pandas as pd
from ecg_pipeline import ECGPipeline

# Inisialisasi pipeline
pipeline = ECGPipeline()

# Path dataset
root_dataset = "dataset"  # Ganti sesuai dengan path dataset kamu

# Proses dataset terlabel
features_df, labels = pipeline.process_labeled_dataset(root_dataset)

# Training dan simpan model
if features_df is not None:
    print("\n✅ Melatih model...")
    pipeline.train_and_predict(features_df, labels)
    
    # Simpan model dan scaler
    #joblib.dump(pipeline.predictor.model, "ecg_model.pkl")
    #joblib.dump(pipeline.predictor.scaler, "ecg_scaler.pkl")
    #print("✅ Model dan scaler berhasil disimpan!")
else:
    print("❌ Dataset tidak ditemukan atau kosong.")
