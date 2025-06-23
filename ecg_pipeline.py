# ECG Prediction System
# PNG/JPG -> CSV -> Model Prediction

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.signal import find_peaks
import os
import warnings
warnings.filterwarnings('ignore')
import joblib

class ECGImageProcessor:
    """
    Class untuk mengkonversi gambar ECG menjadi data CSV
    """
    
    def __init__(self):
        self.sampling_rate = 500  # Hz
        self.duration = 10  # seconds
        
    def preprocess_image(self, image_path):
        """
        Preprocessing gambar ECG
        """
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
            
        # Convert ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold untuk mendapatkan sinyal ECG
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Noise reduction
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned, gray
    
    def extract_ecg_signal(self, processed_img):
        """
        Ekstrak sinyal ECG dari gambar yang sudah diproses
        """
        height, width = processed_img.shape
        
        # Mencari kontur untuk mendapatkan garis ECG
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Jika tidak ada kontur, gunakan metode sederhana
            return self._simple_signal_extraction(processed_img)
        
        # Ambil kontur terbesar (diasumsikan sebagai sinyal ECG utama)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Ekstrak koordinat
        points = []
        for point in largest_contour:
            x, y = point[0]
            points.append((x, height - y))  # Flip Y coordinate
        
        # Sort berdasarkan x
        points.sort(key=lambda p: p[0])
        
        if len(points) < 10:
            return self._simple_signal_extraction(processed_img)
        
        # Interpolasi untuk mendapatkan sampling yang konsisten
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Resample ke sampling rate yang diinginkan
        target_length = self.sampling_rate * self.duration
        x_new = np.linspace(min(x_coords), max(x_coords), target_length)
        y_new = np.interp(x_new, x_coords, y_coords)
        
        # Normalize
        y_normalized = (y_new - np.mean(y_new)) / (np.std(y_new) + 1e-8)
        
        return y_normalized
    
    def _simple_signal_extraction(self, processed_img):
        """
        Metode sederhana untuk ekstrak sinyal jika kontur tidak berhasil
        """
        height, width = processed_img.shape
        
        # Ambil rata-rata kolom untuk setiap baris
        signal_data = []
        for x in range(width):
            column = processed_img[:, x]
            # Cari posisi piksel putih (sinyal)
            white_pixels = np.where(column > 0)[0]
            if len(white_pixels) > 0:
                # Ambil posisi tengah
                avg_y = np.mean(white_pixels)
                signal_data.append(height - avg_y)  # Flip Y
            else:
                signal_data.append(height // 2)  # Default ke tengah
        
        # Resample
        target_length = self.sampling_rate * self.duration
        if len(signal_data) > 0:
            x_old = np.linspace(0, 1, len(signal_data))
            x_new = np.linspace(0, 1, target_length)
            y_new = np.interp(x_new, x_old, signal_data)
            
            # Normalize
            y_normalized = (y_new - np.mean(y_new)) / (np.std(y_new) + 1e-8)
            return y_normalized
        else:
            # Return dummy signal
            return np.random.randn(target_length) * 0.1
    
    def image_to_csv(self, image_path, output_csv=None):
        """
        Convert gambar ECG ke CSV
        """
        try:
            # Preprocess image
            processed_img, original = self.preprocess_image(image_path)
            
            # Extract signal
            ecg_signal = self.extract_ecg_signal(processed_img)
            
            # Create time axis
            time_axis = np.linspace(0, self.duration, len(ecg_signal))
            
            # Create DataFrame
            df = pd.DataFrame({
                'time': time_axis,
                'ecg_signal': ecg_signal,
                'filename': os.path.basename(image_path)
            })
            
            # Save to CSV if path provided
            if output_csv:
                df.to_csv(output_csv, index=False)
                print(f"Saved ECG data to {output_csv}")
            
            return df
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

class ECGFeatureExtractor:
    """
    Class untuk ekstrak fitur dari sinyal ECG
    """
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def extract_features(self, ecg_signal):
        """
        Ekstrak berbagai fitur dari sinyal ECG
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(ecg_signal)
        features['std'] = np.std(ecg_signal)
        features['var'] = np.var(ecg_signal)
        features['min'] = np.min(ecg_signal)
        features['max'] = np.max(ecg_signal)
        features['range'] = features['max'] - features['min']
        features['skewness'] = self._calculate_skewness(ecg_signal)
        features['kurtosis'] = self._calculate_kurtosis(ecg_signal)
        
        # Peak detection features
        peaks, peak_properties = find_peaks(ecg_signal, height=0.1, distance=self.sampling_rate//3)
        features['num_peaks'] = len(peaks)
        
        if len(peaks) > 1:
            # Heart rate estimation
            peak_intervals = np.diff(peaks) / self.sampling_rate  # dalam detik
            features['avg_rr_interval'] = np.mean(peak_intervals)
            features['std_rr_interval'] = np.std(peak_intervals)
            features['heart_rate'] = 60 / features['avg_rr_interval'] if features['avg_rr_interval'] > 0 else 0
            features['hrv'] = features['std_rr_interval']
        else:
            features['avg_rr_interval'] = 0
            features['std_rr_interval'] = 0
            features['heart_rate'] = 0
            features['hrv'] = 0
        
        # Frequency domain features
        freqs, psd = signal.welch(ecg_signal, fs=self.sampling_rate, nperseg=1024)
        
        # Power in different frequency bands
        features['power_vlf'] = self._power_in_band(freqs, psd, 0.003, 0.04)  # Very Low Frequency
        features['power_lf'] = self._power_in_band(freqs, psd, 0.04, 0.15)   # Low Frequency
        features['power_hf'] = self._power_in_band(freqs, psd, 0.15, 0.4)    # High Frequency
        features['lf_hf_ratio'] = features['power_lf'] / (features['power_hf'] + 1e-8)
        
        # Spectral features
        features['dominant_freq'] = freqs[np.argmax(psd)]
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        
        return features
    
    def _calculate_skewness(self, data):
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def _power_in_band(self, freqs, psd, low_freq, high_freq):
        """Calculate power in frequency band"""
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.trapz(psd[band_mask], freqs[band_mask])

class ECGPredictor:
    """
    Class untuk prediksi berdasarkan fitur ECG
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()
        self.feature_extractor = ECGFeatureExtractor()
        self.is_trained = False
        
    def prepare_features_from_csv(self, csv_data):
        """
        Persiapkan fitur dari data CSV
        """
        if isinstance(csv_data, str):
            df = pd.read_csv(csv_data)
        else:
            df = csv_data
        
        # Group by filename jika ada multiple files
        if 'filename' in df.columns:
            grouped = df.groupby('filename')
            features_list = []
            
            for filename, group in grouped:
                ecg_signal = group['ecg_signal'].values
                features = self.feature_extractor.extract_features(ecg_signal)
                features['filename'] = filename
                features_list.append(features)
            
            return pd.DataFrame(features_list)
        else:
            # Single ECG signal
            ecg_signal = df['ecg_signal'].values
            features = self.feature_extractor.extract_features(ecg_signal)
            return pd.DataFrame([features])
    
    def create_dummy_labels(self, features_df):
        """
        Buat label dummy untuk demonstrasi
        Dalam implementasi nyata, Anda perlu label yang sebenarnya
        """
        np.random.seed(42)
        n_samples = len(features_df)
        
        # Simulasi label berdasarkan heart rate
        labels = []
        for _, row in features_df.iterrows():
            hr = row.get('heart_rate', 70)
            if hr < 60:
                label = 'bradycardia'  # Detak jantung lambat
            elif hr > 100:
                label = 'tachycardia'  # Detak jantung cepat
            else:
                label = 'normal'
            labels.append(label)
        
        return np.array(labels)
    
    def train(self, features_df, labels=None):
        """
        Train model
        """
        if labels is None:
            labels = self.create_dummy_labels(features_df)
        
        # Hapus kolom non-numerik
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            numeric_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        print("Training completed!")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        self.is_trained = True
        return X_test, y_test, y_pred
    
    def predict(self, features_df):
        """
        Prediksi berdasarkan fitur
        """
        if not self.is_trained:
            raise ValueError("Model belum ditraining. Jalankan train() terlebih dahulu.")
        
        # Hapus kolom non-numerik
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_features = numeric_features.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(numeric_features)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

class ECGPipeline:
    """
    Pipeline lengkap untuk ECG prediction
    """
    
    def __init__(self):
        self.image_processor = ECGImageProcessor()
        self.predictor = ECGPredictor()
    
    def process_single_image(self, image_path, save_csv=True):
        """
        Proses single image dari awal hingga prediksi
        """
        print(f"Processing {image_path}...")
        
        # Step 1: Convert image to CSV
        csv_data = self.image_processor.image_to_csv(image_path)
        
        if csv_data is None:
            return None
        
        if save_csv:
            csv_filename = image_path.replace('.png', '.csv').replace('.jpg', '.csv').replace('.jpeg', '.csv')
            csv_data.to_csv(csv_filename, index=False)
            print(f"CSV saved as {csv_filename}")
        
        # Step 2: Extract features
        features_df = self.predictor.prepare_features_from_csv(csv_data)
        
        return csv_data, features_df
    
    def process_folder(self, folder_path, output_folder=None):
        """
        Proses semua gambar dalam folder
        """
        if output_folder is None:
            output_folder = folder_path
        
        all_csv_data = []
        all_features = []
        
        # Supported image formats
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext.lower())])
        
        if not image_files:
            print(f"No image files found in {folder_path}")
            return None, None
        
        print(f"Found {len(image_files)} image files")
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            csv_data, features_df = self.process_single_image(image_path, save_csv=False)
            
            if csv_data is not None and features_df is not None:
                all_csv_data.append(csv_data)
                all_features.append(features_df)
        
        if all_csv_data:
            # Combine all CSV data
            combined_csv = pd.concat(all_csv_data, ignore_index=True)
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # Save combined data
            combined_csv.to_csv(os.path.join(output_folder, 'combined_ecg_data.csv'), index=False)
            combined_features.to_csv(os.path.join(output_folder, 'combined_features.csv'), index=False)
            
            print(f"Combined data saved to {output_folder}")
            return combined_csv, combined_features
        
        return None, None
    
    def train_and_predict(self, features_df, labels=None):
        """
        Train model dan lakukan prediksi
        """
        print("Training model...")
        X_test, y_test, y_pred = self.predictor.train(features_df, labels)
        
        return X_test, y_test, y_pred
    
    def predict_new_image(self, image_path):
        """
        Prediksi untuk gambar baru
        """
        if not self.predictor.is_trained:
            raise ValueError("Model belum ditraining!")
        
        csv_data, features_df = self.process_single_image(image_path, save_csv=False)
        
        if csv_data is None or features_df is None:
            return None, None
        
        predictions, probabilities = self.predictor.predict(features_df)
        
        return predictions, probabilities

    def process_labeled_dataset(self, root_folder):
        """
        Proses folder dataset utama yang berisi subfolder per kelas
        """
        all_features = []
        all_labels = []

        for subfolder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue  # Hanya folder

            label = self._infer_label_from_folder(subfolder)
            print(f"\n📁 Memproses folder: {subfolder} --> Label: {label}")

            _, features_df = self.process_folder(subfolder_path)

            if features_df is not None:
                features_df['label'] = label
                all_features.append(features_df)
                all_labels.extend([label] * len(features_df))

        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            print(f"\n✅ Total sampel diproses: {len(combined_features)}")
            return combined_features, np.array(all_labels)
        else:
            print("❌ Tidak ada fitur yang berhasil diekstrak dari folder.")
            return None, None

    def _infer_label_from_folder(self, folder_name):
        """
        Tentukan label dari nama folder (ubah sesuai konvensi nama folder kamu)
        """
        name = folder_name.lower()
        if "abnormal" in name:
            return "Abnormal"
        elif "normal" in name:
            return "Normal"
        elif "history" in name:
            return "History of MI"
        elif "infarction" in name:
            return "Myocardial Infarction"
        else:
            return "Unknown"