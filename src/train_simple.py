"""
VSL Model Trainer - Train model đơn giản và nhanh
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import glob
from keras.callbacks import EarlyStopping

import json

def load_data(dataset_dir='../data/raw'):
    """Load data từ folder"""
    X, y = [], []
    
    # Load metadata
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Found {len(metadata['signs'])} signs: {list(metadata['signs'].keys())}")
    
    # Load data
    for sign_name in os.listdir(dataset_dir):
        sign_path = os.path.join(dataset_dir, sign_name)
        
        if not os.path.isdir(sign_path):
            continue
        
        sample_files = glob.glob(os.path.join(sign_path, '*.npy'))
        
        for sample_file in sample_files:
            sequence = np.load(sample_file)
            X.append(sequence)
            y.append(sign_name)
        
        print(f"  ✓ {sign_name}: {len(sample_files)} samples")
    
    return np.array(X), np.array(y)

def build_model(sequence_length, n_features, n_classes):
    """Build simple LSTM model"""
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(sequence_length, n_features)),
        # Giảm overfitting (tắt ngẫu nhiên 30% neuron)
        keras.layers.Dropout(0.3),
        # Fully Connected để học quan hệ phi tuyến
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("\n" + "="*50)
    print("VSL MODEL TRAINER")
    print("="*50)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    X, y = load_data(dataset_dir='../data/raw')
    print(f"✓ Loaded {len(X)} samples")
    
    # 2. mã hóa nhãn thành số 0 1 2
    print("\n[2/4] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"✓ Classes: {label_encoder.classes_}")
    
    # 3. chia data 80 20 và điều lớp
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Build model
    print("\n[3/4] Building model...")
    model = build_model(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
        n_classes=len(label_encoder.classes_)
    )
    model.summary()
    
    # 5. Train
    print("\n[4/4] Training...")
    print("-"*100)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        verbose=1,
        callbacks=[early_stop]
    )
    
    # 6. Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    #  Độ chính xác trên dữ liệu chưa từng thấy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # 7. Per-class accuracy
    print("\n" + "="*50)
    print("PER-CLASS ACCURACY")
    print("="*50)
    # Lấy class có xác suất cao nhất
    y_pred = np.argmax(model.predict(X_test), axis=1)
    # Model nhận diện MỖI DẤU tốt tới mức nào
    for i, sign in enumerate(label_encoder.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_test[mask]).mean()
            print(f"{sign:15s}: {acc*100:.1f}%")
    
    # 8. Save
    os.makedirs('models', exist_ok=True)
    model.save('models/vsl_model.h5')
    np.save('models/label_encoder.npy', label_encoder.classes_)
    
    print("\n✓ Model saved: models/vsl_model.h5")
    print("✓ Labels saved: models/label_encoder.npy")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("\nNext: Run 'python src\\test_realtime.py' to test!")

if __name__ == '__main__':
    main()