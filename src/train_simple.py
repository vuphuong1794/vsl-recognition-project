"""
VSL Model Trainer - Phiên bản tự động quét thư mục
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import glob

def load_data(dataset_dir):
    """Load data bằng cách quét toàn bộ thư mục"""
    X, y = [], []
    
    print(f"📂 Đang quét data tại: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Lỗi: Không tìm thấy thư mục {dataset_dir}")
        return np.array([]), np.array([])

    # Lấy danh sách tất cả các folder con
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    
    if not folders:
        print("❌ Không tìm thấy folder nào trong data/raw!")
        return np.array([]), np.array([])

    print(f"🔍 Tìm thấy {len(folders)} thư mục nhãn: {folders}")

    count_per_label = {}

    for sign_name in folders:
        sign_path = os.path.join(dataset_dir, sign_name)
        
        # Tìm tất cả file .npy trong folder đó
        sample_files = glob.glob(os.path.join(sign_path, '*.npy'))
        
        if len(sample_files) == 0:
            print(f"⚠️ Cảnh báo: Folder '{sign_name}' bị rỗng, bỏ qua.")
            continue
            
        for sample_file in sample_files:
            try:
                sequence = np.load(sample_file)
                # Kiểm tra shape để đảm bảo data không bị lỗi
                if sequence.shape == (30, 126): 
                    X.append(sequence)
                    y.append(sign_name)
                else:
                    print(f"⚠️ Bỏ qua file lỗi shape {sequence.shape}: {sample_file}")
            except Exception as e:
                print(f"❌ Lỗi đọc file {sample_file}: {e}")

        count_per_label[sign_name] = len(sample_files)
        # print(f"   + {sign_name}: {len(sample_files)} mẫu") # Bỏ comment nếu muốn log dài

    print("\n📊 Thống kê dữ liệu:")
    for label, count in count_per_label.items():
        print(f"   - {label}: {count} mẫu")

    return np.array(X), np.array(y)

def build_model(sequence_length, n_features, n_classes):
    """Build simple LSTM model"""
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, n_features)),
        
        # LSTM Layer 1
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        
        # LSTM Layer 2
        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dropout(0.2),
        
        # Dense Layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        
        # Output Layer
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
    print("VSL MODEL TRAINER (AUTO SCAN)")
    print("="*50)
    
    # 1. Xác định đường dẫn chuẩn (Absolute Path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, '../data/raw') # Trỏ ra folder data/raw
    models_dir = os.path.join(current_dir, '../models')
    
    # 2. Load data
    print("\n[1/4] Loading data...")
    X, y = load_data(dataset_dir)
    
    if len(X) == 0:
        print("\n❌ KHÔNG CÓ DATA ĐỂ TRAIN! Vui lòng chạy auto_collect_data.py trước.")
        return

    print(f"\n✅ Tổng cộng: {len(X)} mẫu")
    
    # 3. Encode labels
    print("\n[2/4] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    classes = label_encoder.classes_
    print(f"✅ Đã mã hóa {len(classes)} nhãn: {classes}")
    
    # 4. Split data
    # Stratify giúp chia đều các nhãn trong tập train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set:  {len(X_test)} samples")
    
    # 5. Build model
    print("\n[3/4] Building model...")
    model = build_model(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
        n_classes=len(classes)
    )
    model.summary()
    
    # 6. Train
    print("\n[4/4] Training...")
    
    # Callback: Dừng sớm nếu không học thêm được nữa để tiết kiệm thời gian
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100, # Tăng epoch lên vì có early stopping lo rồi
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 7. Evaluate
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # 8. Save
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'vsl_model.h5')
    encoder_save_path = os.path.join(models_dir, 'label_encoder.npy')
    
    model.save(model_save_path)
    np.save(encoder_save_path, classes)
    
    print(f"\n✓ Model saved: {model_save_path}")
    print(f"✓ Labels saved: {encoder_save_path}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)

if __name__ == '__main__':
    main()