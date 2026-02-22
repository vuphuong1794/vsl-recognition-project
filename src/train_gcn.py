"""
VSL Graph Convolutional Network (GCN) Trainer
Sử dụng kiến trúc ST-GCN (Spatio-Temporal Graph Convolutional Networks)
Phù hợp cho nhận diện dựa trên Skeleton (MediaPipe Holistic).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. ĐỊNH NGHĨA GRAPH (Cấu trúc xương)
# ==========================================
def get_adjacency_matrix(num_nodes):
    """
    Tạo ma trận kề (Adjacency Matrix) A biểu diễn kết nối các khớp.
    MediaPipe Holistic (Pose 33 + Face 468 + Hands 21x2) quá lớn.
    Ở đây ta sẽ tập trung vào các điểm quan trọng (Key Keypoints) để GCN hiệu quả:
    - Pose: 33 điểm
    - Hands: 21x2 = 42 điểm
    Tổng: 75 điểm quan trọng (Bỏ qua Face dày đặc để giảm tính toán)
    """
    # Danh sách các kết nối (Edge) dựa trên MediaPipe Pose & Hand topology
    # Cần map lại index từ vector 1659 điểm gốc về 75 điểm chọn lọc.
    # Tuy nhiên, để đơn giản cho demo này, ta sẽ dùng "Learnable Adjacency Matrix" 
    # hoặc coi như full-connected graph có trọng số học được.
    
    # Ở đây dùng A matrix đơn vị + Learnable Mask (A_adaptive) trong layer GCN
    # Return None để model tự học cấu trúc (Adaptive Graph)
    return None

# ==========================================
# 2. XÂY DỰNG DATA LOADER
# ==========================================
def load_data_gcn(dataset_dir):
    """
    Load data và reshape cho GCN.
    Input gốc: (N, 30, 1659) -> (Sequence, Features)
    GCN cần tách tọa độ (x,y,z) ra khỏi số lượng node.
    
    MediaPipe Holistic flatten: 
    - Pose: 0-98 (33 points * 3)
    - Face: 99-1532 (478 points * 3) -> Sẽ bỏ qua hoặc giảm chiều
    - Left Hand: 1533-1595 (21 points * 3)
    - Right Hand: 1596-1658 (21 points * 3)
    
    Ta sẽ trích xuất 75 điểm quan trọng: Pose(33) + LHand(21) + RHand(21) = 75 points
    Shape đích: (N, Frames, Nodes, Channels) = (N, 30, 75, 3)
    """
    X, y = [], []
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    
    print("🔍 Đang tải dữ liệu và tái cấu trúc cho GCN...")
    
    for sign_name in folders:
        sign_path = os.path.join(dataset_dir, sign_name)
        files = glob.glob(os.path.join(sign_path, '*.npy'))
        
        for f in files:
            try:
                seq = np.load(f) # Shape (30, 1659)
                if seq.shape != (30, 1659): continue
                
                # --- TRÍCH XUẤT KEYPOINTS QUAN TRỌNG ---
                # 1. Pose: 33 điểm đầu (index 0-98)
                pose = seq[:, 0:99]
                
                # 2. Hands: 42 điểm cuối (index 1533-1659)
                hands = seq[:, 1533:1659]
                
                # Gộp lại: (30, 99 + 126) = (30, 225) -> tương ứng 75 điểm * 3
                skeleton = np.concatenate([pose, hands], axis=1)
                
                # Reshape: (30, 75, 3) -> (Frames, Nodes, Channels)
                skeleton_reshaped = skeleton.reshape(30, 75, 3)
                
                X.append(skeleton_reshaped)
                y.append(sign_name)
            except:
                pass
                
    return np.array(X), np.array(y)

# ==========================================
# 3. MÔ HÌNH GCN (Graph Conv)
# ==========================================
class GraphConv(layers.Layer):
    """Lớp GCN cơ bản với ma trận kề học được"""
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        # input_shape: (Batch, Frame, Node, Channel)
        self.nodes = input_shape[2]
        self.in_channels = input_shape[3]
        
        # Learnable Adjacency Matrix (A) size (Node, Node)
        self.A = self.add_weight(
            name="adjacency_matrix",
            shape=(self.nodes, self.nodes),
            initializer="uniform",
            trainable=True
        )
        
        # Weight matrix W size (Channel_in, Channel_out)
        self.W = self.add_weight(
            name="weight_matrix",
            shape=(self.in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        # inputs: (B, T, V, C)
        # 1. Graph Convolution: X' = A * X * W
        # Thực hiện phép nhân A * X trước: (V, V) * (B, T, V, C) -> (B, T, V, C)
        # Sử dụng einsum cho linh hoạt: 'vw, btv c -> btwc'
        x = tf.einsum('vw,btwc->btvc', self.A, inputs)
        
        # 2. Nhân với trọng số W: (B, T, V, C_in) * (C_in, C_out)
        x = tf.matmul(x, self.W)
        
        return tf.nn.relu(x)

class STGCN_Block(layers.Layer):
    """Khối Spatio-Temporal: GCN (Không gian) + TCN (Thời gian)"""
    def __init__(self, out_channels, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.gcn = GraphConv(out_channels)
        self.tcn = layers.Conv2D(out_channels, kernel_size=(9, 1), padding='same', activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()
        self.residual = layers.Conv2D(out_channels, kernel_size=(1, 1), padding='same')

    def call(self, inputs):
        # 1. Spatial GCN
        x = self.gcn(inputs)
        
        # 2. Temporal CNN (Conv trên trục thời gian Frame)
        x = self.tcn(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        # Residual connection
        res = self.residual(inputs)
        return layers.Add()([x, res])

def build_st_gcn_model(input_shape, num_classes):
    """Xây dựng mô hình ST-GCN hoàn chỉnh"""
    inputs = layers.Input(shape=input_shape) # (30, 75, 3)
    
    # Data normalization
    x = layers.BatchNormalization()(inputs)
    
    # ST-GCN Blocks
    x = STGCN_Block(64)(x)
    x = STGCN_Block(64)(x)
    x = STGCN_Block(128)(x)
    x = STGCN_Block(128)(x)
    x = STGCN_Block(256)(x)
    
    # Global Pooling
    # Pool theo thời gian và node để ra vector đặc trưng
    x = layers.GlobalAveragePooling2D()(x) 
    
    # Classification Head
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="VSL_ST_GCN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 4. HUẤN LUYỆN
# ==========================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, '../data/raw')
    models_dir = os.path.join(current_dir, '../models')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    X, y = load_data_gcn(dataset_dir)
    
    if len(X) == 0:
        print("❌ Không tìm thấy dữ liệu hợp lệ!")
        return

    print(f"✅ Data shape: {X.shape}") # (N, 30, 75, 3)
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"🏷️ Classes: {classes}")
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    # 4. Build Model
    input_shape = (30, 75, 3) # (Frames, Nodes, Channels)
    model = build_st_gcn_model(input_shape, len(classes))
    model.summary()
    
    # 5. Train
    print("\n🚀 Bắt đầu huấn luyện GCN...")
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(models_dir, 'best_gcn_model.h5'),
        save_best_only=True, monitor='val_accuracy'
    )
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=16,
        callbacks=[checkpoint, early_stop]
    )
    
    # 6. Evaluate
    print("\n📊 Đánh giá mô hình...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save labels
    np.save(os.path.join(models_dir, 'label_encoder_gcn.npy'), classes)
    
    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(current_dir, '../results/gcn_training_history.png'))
    print("✅ Đã lưu biểu đồ training.")

if __name__ == '__main__':
    main()
