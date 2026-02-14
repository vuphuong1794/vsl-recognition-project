import os
import tensorflow as tf
import numpy as np

# 1. Xác định đường dẫn tuyệt đối đến file script hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Tạo đường dẫn đến file model. 
# GIẢ SỬ file .h5 nằm cùng thư mục 'src' với file convert.py:
model_path = os.path.join(current_dir, 'vsl_model.h5')

# HOẶC nếu file nằm trong thư mục 'models' ngang hàng với 'src':
# model_path = os.path.join(current_dir, '../models/vsl_model.h5')

print(f"Đang tìm model tại: {model_path}")

if not os.path.exists(model_path):
    print("LỖI: Không tìm thấy file model! Hãy kiểm tra lại đường dẫn.")
    exit()

# 3. Load model với đường dẫn tuyệt đối
model = tf.keras.models.load_model(model_path)

# 2. Convert sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
]
tflite_model = converter.convert()

# 3. Lưu file .tflite
with open('vsl_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 4. Xuất danh sách nhãn (Labels) từ file .npy ra text để copy vào Android
labels_path = os.path.join(current_dir, 'label_encoder.npy')
labels = np.load(labels_path, allow_pickle=True)
print("Labels để copy vào Kotlin:", list(labels))