"""
VSL Real-time Tester - GCN Version
Sử dụng mô hình ST-GCN đã huấn luyện để nhận diện thời gian thực.
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VSLGCNTester:
    def __init__(self):
        # 1. Load Model & Labels
        self._load_resources()
        
        # 2. Initialize MediaPipe Detectors
        print("Initializing MediaPipe Holistic...")
        self._init_detectors()
        
        # Buffer: Lưu 30 frames gần nhất
        self.buffer = deque(maxlen=30)
        
        print("✓ Ready to start!")

    def _load_resources(self):
        # Tìm model GCN
        model_path = '../models/best_gcn_model.h5'
        if not os.path.exists(model_path):
            # Thử đường dẫn local nếu chạy từ thư mục src
            model_path = 'models/best_gcn_model.h5'
            
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}!")
            print("Please run train_gcn.py first.")
            exit(1)
            
        # Tìm label encoder
        encoder_path = os.path.join(os.path.dirname(model_path), 'label_encoder_gcn.npy')
        if not os.path.exists(encoder_path):
            print(f"❌ Label encoder not found at {encoder_path}!")
            exit(1)

        print(f"Loading GCN model from: {model_path}")
        
        # Load model với custom layer GraphConv và STGCN_Block
        # Cần định nghĩa lại hoặc dùng custom_objects nếu load model đã save full
        # Tuy nhiên, Keras save format .h5 thường lưu cả kiến trúc.
        # Nếu gặp lỗi custom layer, ta cần import class từ train_gcn
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print("⚠️ Loading custom model directly failed. Trying with custom_objects...")
            # Định nghĩa lại custom layers (cần giống hệt train_gcn.py)
            # Trong thực tế nên tách layers ra file riêng để import.
            # Ở đây ta giả định model load được hoặc cần copy class vào đây.
            # Để đơn giản, ta sẽ import từ train_gcn nếu file đó nằm cùng thư mục
            try:
                from train_gcn import GraphConv, STGCN_Block
                self.model = tf.keras.models.load_model(model_path, 
                                custom_objects={'GraphConv': GraphConv, 'STGCN_Block': STGCN_Block})
            except ImportError:
                print("❌ Could not import custom layers from train_gcn.py")
                exit(1)

        self.labels = np.load(encoder_path, allow_pickle=True)
        print(f"✓ Labels loaded: {self.labels}")

    def _init_detectors(self):
        # Download models
        models = {
            'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        }
        import urllib.request
        for name, url in models.items():
            if not os.path.exists(name):
                print(f"Downloading {name}...")
                try:
                    urllib.request.urlretrieve(url, name)
                except: pass

        # Hand
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2, min_hand_detection_confidence=0.5)
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

        # Pose
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(base_options=base_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        
        # Face (Cần thiết để đồng bộ với logic collect, dù GCN có thể không dùng hết)
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        self.face_detector = vision.FaceLandmarker.create_from_options(options)

    def extract_full_keypoints(self, hand_result, face_result, pose_result):
        """
        Trích xuất toàn bộ 1659 điểm (giống hệt lúc collect).
        Sau đó sẽ lọc lấy 75 điểm cho GCN.
        """
        keypoints = []
        
        # 1. Pose (99)
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0] * 99)
            
        # 2. Face (1434)
        if face_result.face_landmarks:
            for lm in face_result.face_landmarks[0]:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0] * 1434)
            
        # 3. Hands (126)
        hand_kps = []
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for lm in hand_landmarks:
                    hand_kps.extend([lm.x, lm.y, lm.z])
        while len(hand_kps) < 126:
            hand_kps.extend([0] * 63)
        keypoints.extend(hand_kps[:126])
        
        return np.array(keypoints)

    def preprocess_for_gcn(self, buffer_data):
        """
        Chuyển đổi buffer (30, 1659) -> Input GCN (1, 30, 75, 3)
        """
        # 1. Lấy chuỗi raw
        seq = np.array(buffer_data) # (30, 1659)
        
        # 2. Trích xuất 75 điểm quan trọng
        # Pose: 0-99 (33 điểm * 3)
        pose = seq[:, 0:99]
        # Hands: 1533-1659 (42 điểm * 3)
        hands = seq[:, 1533:1659]
        
        # Gộp lại: (30, 225)
        skeleton = np.concatenate([pose, hands], axis=1)
        
        # 3. Reshape thành (1, 30, 75, 3)
        # 1: Batch size
        # 30: Frames
        # 75: Nodes
        # 3: Channels (x, y, z)
        gcn_input = skeleton.reshape(1, 30, 75, 3)
        
        return gcn_input

    def draw_debug(self, frame, pose_result, hand_result):
        """Vẽ khung xương đơn giản"""
        h, w, _ = frame.shape
        if pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks[0]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 255), -1)
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("❌ Cannot open camera")
                return

        print("\n=== VSL GCN REAL-TIME TEST ===")
        print("Press 'Q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Detect
            hand_res = self.hand_detector.detect(mp_image)
            face_res = self.face_detector.detect(mp_image) # Cần chạy để giữ đồng bộ index
            pose_res = self.pose_detector.detect(mp_image)
            
            # Draw
            frame = self.draw_debug(frame, pose_res, hand_res)
            
            # Process
            kps = self.extract_full_keypoints(hand_res, face_res, pose_res)
            self.buffer.append(kps)
            
            sign = ""
            conf = 0.0
            
            if len(self.buffer) == 30:
                # Prepare input for GCN
                gcn_input = self.preprocess_for_gcn(self.buffer)
                
                # Predict
                pred = self.model.predict(gcn_input, verbose=0)[0]
                conf = np.max(pred)
                idx = np.argmax(pred)
                
                if conf > 0.7: # Ngưỡng tin cậy cao hơn cho GCN
                    sign = self.labels[idx]
            
            # UI
            cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)
            if sign:
                cv2.putText(frame, f"GCN: {sign.upper()}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {conf:.1%}", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Waiting...", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                
            cv2.putText(frame, f"Buffer: {len(self.buffer)}/30", (20, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            cv2.imshow('VSL GCN Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tester = VSLGCNTester()
    tester.run()
