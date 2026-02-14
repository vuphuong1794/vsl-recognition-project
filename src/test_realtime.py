"""
VSL Real-time Tester - MediaPipe 0.10.32 Task API
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VSLTester:
    def __init__(self):
        # Find model path
        model_paths = ['../models/vsl_model.h5', 'models/vsl_model.h5']
        encoder_paths = ['../models/label_encoder.npy', 'models/label_encoder.npy']
        
        model_path = None
        encoder_path = None
        
        for p in model_paths:
            if os.path.exists(p):
                model_path = p
                break
        
        for p in encoder_paths:
            if os.path.exists(p):
                encoder_path = p
                break
        
        if model_path is None or encoder_path is None:
            print("Model not found!")
            print("Please train the model first: python src/train_simple.py")
            exit(1)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.labels = np.load(encoder_path, allow_pickle=True)
        
        # Initialize MediaPipe Hand Landmarker
        landmarker_path = 'hand_landmarker.task'
        if not os.path.exists(landmarker_path):
            print("Downloading hand landmarker model...")
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, landmarker_path)
            print("✓ Model downloaded")
        
        base_options = python.BaseOptions(model_asset_path=landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Buffer
        self.buffer = deque(maxlen=30)
        
        print("✓ Model loaded")
        print(f"✓ Signs: {self.labels}")
    
    def extract_keypoints(self, detection_result):
        """Extract 126 features"""
        keypoints = []
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints = [0] * 63
        
        while len(keypoints) < 126:
            keypoints.extend([0] * 63)
        
        return keypoints[:126]
    
    def normalize_keypoints(self, keypoints):
        """Normalize"""
        keypoints = np.array(keypoints).reshape(-1, 3)
        
        for hand_idx in range(2):
            start = hand_idx * 21
            end = start + 21
            hand_kps = keypoints[start:end]
            
            if np.sum(hand_kps) != 0:
                wrist = hand_kps[0].copy()
                hand_kps = hand_kps - wrist
                keypoints[start:end] = hand_kps
        
        return keypoints.flatten().tolist()
    
    def draw_landmarks(self, frame, detection_result):
        """Draw hand landmarks"""
        if not detection_result.hand_landmarks:
            return frame
        
        h, w, _ = frame.shape
        
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            
            for connection in connections:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
        
        return frame
    
    def predict(self):
        """Predict sign"""
        if len(self.buffer) < 30:
            return None, 0.0
        
        seq = np.array(list(self.buffer))
        seq = np.expand_dims(seq, axis=0)
        
        pred = self.model.predict(seq, verbose=0)[0]
        conf = np.max(pred)
        idx = np.argmax(pred)
        
        if conf < 0.6:
            return None, conf
        
        return self.labels[idx], conf
    
    def run(self):
        """Run real-time test"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(" Cannot open camera!")
            return
        
        print("\n" + "="*50)
        print("VSL REAL-TIME TEST")
        print("="*50)
        print("Press 'Q' to quit")
        print("="*50 + "\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Detect hands
            detection_result = self.detector.detect(mp_image)
            
            # Draw landmarks
            frame = self.draw_landmarks(frame, detection_result)
            
            # Extract & predict
            kps = self.extract_keypoints(detection_result)
            kps = self.normalize_keypoints(kps)
            self.buffer.append(kps)
            
            sign, conf = self.predict()
            
            # Draw UI
            h, w = frame.shape[:2]
            
            # Background
            cv2.rectangle(frame, (0, 0), (400, 120), (0, 0, 0), -1)
            
            if sign:
                cv2.putText(frame, sign.upper(), (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {conf:.0%}", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Waiting...", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            
            # Buffer status
            cv2.putText(frame, f"Buffer: {len(self.buffer)}/30", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Instructions
            cv2.putText(frame, "Press 'Q' to quit", 
                       (w-200, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('VSL Real-time Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Test complete")

def main():
    tester = VSLTester()
    tester.run()

if __name__ == '__main__':
    main()