"""
VSL Data Collector - MediaPipe 0.10.32+ (Task API)
"""

import cv2
import numpy as np
import os
import json
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VSLDataCollector:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing MediaPipe 0.10.32 Task API...")
        
        # Download hand landmarker model if not exists
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("✓ Model downloaded")
        
        # Create HandLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Metadata
        self.metadata = {
            'signs': {},
            'total_samples': 0
        }
        
        print("✓ MediaPipe 0.10.32 initialized with Task API")
        
    def extract_keypoints(self, detection_result):
        """Extract 126 features from detection result"""
        keypoints = []
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints = [0] * 63
        
        # Pad to 126 (2 hands)
        while len(keypoints) < 126:
            keypoints.extend([0] * 63)
        
        return keypoints[:126]
    
    def normalize_keypoints(self, keypoints):
        """Normalize relative to wrist"""
        keypoints = np.array(keypoints).reshape(-1, 3)
        
        for hand_idx in range(2):
            start_idx = hand_idx * 21
            end_idx = start_idx + 21
            hand_kps = keypoints[start_idx:end_idx]
            
            if np.sum(hand_kps) != 0:
                wrist = hand_kps[0].copy()
                hand_kps = hand_kps - wrist
                keypoints[start_idx:end_idx] = hand_kps
        
        return keypoints.flatten().tolist()
    
    def draw_landmarks(self, frame, detection_result):
        """Draw hand landmarks on frame"""
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
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]
            
            for connection in connections:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
        
        return frame
    
    def collect_sign(self, sign_name, num_samples=30, sequence_length=30):
        """Collect data for one sign"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(" Cannot open camera!")
            return 0
        
        sign_dir = os.path.join(self.output_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        sample_count = 0
        recording = False
        sequence = []
        
        print(f"\n{'='*60}")
        print(f"Collecting: {sign_name.upper()}")
        print(f"Target: {num_samples} samples")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("- Press 'S' to START recording")
        print("- Perform the sign for 1 second")
        print("- Press 'Q' to QUIT")
        print(f"- Progress: {sample_count}/{num_samples}\n")
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            detection_result = self.detector.detect(mp_image)
            
            # Draw landmarks
            frame = self.draw_landmarks(frame, detection_result)
            
            # UI
            h, w = frame.shape[:2]
            status_text = "RECORDING..." if recording else "Press 'S' to START"
            status_color = (0, 0, 255) if recording else (0, 255, 0)
            
            cv2.putText(frame, status_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame, f"Sign: {sign_name}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if recording:
                cv2.putText(frame, f"Frames: {len(sequence)}/{sequence_length}", 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Progress bar
                bar_width = 300
                bar_x = 10
                bar_y = 180
                progress = len(sequence) / sequence_length
                
                cv2.rectangle(frame, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), 
                             (bar_x + int(bar_width * progress), bar_y + 20), 
                             (0, 255, 255), -1)
            
            cv2.putText(frame, "Press 'S' to start | 'Q' to quit", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Collect keypoints
            if recording:
                keypoints = self.extract_keypoints(detection_result)
                keypoints = self.normalize_keypoints(keypoints)
                sequence.append(keypoints)
                
                if len(sequence) >= sequence_length:
                    sample_path = os.path.join(sign_dir, f'sample_{sample_count:03d}.npy')
                    np.save(sample_path, np.array(sequence))
                    
                    sample_count += 1
                    sequence = []
                    recording = False
                    
                    print(f"✓ Saved sample {sample_count}/{num_samples}")
                    time.sleep(0.3)
            
            cv2.imshow('VSL Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                if not recording:
                    recording = True
                    sequence = []
                    print(f"→ Recording sample {sample_count + 1}...")
            elif key == ord('q') or key == ord('Q'):
                print("\n Stopped by user.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.metadata['signs'][sign_name] = {
            'num_samples': sample_count,
            'sequence_length': sequence_length
        }
        self.metadata['total_samples'] += sample_count
        
        print(f"\n✓ Completed: {sign_name} - {sample_count} samples")
        return sample_count
    
    def save_metadata(self):
        """Save metadata"""
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ Metadata saved: {metadata_path}")

def main():
    collector = VSLDataCollector(output_dir='../data/raw')
    
    signs = [
        'xin_chao',
        'cam_on',
        'i_love_you',
    ]
    
    print("\n" + "="*60)
    print("VSL DATA COLLECTION TOOL")
    print("="*60)
    print(f"Signs: {', '.join(signs)}")
    print("="*60)
    
    for sign in signs:
        response = input(f"\nCollect '{sign}'? (y/n): ").lower()
        if response == 'y':
            collector.collect_sign(sign, num_samples=30)
        else:
            print(f"⊘ Skipped '{sign}'")
    
    collector.save_metadata()
    
    print("\n" + "="*60)
    print("✓ COLLECTION COMPLETE!")
    print(f"Total: {collector.metadata['total_samples']} samples")
    print("="*60)

if __name__ == '__main__':
    main()