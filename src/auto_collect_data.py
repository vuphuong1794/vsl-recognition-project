"""
VSL Auto Collector Holistic from JSON - MediaPipe Task API
Phiên bản "Super Augmentation Holistic": Tạo >35 biến thể (Tay + Mặt + Dáng) từ 1 video.
"""

import cv2
import numpy as np
import os
import json
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class VSLAutoCollector:
    def __init__(self, json_path, output_dir='../data/raw'):
        self.output_dir = output_dir
        self.json_path = json_path
        self.sequence_length = 30 
        
        # Tạo thư mục lưu data
        os.makedirs(output_dir, exist_ok=True)

        print("Initializing MediaPipe Holistic (Auto Collector)...")
        self._setup_models()
        self._init_detectors()

    def _setup_models(self):
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
                except Exception as e:
                    print(f"❌ Failed to download {name}: {e}")

    def _init_detectors(self):
        # Hand
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2, min_hand_detection_confidence=0.3)
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

        # Face
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1)
        self.face_detector = vision.FaceLandmarker.create_from_options(options)

        # Pose
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(base_options=base_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)

    def process_json(self, target_list=None, limit=5):
        """Đọc file JSON và xử lý video."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Xử lý định dạng JSON List hay Dict
            video_list = data if isinstance(data, list) else data.get('words', [])
            
            data_to_process = []

            # LOGIC LỌC TỪ
            if target_list and len(target_list) > 0:
                print(f"🎯 Đang tìm kiếm các từ: {target_list}")
                targets_lower = [t.lower().strip() for t in target_list]
                
                for item in video_list:
                    gloss = item.get('gross', '').strip()
                    if gloss.lower() in targets_lower:
                        data_to_process.append(item)
                
                if len(data_to_process) == 0:
                    print(f"⚠️ Không tìm thấy từ nào trong danh sách yêu cầu!")
                    return
            else:
                data_to_process = video_list[:limit]
            
            print(f"✅ Tìm thấy {len(data_to_process)} video phù hợp. Bắt đầu xử lý...")
            
            for index, item in enumerate(data_to_process):
                gloss = item.get('gross')
                url = item.get('url')
                
                if gloss and url:
                    safe_name = gloss.replace(" ", "_").lower()
                    print(f"\n[{index+1}/{len(data_to_process)}] Đang học từ: '{gloss}'...")
                    self.process_single_video(safe_name, url)
                
        except Exception as e:
            print(f"Lỗi khi xử lý JSON: {e}")
            import traceback
            traceback.print_exc()

    def process_single_video(self, sign_name, video_url):
        cap = cv2.VideoCapture(video_url)
        
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_url}")
            return

        raw_sequence = [] 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect Holistic
            hand_res = self.hand_detector.detect(mp_image)
            face_res = self.face_detector.detect(mp_image)
            pose_res = self.pose_detector.detect(mp_image)
            
            # Extract combined keypoints
            kps = self.extract_keypoints(hand_res, face_res, pose_res)
            # Normalize (Optional: currently returning raw relative coords)
            norm_kps = self.normalize_keypoints(kps) 
            raw_sequence.append(kps)
            
        cap.release()

        if len(raw_sequence) < 10:
            print(f"⚠️ Video quá ngắn ({len(raw_sequence)} frames). Bỏ qua.")
            return

        # Tạo augmentation (Phiên bản Holistic)
        self.generate_augmentations(sign_name, raw_sequence)

    def extract_keypoints(self, hand_result, face_result, pose_result):
        """Extract combined keypoints: Pose(99) + Face(1434) + Hands(126) = 1659"""
        keypoints = []
        
        # 1. Pose (33 * 3)
        if pose_result.pose_landmarks:
            for landmark in pose_result.pose_landmarks[0]:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0] * 99)
            
        # 2. Face (478 * 3)
        if face_result.face_landmarks:
            for landmark in face_result.face_landmarks[0]:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0] * 1434)
            
        # 3. Hands (21 * 2 * 3)
        hand_kps = []
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                for landmark in hand_landmarks:
                    hand_kps.extend([landmark.x, landmark.y, landmark.z])
        while len(hand_kps) < 126:
            hand_kps.extend([0] * 63)
        keypoints.extend(hand_kps[:126])
        
        return keypoints

    def resample_sequence(self, sequence, target_len):
        if len(sequence) == target_len:
            return np.array(sequence)
        
        resampled = []
        sequence = np.array(sequence)
        length = len(sequence)
        indices = np.linspace(0, length - 1, target_len)
        
        for i in indices:
            low = int(math.floor(i))
            high = int(math.ceil(i))
            weight = i - low
            
            if high >= length:
                resampled.append(sequence[length-1])
            else:
                frame = sequence[low] * (1 - weight) + sequence[high] * weight
                resampled.append(frame)
                
        return np.array(resampled)

    # ==========================================================
    # 🚀 SUPER AUGMENTATION ENGINE (Holistic Version)
    # ==========================================================
    def apply_rotation(self, data_reshaped, angle):
        """Hàm phụ trợ để xoay dữ liệu 3D"""
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rot_data = data_reshaped.copy()
        # Chỉ xoay toạ độ X, Y (2 cột đầu tiên của dimension cuối)
        xy_coords = rot_data[:, :, :2] 
        rot_xy = np.dot(xy_coords, rot_matrix)
        rot_data[:, :, :2] = rot_xy
        
        # Flatten lại về vector đặc trưng (1659)
        return rot_data.reshape(self.sequence_length, -1)

    def generate_augmentations(self, sign_name, raw_sequence):
        save_path = os.path.join(self.output_dir, sign_name)
        os.makedirs(save_path, exist_ok=True)
        
        base_data = self.resample_sequence(raw_sequence, self.sequence_length)
        
        # Shape chuẩn Holistic: (30, 553, 3) vì 1659 / 3 = 553 điểm landmark
        num_landmarks = base_data.shape[1] // 3
        base_data_reshaped = base_data.reshape(self.sequence_length, num_landmarks, 3) 

        augmentations = []

        # === PHẦN 1: DỮ LIỆU GỐC & CƠ BẢN (2 file) ===
        augmentations.append(("org", base_data))
        
        noise = np.random.normal(0, 0.002, base_data.shape)
        augmentations.append(("noise", base_data + noise))

        # === PHẦN 2: CÁC BIẾN THỂ HÌNH HỌC (GỐC) ===
        angles = [-10, -5, 5, 10]        # 4 góc xoay
        scales = [0.9, 0.95, 1.05, 1.1] # 4 mức co giãn
        shifts = [                      # 4 mức dịch chuyển
            (0.02, 0), (-0.02, 0),
            (0, 0.02), (0, -0.02)
        ]

        # 2.1 Xoay (4 file)
        for angle in angles:
            aug_data = self.apply_rotation(base_data_reshaped, angle)
            augmentations.append((f"rot{angle}", aug_data))

        # 2.2 Scale (4 file)
        for scale in scales:
            aug_data = base_data * scale
            augmentations.append((f"scale{scale}", aug_data))

        # 2.3 Shift (4 file)
        for idx, (sx, sy) in enumerate(shifts):
            shift_data = base_data.copy()
            # Cộng shift cho X và Y (giả định normalized)
            # Reshape tạm để cộng đúng cột
            temp = shift_data.reshape(self.sequence_length, -1, 3)
            temp[:, :, 0] += sx
            temp[:, :, 1] += sy
            augmentations.append((f"shift{idx}", temp.reshape(self.sequence_length, -1)))

        # === PHẦN 3: LẬT GƯƠNG (Flip) ===
        mirror_reshaped = base_data_reshaped.copy()
        mirror_reshaped[:, :, 0] = -mirror_reshaped[:, :, 0] # Đảo trục X
        mirror_flat = mirror_reshaped.reshape(self.sequence_length, -1)
        
        augmentations.append(("flip_org", mirror_flat))

        # 3.1 Flip + Xoay (4 file)
        for angle in angles:
            aug_data = self.apply_rotation(mirror_reshaped, -angle) 
            augmentations.append((f"flip_rot{angle}", aug_data))

        # LƯU FILE
        count = 0
        for suffix, data in augmentations:
            filename = f"{sign_name}_{suffix}.npy"
            file_path = os.path.join(save_path, filename)
            np.save(file_path, data.astype(np.float32))
            count += 1
            
        print(f"   -> Đã tạo {count} file training (Holistic Augmentation) tại: {save_path}")


if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'data.json')
    output_dir = os.path.join(current_dir, '../data/raw')

    # ==========================================
    # 📝 DANH SÁCH TỪ BẠN MUỐN HỌC TẠI ĐÂY
    # ==========================================
    words_to_learn = [
        "vui mừng", 
        "buổi sáng", 
        "cảm ơn",
        "địa chỉ",
        "xin lỗi",
        "tạm biệt"
    ]

    print(f"Đang đọc data từ: {json_path}")
    
    if os.path.exists(json_path):
        collector = VSLAutoCollector(json_path=json_path, output_dir=output_dir)
        collector.process_json(target_list=words_to_learn)
    else:
        print(f"❌ Không tìm thấy file: {json_path}")
