"""
VSL Auto Collector from JSON - MediaPipe Task API
Phiên bản "Super Augmentation": Tạo >35 biến thể từ 1 video.
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

        # Khởi tạo MediaPipe
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Đang tải model hand_landmarker...")
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_json(self, target_list=None, limit=5):
        """Đọc file JSON và xử lý video."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data_to_process = []

            # LOGIC LỌC TỪ
            if target_list and len(target_list) > 0:
                print(f"🎯 Đang tìm kiếm các từ: {target_list}")
                targets_lower = [t.lower().strip() for t in target_list]
                
                for item in data:
                    gloss = item.get('gross', '').strip()
                    if gloss.lower() in targets_lower:
                        data_to_process.append(item)
                
                if len(data_to_process) == 0:
                    print(f"⚠️ Không tìm thấy từ nào trong danh sách yêu cầu!")
                    return
            else:
                data_to_process = data[:limit]
            
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
            
            detection_result = self.detector.detect(mp_image)
            kps = self.extract_keypoints(detection_result)
            norm_kps = self.normalize_keypoints(kps)
            raw_sequence.append(norm_kps)
            
        cap.release()

        if len(raw_sequence) < 10:
            print(f"⚠️ Video quá ngắn ({len(raw_sequence)} frames). Bỏ qua.")
            return

        # Tạo augmentation (Phiên bản mới)
        self.generate_augmentations(sign_name, raw_sequence)

    def extract_keypoints(self, detection_result):
        keypoints = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        target_len = 126
        while len(keypoints) < target_len:
            keypoints.extend([0.0] * (target_len - len(keypoints)))
            
        return keypoints[:target_len]

    def normalize_keypoints(self, keypoints):
        kps = np.array(keypoints).reshape(-1, 3)
        for hand_idx in range(2):
            start = hand_idx * 21
            end = start + 21
            hand_kps = kps[start:end]
            if np.sum(hand_kps) != 0:
                wrist = hand_kps[0].copy()
                kps[start:end] = hand_kps - wrist
        return kps.flatten()

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
    # 🚀 SUPER AUGMENTATION ENGINE (Tạo >35 mẫu)
    # ==========================================================
    def apply_rotation(self, data_reshaped, angle):
        """Hàm phụ trợ để xoay dữ liệu"""
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rot_data = data_reshaped.copy()
        xy_coords = rot_data[:, :, :2] 
        rot_xy = np.dot(xy_coords, rot_matrix)
        rot_data[:, :, :2] = rot_xy
        return rot_data.reshape(self.sequence_length, -1)

    def generate_augmentations(self, sign_name, raw_sequence):
        save_path = os.path.join(self.output_dir, sign_name)
        os.makedirs(save_path, exist_ok=True)
        
        base_data = self.resample_sequence(raw_sequence, self.sequence_length)
        # Shape chuẩn để biến đổi hình học: (30, 42, 3)
        base_data_reshaped = base_data.reshape(self.sequence_length, -1, 3) 

        augmentations = []

        # === PHẦN 1: DỮ LIỆU GỐC & CƠ BẢN (2 file) ===
        augmentations.append(("org", base_data))
        
        # Thêm nhiễu nhẹ (Noise)
        noise = np.random.normal(0, 0.002, base_data.shape)
        augmentations.append(("noise", base_data + noise))

        # === PHẦN 2: CÁC BIẾN THỂ HÌNH HỌC (GỐC) ===
        # Định nghĩa các tham số biến đổi
        angles = [-12, -8, -4, 4, 8, 12]        # 6 góc xoay
        scales = [0.85, 0.9, 0.95, 1.05, 1.1, 1.15] # 6 mức co giãn
        shifts = [                              # 4 mức dịch chuyển
            (0.03, 0), (-0.03, 0),  # Trái/Phải
            (0, 0.03), (0, -0.03)   # Lên/Xuống
        ]

        # 2.1 Xoay (6 file)
        for angle in angles:
            aug_data = self.apply_rotation(base_data_reshaped, angle)
            augmentations.append((f"rot{angle}", aug_data))

        # 2.2 Scale (6 file)
        for scale in scales:
            aug_data = base_data * scale
            augmentations.append((f"scale{scale}", aug_data))

        # 2.3 Shift (4 file)
        for idx, (sx, sy) in enumerate(shifts):
            shift_data = base_data.copy()
            # Dữ liệu dạng phẳng, shift cộng thẳng vào
            # Tuy nhiên, shift x, y cần cẩn thận hơn, ở đây ta cộng đều (đơn giản hóa)
            # Vì đã normalize, cộng đều vào toàn bộ frame coi như shift tâm
            # Để chính xác: Ta reshape lại, cộng sx vào cột X, sy vào cột Y
            temp = shift_data.reshape(self.sequence_length, -1, 3)
            temp[:, :, 0] += sx
            temp[:, :, 1] += sy
            augmentations.append((f"shift{idx}", temp.reshape(self.sequence_length, -1)))

        # === PHẦN 3: LẬT GƯƠNG VÀ COMBO (GẤP ĐÔI SỐ LƯỢNG) ===
        # Tạo bản lật gương (Flip)
        mirror_reshaped = base_data_reshaped.copy()
        mirror_reshaped[:, :, 0] = -mirror_reshaped[:, :, 0] # Đảo trục X
        mirror_flat = mirror_reshaped.reshape(self.sequence_length, -1)
        
        augmentations.append(("flip_org", mirror_flat)) # 1 file

        # 3.1 Flip + Xoay (6 file)
        for angle in angles:
            # Xoay ngược chiều lại một chút cho đa dạng
            aug_data = self.apply_rotation(mirror_reshaped, -angle) 
            augmentations.append((f"flip_rot{angle}", aug_data))

        # 3.2 Flip + Scale (6 file)
        for scale in scales:
            aug_data = mirror_flat * scale
            augmentations.append((f"flip_scale{scale}", aug_data))
        
        # 3.3 Flip + Shift (4 file)
        for idx, (sx, sy) in enumerate(shifts):
            temp = mirror_reshaped.copy()
            temp[:, :, 0] += sx
            temp[:, :, 1] += sy
            augmentations.append((f"flip_shift{idx}", temp.reshape(self.sequence_length, -1)))

        # === TỔNG KẾT ===
        # Org(1) + Noise(1) + Rot(6) + Scale(6) + Shift(4) = 18
        # Flip(1) + FlipRot(6) + FlipScale(6) + FlipShift(4) = 17
        # Tổng cộng: 35 file training chất lượng cao.

        # LƯU FILE
        count = 0
        for suffix, data in augmentations:
            filename = f"{sign_name}_{suffix}.npy"
            file_path = os.path.join(save_path, filename)
            np.save(file_path, data.astype(np.float32))
            count += 1
            
        print(f"   -> Đã tạo {count} file training (Super Augmentation) tại: {save_path}")


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