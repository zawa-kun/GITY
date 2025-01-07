import cv2
import face_recognition
import numpy as np
import os
import hashlib
from datetime import datetime
import time
import requests

class FaceRecognitionProcessor:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.unknown_dir = os.path.join(project_root, "unknown_faces")
        os.makedirs(self.unknown_dir, exist_ok=True)
        self.last_save_time = {}
        self.min_save_interval = 30

    def get_frame_from_ip_camera(self, camera_url):
        """IPカメラから画像フレームを取得"""
        try:
            response = requests.get(camera_url)
            if response.status_code == 200:
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                return True, frame
            return False, None
        except Exception as e:
            print(f"Error capturing frame from IP camera: {str(e)}")
            return False, None

    def _draw_face_on_frame(self, frame, face_location, name):
        """フレーム上に顔の枠と名前を描画"""
        top, right, bottom, left = [coord * 4 for coord in face_location]
        
        # 顔の周りに枠を描画
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # 名前を表示する背景を描画
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        
        # 名前を表示
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    def process_frame(self, camera_url):
        """単一のフレームを処理して表示"""
        # エンコーディングのロード
        known_face_encodings, known_face_names = self.cache_manager.load_face_encodings()
        
        if not known_face_encodings:
            print("No face encodings loaded. Please add images to the 'images' directory.")
            return None

        # フレーム取得
        ret, frame = self.get_frame_from_ip_camera(camera_url)
        if not ret:
            print("Failed to capture frame from camera")
            return None

        # フレームのリサイズと処理
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 顔検出と認識
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # 検出結果
        detected_faces = []
        current_time = time.time()
        
        for i, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            name = "Unknown"
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                elif i < len(face_locations):
                    self._save_unknown_face(frame, face_locations[i], face_encoding, current_time)
            
            # 検出した顔を描画
            self._draw_face_on_frame(frame, face_locations[i], name)
            
            detected_faces.append({
                'name': name,
                'location': face_locations[i]
            })
        
        # フレームを表示
        cv2.imshow('Face Recognition', frame)
        cv2.waitKey(1)  # ウィンドウの更新に必要
        
        return detected_faces

    def _save_unknown_face(self, frame, face_location, face_encoding, current_time):
        """Unknown（未知）の顔を保存する"""
        try:
            top, right, bottom, left = [coord * 4 for coord in face_location]
            
            if (0 <= top < frame.shape[0] and 0 <= bottom < frame.shape[0] and
                0 <= left < frame.shape[1] and 0 <= right < frame.shape[1]):
                
                height = bottom - top
                width = right - left
                expanded_top = max(0, top - int(height * 0.1))
                expanded_bottom = min(frame.shape[0], bottom + int(height * 0.1))
                expanded_left = max(0, left - int(width * 0.1))
                expanded_right = min(frame.shape[1], right + int(width * 0.1))
                
                face_key = hashlib.md5(face_encoding.tobytes()).hexdigest()
                
                if (face_key not in self.last_save_time or 
                    current_time - self.last_save_time[face_key] >= self.min_save_interval):
                    
                    face_img = frame[expanded_top:expanded_bottom, 
                                   expanded_left:expanded_right]
                    if face_img.size > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(self.unknown_dir, 
                                             f"unknown_{timestamp}_{face_key[:8]}.jpg")
                        cv2.imwrite(filename, face_img)
                        print(f"Saved unknown face: {filename}")
                        
                        self.last_save_time[face_key] = current_time
        
        except Exception as e:
            print(f"Error saving unknown face: {str(e)}")

    def cleanup(self):
        """ウィンドウのクリーンアップ"""
        cv2.destroyAllWindows()