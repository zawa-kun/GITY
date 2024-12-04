import face_recognition
import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime, timezone
import hashlib

class FaceEncodingCache:
    def __init__(self, image_dir="images", cache_dir="encoding_cache"):
        self.image_dir = os.path.join(os.getcwd(), image_dir)
        self.cache_dir = os.path.join(os.getcwd(), cache_dir)
        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.pkl")
        self.ensure_dirs_exist()
        
    #キャッシュディレクトリが存在しない場合は作成.
    def ensure_dirs_exist(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    #ファイルのハッシュ値の計算.  
    def get_file_hash(self, filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    #キャッシュファイルのパスを生成.
    def get_cache_path(self, image_name, hash_value):
        return os.path.join(self.cache_dir, f"{image_name}_{hash_value}.pkl")
    
    #キャッシュインデックスを読み込み.
    def load_cache_index(self):
        if os.path.exists(self.cache_index_path):
            with open(self.cache_index_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    #キャッシュインデックスを保存.
    def save_cache_index(self, cache_index):
        with open(self.cache_index_path, 'wb') as f:
            pickle.dump(cache_index, f)
    
    #顔エンコーディングをキャッシュから読み込みまたは新規作成.
    def load_face_encodings(self):
        known_face_encodings = []
        known_face_names = []
        cache_index = self.load_cache_index()
        updated_cache_index = {}
        
        start_time = time.time()
        print("Loading face encodings...")
        
        for image_file in os.listdir(self.image_dir):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(self.image_dir, image_file)
            current_hash = self.get_file_hash(image_path)
            user_name = os.path.splitext(image_file)[0]
            
            # キャッシュが有効かチェック
            cache_valid = False
            if image_file in cache_index:
                cached_hash = cache_index[image_file]['hash']
                if cached_hash == current_hash:
                    cache_path = self.get_cache_path(user_name, current_hash)
                    if os.path.exists(cache_path):
                        try:
                            with open(cache_path, 'rb') as f:
                                encoding = pickle.load(f)
                            known_face_encodings.append(encoding)
                            known_face_names.append(user_name)
                            cache_valid = True
                            print(f"Loaded from cache: {user_name}")
                        except:
                            print(f"Cache corrupted for: {user_name}")
            
            # キャッシュが無効な場合は新規エンコーディング
            if not cache_valid:
                print(f"Generating new encoding for: {user_name}")
                try:
                    user_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(user_image)
                    
                    if not face_encodings:
                        print(f"No face found in: {user_name}")
                        continue
                    
                    user_face_encoding = face_encodings[0]
                    
                    # キャッシュを保存
                    cache_path = self.get_cache_path(user_name, current_hash)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(user_face_encoding, f)
                    
                    known_face_encodings.append(user_face_encoding)
                    known_face_names.append(user_name)
                except Exception as e:
                    print(f"Error processing {user_name}: {str(e)}")
                    continue
            
            # キャッシュインデックスを更新
            updated_cache_index[image_file] = {
                'hash': current_hash,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # 新しいキャッシュインデックスを保存
        self.save_cache_index(updated_cache_index)
        
        # 古いキャッシュファイルを削除
        self.cleanup_old_cache_files(updated_cache_index)
        
        print(f"Loaded {len(known_face_names)} faces in {time.time() - start_time:.2f} seconds")
        return known_face_encodings, known_face_names
    
    def cleanup_old_cache_files(self, current_index):
        """使われていない古いキャッシュファイルを削除"""
        current_cache_files = set()
        for image_file, info in current_index.items():
            user_name = os.path.splitext(image_file)[0]
            cache_path = self.get_cache_path(user_name, info['hash'])
            current_cache_files.add(os.path.basename(cache_path))
        
        for cache_file in os.listdir(self.cache_dir):
            if cache_file.endswith('.pkl') and cache_file != 'cache_index.pkl':
                if cache_file not in current_cache_files:
                    os.remove(os.path.join(self.cache_dir, cache_file))

def face_recg():
    video_capture = cv2.VideoCapture(0)
    
    # キャッシュシステムを初期化
    cache_manager = FaceEncodingCache()
    known_face_encodings, known_face_names = cache_manager.load_face_encodings()
    
    if not known_face_encodings:
        print("No face encodings loaded. Please add images to the 'images' directory.")
        return
    
    # Unknown顔保存用のディレクトリを作成
    unknown_dir = "unknown_faces"
    os.makedirs(unknown_dir, exist_ok=True)
    
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    # Unknown顔の保存を制御する変数
    last_save_time = {}  # 同じ顔の連続保存を防ぐため
    min_save_interval = 30  # 同じ顔の保存間隔（秒）
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            continue
            
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            current_time = time.time()
            
            for i, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:  # 既知の顔がある場合のみ
                    best_match_index = np.argmin(face_distances)
                    name = "Unknown"
                    
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    elif i < len(face_locations):  # Unknown顔の保存処理
                        try:
                            # 顔の位置を元のフレームサイズに戻す
                            top, right, bottom, left = [coord * 4 for coord in face_locations[i]]
                            
                            # フレームの境界チェック
                            if (0 <= top < frame.shape[0] and 0 <= bottom < frame.shape[0] and
                                0 <= left < frame.shape[1] and 0 <= right < frame.shape[1]):
                                
                                # 顔領域を少し広げる（20%増）
                                height = bottom - top
                                width = right - left
                                expanded_top = max(0, top - int(height * 0.1))
                                expanded_bottom = min(frame.shape[0], bottom + int(height * 0.1))
                                expanded_left = max(0, left - int(width * 0.1))
                                expanded_right = min(frame.shape[1], right + int(width * 0.1))
                                
                                # エンコーディングをキーとして使用
                                face_key = hashlib.md5(face_encoding.tobytes()).hexdigest()
                                
                                # 前回の保存から十分な時間が経過しているか確認
                                if (face_key not in last_save_time or 
                                    current_time - last_save_time[face_key] >= min_save_interval):
                                    
                                    # 顔領域を切り出して保存
                                    face_img = frame[expanded_top:expanded_bottom, 
                                                   expanded_left:expanded_right]
                                    if face_img.size > 0:  # 画像が有効か確認
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = os.path.join(unknown_dir, 
                                                             f"unknown_{timestamp}_{face_key[:8]}.jpg")
                                        cv2.imwrite(filename, face_img)
                                        print(f"Saved unknown face: {filename}")
                                        
                                        # 保存時刻を更新
                                        last_save_time[face_key] = current_time
                        
                        except Exception as e:
                            print(f"Error saving unknown face: {str(e)}")
                else:
                    name = "Unknown"
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        
        # 顔の表示処理
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recg()