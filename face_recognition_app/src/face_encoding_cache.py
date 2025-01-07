import face_recognition
import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime, timezone
import hashlib

class FaceEncodingCache:
    #関数：初期化処理.
    def __init__(self, image_dir="images", cache_dir="encoding_cache"):
        #プロジェクトのルートディレクトリを取得(srcフォルダの親ディレクトリ).
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        #各ディレクトリのパスを設定.
        self.image_dir = os.path.join(project_root, image_dir) #顔画像を保存するディレクトリ(デフォルトは"images").
        self.cache_dir = os.path.join(project_root, cache_dir) #エンコーディングデータを保存するディレクトリ(デフォルトは"encoding_cashe").
        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.pkl") #キャッシュの索引情報を保存するファイルパス.

        #必要なディレクトリを作成.
        self.ensure_dirs_exist()
        
    #関数：キャッシュディレクトリ、顔画像ディレクトリ(image)が存在しない場合は作成.
    def ensure_dirs_exist(self):
        os.makedirs(self.image_dir, exist_ok=True) #画像ディレクトリの作成.
        os.makedirs(self.cache_dir, exist_ok=True) #キャッシュディレクトリの作成. exist_ok=True : 存在する場合何もしない.

    #関数：ファイルのハッシュ値の計算.  
    def get_file_hash(self, filepath):
        f = None
        try:
            f = open(filepath, 'rb')
            img_data = f.read() #ファイルデータの読み込み.
            hash_img = hashlib.md5(img_data).hexdigest() #写真のハッシュ化.
            return hash_img
        except IOError as e:
            print(f"ファイル読み込みエラー: {str(e)}")
            return None #もしくはデフォルト値を返す.
        finally:
            if f is not None:
                f.close()

    #関数：キャッシュファイル（画像名+ハッシュ値）のパスを生成.
    def get_cache_path(self, image_name, hash_value):
        cashfile_name =f"{image_name}_{hash_value}.pkl"
        return os.path.join(self.cache_dir, cashfile_name)
    
    #関数：キャッシュインデックスを読み込み.
    def load_cache_index(self):
        if not os.path.exists(self.cache_index_path):
            return{}
        
        f = None
        try:
            f = open(self.cache_index_path,'rb')
            return pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            print(f"キャッシュファイルの読み込みエラー:{str(e)}")
            return {}
        finally:
            if f is not None:
                f.close()
    
    #関数：キャッシュインデックスを保存.
    def save_cache_index(self, cache_index):
        with open(self.cache_index_path, 'wb') as f:
            pickle.dump(cache_index, f)
    
    #関数：顔エンコーディングをキャッシュから読み込みまたは新規作成.
    def load_face_encodings(self):
        known_face_encodings = []
        known_face_names = []
        cache_index = self.load_cache_index()
        updated_cache_index = {}
        
        start_time = time.time()
        print("Loading face encodings...")
        
        for image_file in os.listdir(self.image_dir):
            #条件分岐:ファイルの拡張子確認.
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
              
            image_path = os.path.join(self.image_dir, image_file)
            current_hash = self.get_file_hash(image_path)
            user_name = os.path.splitext(image_file)[0]
            
            """
            キャッシュが有効かチェックする.
            （有効なとき cashe_valid=Tru)
            1.キャッシュが存在するか
            2.キャッシュが最新か（ハッシュ値で確認）
            3.キャッシュファイルが実際に存在するか
            4.キャッシュデータが正常に読み込めるか
            """
            cache_valid = False
            #キャッシュファイルの存在確認.
            if image_file in cache_index:
                #ハッシュ値の比較し、ファイルが変更されていないか確認.
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
            
            """
            新規エンコーディング（キャッシュ無効時のみ）
            1.顔検出がされない場合は、通知.
            2.複数の検出がされた場合最初に検出された顔を使用.
            3.個別のキャッシュを保存.
            4.キャッシュインデックスの更新.
            """
            if not cache_valid:
                print(f"Generating new encoding for: {user_name}")
                try:
                    user_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(user_image)
                    
                    #顔が検出されなかった場合.
                    if not face_encodings:
                        print(f"No face found in: {user_name}")
                        continue
                    
                    #複数の顔が検出された時は最初に検出された顔を使用.
                    user_face_encoding = face_encodings[0]
                    
                    #キャッシュを保存
                    cache_path = self.get_cache_path(user_name, current_hash)
                    with open(cache_path, 'wb') as f:  #wbモード:書き込みモード.
                        pickle.dump(user_face_encoding, f)
                    
                    known_face_encodings.append(user_face_encoding)
                    known_face_names.append(user_name)
                except Exception as e:
                    print(f"Error processing {user_name}: {str(e)}")
                    continue
            
            #キャッシュインデックスを更新.
            updated_cache_index[image_file] = {
                'hash': current_hash,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # 新しいキャッシュインデックスを保存.
        self.save_cache_index(updated_cache_index)
        
        # 古いキャッシュファイルを削除.
        self.cleanup_old_cache_files(updated_cache_index)

        print(f"Loaded {len(known_face_names)} faces in {time.time() - start_time:.2f} seconds")
        return known_face_encodings, known_face_names
    
    
    """
    関数:使われていない古いキャッシュファイルを削除.
    current_index:更新したキャッシュリスト
    それに対し、同じものがあれば削除.
    """
    def cleanup_old_cache_files(self, current_index):
        current_cache_files = set() #空の集合を作成.
        #キャッシュインデックスをループ.
        for image_file, info in current_index.items():
            user_name = os.path.splitext(image_file)[0]
            cache_path = self.get_cache_path(user_name, info['hash'])
            current_cache_files.add(os.path.basename(cache_path))
        
        for cache_file in os.listdir(self.cache_dir):
            if cache_file.endswith('.pkl') and cache_file != 'cache_index.pkl':
                if cache_file not in current_cache_files:
                    os.remove(os.path.join(self.cache_dir, cache_file))