from face_encoding_cache import FaceEncodingCache
from face_recognition_processor import FaceRecognitionProcessor
import time

def main():
    # テスト用のカメラURL
    camera_url = "http://192.168.128.169/capture"  # テスト用のURLを直接指定
    
    cache_manager = FaceEncodingCache()
    processor = FaceRecognitionProcessor(cache_manager)

    try:
        while True:
            # フレームの処理
            detected_faces = processor.process_frame(camera_url)
            
            if detected_faces:
                print("Detected faces:")
                for face in detected_faces:
                    print(f"- {face['name']}")
            
            # 処理間隔（秒）
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping face recognition...")
        processor.cleanup()  # ウィンドウのクリーンアップ

if __name__ == "__main__":
    main()