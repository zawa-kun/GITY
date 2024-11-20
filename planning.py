import face_recognition
import cv2
import numpy as np
import os


# この関数はimagesディレクトリにある顔画像をface_recognitionが認識するよう、known_face_namesとknown_face_encodingsに情報を追加する
# これから考えるべきことは
# 1. 画像の増加に伴ってオーバーヘッドがかかること
# 2. 実行中に画像の追加や削除をしても反映されないこと
# なおimagesディレクトリには、face_recognitionに認識されたい顔画像のみを配置することを想定している
def add_all_face_in_image_dir(known_face_encodings, known_face_names):
    target_dir = os.getcwd() + "/images/"

    for image_file in os.listdir(target_dir):
        user_image = face_recognition.load_image_file(target_dir + image_file)
        user_face_encoding = face_recognition.face_encodings(user_image)[0]
        known_face_encodings.append(user_face_encoding)

        user_name = os.path.splitext(image_file)[0]
        known_face_names.append(user_name)

        print("add: " + user_name)

    return known_face_encodings, known_face_names


# 顔認識用コード
# face_recognitionの例とほぼ同じ
def face_recg():
    video_capture = cv2.VideoCapture(0)

    known_face_encodings = []
    known_face_names = []
    known_face_encodings, known_face_names = add_all_face_in_image_dir(known_face_encodings, known_face_names)

    face_locations = []
    face_encodings = []
    face_names = []

    process_this_frame = True

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # --- add from forum --- #
        code = cv2.COLOR_BGR2RGB
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)
        # --- end --- #

        if process_this_frame:

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = "Unknown"

                # matches[best_match_index]は個人を識別できた場合Trueを返す
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

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

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    print(face_names)


face_recg()
