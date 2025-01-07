# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:34:05 2024

@author: ryo
"""

import cv2
import face_recognition
import time
# 2つの画像を読み込む


def CalcHist(image1, image2):

# ヒストグラムを計算する
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    # ヒストグラムを比較する
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(f'ヒストグラムの類似度: {similarity}')
    
    return similarity

def FaceComp(image1, image2):
    face_locations = face_recognition.face_locations(image1)
    face_encodings = face_recognition.face_encodings(image1, face_locations)
    
    img1_encoding = face_recognition.face_encodings(image1)
    img2_encoding = face_recognition.face_encodings(image2)
    
    
    face_names = []
    current_time = time.time()
    
    for i, face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(img1_encoding, img2_encoding)
        face_distances = face_recognition.face_distance(img1_encoding, img2_encoding)
        
    print(matches)
    print("\n" + face_distances)
    
    return face_distances