# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:24:22 2024

@author: ryo
"""
from face_encoding_cache import FaceEncodingCache
from face_recognition_processor import FaceRecognitionProcessor

import unknown_Hist
import datetime
import shutil
import glob

import os

cameras = ["192.168.128.167", "192.168.128.161", "192.168.128.164", "192.168.128.170", 
           "192.168.128.176", "192.168.128.180", "192.168.128.172", "192.168.128.173",
           "192.168.128.177", "192.168.128.179", "192.168.128.168", "192.168.128.166",
           "192.168.128.165", "192.168.128.160", "192.168.128.162", "192.168.128.169",
           "192.168.128.178", "192.168.128.175", "192.168.128.171", "192.168.128.163",
           "192.168.128.174"]#カメラのアドレス(リクエストの際のurlの一部に使用)


def unknownfinder(file_path):

    
    cache_manager = FaceEncodingCache()
    processor = FaceRecognitionProcessor(cache_manager)
   
    cam_time = datetime.datetime.now() #画像取得を起動した時刻

    today_before = datetime.date.today() # 日付取得
    
    detected_faces = processor.process_frame(cameras[0]) #顔認証関数の呼び出し(こちらはカメラ1台のみ動かすテスト用)
    #for i in range(len(cameras)):
     #detected_faces = processor.process_frame(cameras[i]) #顔認証関数の呼び出し
 
         
    
    today_after = datetime.date.today() # 日付更新か否か取得
    deltaday = today_after - today_before 
    #print(deltaday.days) #テスト用
    #一日？おきに、保存されたunknown検出のログを管理者？に転送
    #↑日付更新時とする
    if deltaday.days > 0: #0で日付更新していない。1で日付更新している
        input_path = file_path + "unknown_faces/"
        output_path = file_path + "save_unknown_faces_yesterday/"
        move_file_list = glob.glob(input_path + "*")
        for item in move_file_list:
            shutil.move(item, output_path)
         
        print("renewal date")
        
        unknown_Hist.Unknown_Hist(output_path)
            