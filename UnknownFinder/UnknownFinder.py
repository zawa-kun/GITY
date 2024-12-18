# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:24:22 2024

@author: ryo
"""

import planning
import unknown_Hist
import datetime
import shutil
import glob
import random as rd
import os


def unknownfinder(file_path):

    

    #デバッグモード(テキストの保存)
    debug = 0

   
    last_cam_time = datetime.datetime.now() #最後にカメラを起動した時刻(ここは起動時のエラー対策)

    today_before = datetime.date.today() # 日付取得
    
    face_num, unknown_num = planning.face_recg() #顔認証関数の呼び出し
    
    #顔が検出されたならば、その顔が知っているものか判定
    if face_num >= 1:
     
        
        cam_id = rd.randint(0, 15) #テスト用
        
        dt_found = last_cam_time.strftime('%Y-%m-%d %H:%M:%S') #現在時刻の文字列化
            
        if debug == 1:
                
            file = open(file_path + 'unknown_faces/test.txt', 'w') #テキストファイルの記述
            file.write("found_time " + dt_found + "\ncam_id " + str(cam_id) + "\nface_num " + str(face_num) + "\nunknown_num " + str(unknown_num))
            file.close()
    
        if unknown_num == 0:
            #知っている顔しかいないのなら、その画像は破棄
            print("no Unknown")
            if debug == 1:
                os.remove(file_path + 'unknown_faces/test.txt')
            os.remove(file_path + 'unknown_faces/test.jpg')
        else : 
            #知らない顔(unknown)が存在したならば、その時刻、位置、画像を保存
            print("Unknown found") 
            print ("found_time" + dt_found + " cam_id " + str(cam_id)) #unknown found になったときの時刻とカメラ位置の記述
            dt_found = last_cam_time.strftime('%Y_%m_%d %H_%M_%S')
            if debug == 1:
                filename = file_path + 'unknown_faces/test_' + dt_found + '.txt'
            filename_img = file_path +  'unknown_faces/test_' + dt_found + '.jpg'
            if debug == 1:
                os.rename(file_path + 'unknown_faces/test.txt',
                      filename)
            os.rename(file_path + 'unknown_faces/test.jpg',
                      filename_img)
            
    else :
        print("no face")    
        os.remove(file_path + 'unknown_faces/test.jpg')
        
         
    
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
            