# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:24:11 2024

@author: ryo
"""

import UnknownFinder
import datetime

file_path = 'C:/Users/ryo/.spyder-py3/python_practice/cam_test/'



dt_start = datetime.datetime.now()#プログラム開始時点の現在時刻
last_cam_time = datetime.datetime.now() #最後にカメラを起動した時刻(ここは起動時のエラー対策)
while True:
    today_before = datetime.date.today() # 日付取得
    
    spent_time = datetime.datetime.now() - last_cam_time #前回カメラを起動してから立った時間を計算する用のやつ
    
    if (spent_time.seconds >= 5.0):
        last_cam_time = datetime.datetime.now()
        UnknownFinder.unknownfinder(file_path)
        
    #プログラム開始時点からどれだけ時間が経ったか(test)
    spenttime = datetime.datetime.now() - dt_start
    
    if spenttime.total_seconds() >= 10.0:
        print("end")
        break


