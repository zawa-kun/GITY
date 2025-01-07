# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:17:13 2024

@author: ryo
"""

import requests
import cv2

#ファイルの位置を指定
file_path = 'C:/Users/ryo/.spyder-py3/python_practice/cam_test/test_image/'


# get()メソッドでGETリクエストを送信する
response = requests.get("http://192.168.128.165/capture")

# 結果を出力する
print(response.content)


image = response.content

#保存名を設定する
image_name = "image.jpg"

#画像を保存する
with open(file_path + image_name, "wb") as f:
    f.write(image)
    
#画像をjpeg形式で再読み込み
image_jpg = cv2.imread( file_path + image_name)

#テスト用
cv2.imshow('test', image_jpg)

cv2.waitKey(0)
cv2.destroyAllWindows()

