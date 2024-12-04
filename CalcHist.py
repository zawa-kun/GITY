# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:34:05 2024

@author: ryo
"""

import cv2
# 2つの画像を読み込む


def CalcHist(image1, image2):

# ヒストグラムを計算する
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    # ヒストグラムを比較する
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(f'ヒストグラムの類似度: {similarity}')
    
    return similarity