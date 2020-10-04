# coding: utf-8

import argparse
from os.path import basename, splitext
import numpy as np
import cv2
import dlib

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inlist")
    parser.add_argument("outd")
    parser.add_argument("face_parts_detector", \
        help="shaper_predictor_68_fece_landmarks.dat")
    parser.add_argument("--face_norm", type=int, default=224, \
        help="ROI of face is resized to this argument; default is '224'.")
    parser.add_argument("--ear", type=float, default=0.2, \
        help="Thredhold of EAR (eyes aspect ratio); default is '0.2'.")

def _main(inf):
    cap = cv2.VideoCapture(inf)
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(args.face_parts_detector)
    while True:
        ret, frame = cap.read()
        if not ret:

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_detector(frame)
    cap.release()
    return 0

# 顔検出をして，顔領域を抽出→リサイズ
# 顔領域に対してlandmark detectionをする
# landmarkからEAR計算
# landmarkから目の領域を抽出→二値化などを使って黒目の重心をとる
# EAR，黒目の重心の推移を記録して外部ファイルに出力する
#   この時，EARが閾値を下回った値も記録して，何かに吐き出す

# 参考
# EARの計算：https://qiita.com/mogamin/items/a65e2eaa4b27aa0a1c23#dlib%E7%89%88-%E5%AE%9F%E8%A3%85%E3%81%A8%E7%B5%90%E6%9E%9C
# 黒目の重心の取得：https://cppx.hatenablog.com/entry/2017/12/25/231121