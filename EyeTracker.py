# coding: utf-8

import argparse
from os import makedirs
from os.path import basename, splitext, isfile
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
import cv2
import dlib

# from pdb import set_trace
# import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inlist")
    parser.add_argument("outd")
    # face detection
    parser.add_argument("--face_detector", "-f", choices=("opencv", "dlib", "dlib_cnn", "mtcnn"), default="dlib")
    parser.add_argument("--opencv_face_detector", \
        default="./models/opencv/haarcascade_frontalface_default.xml")
    parser.add_argument("--dlib_cnn_face_detector", \
        default="./models/dlib/mmod_human_face_detector.dat")
    # landmark detection
    parser.add_argument("--landmark_detecotr", "-l", choices=("dlib", ), default="dlib")
    parser.add_argument("--dlib_landmark_detector", \
        default="./models/dlib/shape_predictor_68_face_landmarks.dat")
    # resize, threshold, color_mode
    parser.add_argument("--face_size", type=int, default=224, \
        help="ROI of face is resized to this argument; default is '224'.")
    parser.add_argument("--ear_threshold", type=float, default=0.2, \
        help="Thredhold of EAR (eyes aspect ratio); default is '0.2'.")
    # demo
    parser.add_argument("--demo", "-d", action="store_true", default=False)
    parser.add_argument("--demo_index", "-i", type=int, default=0)
    parser.add_argument("--demo_frame", type=int, nargs=2)
    # others
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--binary_threthold", "-b", type=int, default=20)
    parser.add_argument("--njobs", "-n", type=int, default=1)
    # check and return
    args =  parser.parse_args()
    if args.njobs>cpu_count():
        print("$njobs is larger than # of CPUs (%d); $njobs is changed to # of CPUs." %cpu_count())
        args.njobs = cpu_count()
    if args.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                pass
            else:
                print("WARNING: You can not use GPU.")
                args.use_gpu = False
        except ImportError:
            exit("You did not install 'PyTorch'. You do not have to active $use_gpu.")
    args.binary_threthold = min(255, max(args.binary_threthold, 0))
    args.skip = max(0, args.skip) + 1
    return args

def mapping(x, before_min, before_max, after_min, after_max):
    before_scaled = float(x - before_min) / (before_max - before_min)
    return before_scaled * (after_max - after_min) + after_min

def convert_time(time):
    h, m, s = 0, 0, time
    if s>=60:
        m = int(s // 60)
        s -= 60*m
    if m>=60:
        h = m // 60
        s -= 60*h
    return "%s:%s:%2.4f" %(str(h).zfill(2), str(m).zfill(2), s)


class EyeDetector:
    def __init__(self):
        # Initalize face detector
        if args.face_detector=="opencv":
            self.face_detector = cv2.CascadeClassifier(args.opencv_face_detector)
        elif args.face_detector=="dlib":
            self.face_detector = dlib.get_frontal_face_detector()
        elif args.face_detector=="dlib_cnn":
            if not isfile(args.dlib_cnn_face_detector):
                raise FileNotFoundError("%s is not found." %args.dlib_cnn_face_detector)
            self.face_detector = dlib.cnn_face_detection_model_v1(args.dlib_cnn_face_detector)
        elif args.face_detector=="mtcnn":
            self.device = "gpu" if args.use_gpu else "cpu"
            self.face_detector = MTCNN(select_largest=False, device=self.device)
        # Initalize landmark detector
        if args.landmark_detecotr=="dlib":
            if not isfile(args.dlib_landmark_detector):
                raise FileNotFoundError("%s is not found." %args.dlib_landmark_detector)
            self.landmark_detecotr = dlib.shape_predictor(args.dlib_landmark_detector)

    def normalize_histogram(self, frame, mode):
        # BGRからHSVに変換
        # V（輝度）を正規化して，再度BGRに変換
        # 実施タイミングは顔検出の前

    def get_EAR(self, ps):
        p1, p2, p3, p4, p5, p6 = ps
        ear = (distance.euclidean(p2, p6) + distance.euclidean(p3, p5)) / (2 * distance.euclidean(p1, p4))
        return ear

    def BlinkEstimator(self, right_eyes, left_eyes):
        right_ear = self.get_EAR(right_eyes)
        left_ear = self.get_EAR(left_eyes)
        ear = (right_ear + left_ear) / 2
        rslt = args.ear_threshold>=ear
        return ear, rslt

    def get_landmarks(self, face):
        h, w, c = face.shape
        landmarks = self.landmark_detecotr(face, dlib.rectangle(0, 0, w, h))
        out = []
        for i in range(64):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            out.append([x, y])
        return np.array(out, dtype=np.float32)

    def get_eyeROI(self, face, landmarks):
        # left eye
        x1_l = landmarks[42][0]
        x2_l = landmarks[45][0]
        y1_l = min(landmarks[42:48][:, 1])
        y2_l = max(landmarks[42:48][:, 1])
        left_eye = self.get_ROI(face, x1_l, x2_l, y1_l, y2_l)
        # right eye
        x1_r = landmarks[36][0]
        x2_r = landmarks[39][0]
        y1_r = min(landmarks[36:42][:, 1])
        y2_r = max(landmarks[36:42][:, 1])
        right_eye = self.get_ROI(face, x1_r, x2_r, y1_r, y2_r)
        return left_eye, right_eye, (x1_l, x2_l, y1_l, y2_l), (x1_r, x2_r, y1_r, y2_r)    

    def eyeCnterEstimator(self, face, landmarks, ):
        left_eye, right_eye, eye_axisL, eye_axisR = self.get_eyeROI(face, landmarks)
        # left eye
        try:
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            ret, left_eye = cv2.threshold(left_eye, args.binary_threthold, 255, cv2.THRESH_BINARY_INV)
            left_moments = cv2.moments(left_eye, False)
            left_center = np.array([left_moments["m10"]/left_moments["m00"], left_moments["m01"]/left_moments["m00"]])
        except ZeroDivisionError:
            left_center = [-1, -1]
        # right eye
        try:
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            ret, right_eye = cv2.threshold(right_eye, args.binary_threthold, 255, cv2.THRESH_BINARY)
            right_moments = cv2.moments(right_eye, False)
            right_center = np.array([right_moments["m10"]/right_moments["m00"], right_moments["m01"]/right_moments["m00"]])
        except ZeroDivisionError:
            right_center = [-1, -1]
        return left_center, right_center, eye_axisL, eye_axisR

    def get_eyeLandmarks(self, face, landmarks):
        h, w, c = face.shape
        landmarks = self.get_landmarks(face)
        if args.landmark_detecotr=="dlib":
            right_eyes = landmarks[36: 42]
            left_eyes = landmarks[42: 48]
        return right_eyes, left_eyes

    def get_ROI(self, frame, x1, x2, y1, y2):
        x1, x2, y1, y2 = list(map(int, [x1, x2, y1, y2]))
        if len(frame.shape)==3:
            return frame[y1: y2, x1: x2, :]
        else:
            return frame[y1: y2, x1: x2]

    def adjust_frame(self, frame):
        if len(frame.shape)==2 or frame.shape[2]==1:
            if len(frame.shape)==2:
                frame = frame.reshape(frame.shape[0], frame.shape[1], 1)
            frame = np.concatenate([frame, frame, frame], axis=2)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def face_detection(self, frame):
        if args.face_detector=="opencv":
            # Face detection using OpenCV
            faces = self.face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if len(faces)>0:
                x1, y1, w, h = faces[0]
                x2 = x1 + w
                y2 = y1 + h
            else:
                return None, None, None
        elif args.face_detector in ("dlib", "dlib_cnn"):
            # Face detection using dlib
            dets = self.face_detector(frame, 1)
            if len(dets)>0:
                det = dets[0]
                if args.face_detector=="dlib_cnn":
                    x1, x2, y1, y2 = det.rect.left(), det.rect.right(), det.rect.top(), det.rect.bottom()
                elif args.face_detector=="dlib":
                    x1, x2, y1, y2 = det.left(), det.right(), det.top(), det.bottom()
            else:
                return None, None, None
        elif args.face_detector=="mtcnn":
            x1, y1, x2, y2 = self.face_detector.detect(frame)[0].astype(np.int16)[0]
        absL = abs((x2-x1) - (y2-y1))
        if (x2-x1)>(y2-y1):
            x2 -= absL
        elif (y2-y1)>(x2-x1):
            y1 += absL
        face_org = (x1, x2, y1, y2)
        face = self.get_ROI(frame, x1, x2, y1, y2)
        h, w, c = face.shape
        # To resize face-ROI
        h, w, c = face.shape
        if h!=args.face_size:
            method = cv2.INTER_AREA if args.face_size>h else cv2.INTER_CUBIC
            face = cv2.resize(face, (args.face_size, args.face_size), interpolation=method)
        return face, (x1, x2, y1, y2), face_org

    def writer_init(self, inf, outf):
        cap = cv2.VideoCapture(inf)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        h, w, c = frame.shape
        writer = cv2.VideoWriter(outf, cv2.VideoWriter_fourcc("m", "p", "4", "v"), int(fps), (w, h))
        return writer

    def _main(self, inf):
        cap = cv2.VideoCapture(inf)
        fps = cap.get(cv2.CAP_PROP_FPS)
        EARs, Blinks, Times = [], [], []
        eyeR_X, eyeR_Y, eyeL_X, eyeL_Y = [], [], [], []
        times = 0.00
        out = "Time\tEAR\tBlink\tface-size\tface_x1\tface_y1\tLeft-eye_x\tLeft-eye_y\tRight-eye_x\tRight_eye_y\n"
        if args.demo:
            writer = self.writer_init(inf, "%s/%s.demo.mp4" %(args.outd, splitext(basename(inf))[0]))
        else:
            writer = None
        cnt = -1
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as t:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cnt += 1
                if cnt%args.skip!=0:
                    t.update(1)
                    continue
                if args.demo and args.demo_frame is not None:
                    if (cnt+1)<args.demo_frame[0]:
                        t.update(1)
                        continue
                times += (fps/1000)
                frame = self.adjust_frame(frame)
                face, face_axis, face_org_axis = self.face_detection(frame)
                if face is None:
                    ear, blink = -1, None
                    left_center, right_center = [-1, -1], [-1, -1]
                    face_org_axis = [-1, -1, -1, -1]
                    face_org_size = -1
                else:
                    face_org_size = face_org_axis[1]-face_org_axis[0]
                    landmarks = self.get_landmarks(face)
                    right_eyes, left_eyes = self.get_eyeLandmarks(face, landmarks)
                    EAR, blink = self.BlinkEstimator(right_eyes, left_eyes)
                    if blink:
                        left_center, right_center = [-1, -1], [-1, -1]
                    else:
                        left_center, right_center, eye_axisL, eye_axisR = self.eyeCnterEstimator(face, landmarks)
                        left_center[0] = \
                            mapping(left_center[0]+eye_axisL[0], 0, args.face_size, 0, face_org_size) + face_axis[0]
                        left_center[1] = \
                            mapping(left_center[1]+eye_axisL[2], 0, args.face_size, 0, face_org_size) + face_axis[2]
                        right_center[0] = \
                            mapping(right_center[0]+eye_axisR[0], 0, args.face_size, 0, face_org_size) + face_axis[0]
                        right_center[1] = \
                            mapping(right_center[1]+eye_axisR[2], 0, args.face_size, 0, face_org_size) + face_axis[2]
                out += "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    %(convert_time(times), EAR, blink, face_org_size, face_org_axis[0], face_org_axis[2], left_center[0], left_center[1], right_center[0], right_center[1])
                if args.demo:
                    writer.write(self.demo(frame, face_org_axis, landmarks, EAR, blink, left_center, right_center))
                    if args.demo_frame is not None:
                        if (cnt+1)>=args.demo_frame[1]:
                            break
                t.update(1)
        cap.release()
        if writer is not None:
            writer.release()
        outf = "%s/%s.tsv" %(args.outd, splitext(basename(inf))[0])
        with open(outf, "w") as fd:
            fd.write(out)
        print("%s is saved." %outf)
        return 0

    def demo(self, frame, face_axis, landmarks, EAR, blink, left_center, right_center):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if blink is None:
            return frame
        text = "EAR = %.3f" %EAR
        if blink:
            text += " Blink!"
            color = (255, 0, 255)
        else:
            color = (0, 0, 255)
        frame = cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 10)
        frame = cv2.rectangle(frame, (face_axis[0], face_axis[2]), (face_axis[1], face_axis[3]), color, 3)
        for i in range(36, 48):
            pt = landmarks[i]
            pt[0] = mapping(pt[0], 0, args.face_size, 0, face_axis[1]-face_axis[0])+face_axis[0]
            pt[1] = mapping(pt[1], 0, args.face_size, 0, face_axis[3]-face_axis[2])+face_axis[2]
            frame = cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)
        frame = cv2.circle(frame, tuple(map(int, left_center)), 3, (0, 255, 255), -1)
        frame = cv2.circle(frame, tuple(map(int, right_center)), 3, (0, 255, 255), -1)
        return frame

    def __call__(self, inf):
        self._main(inf)
        return 0

if __name__=="__main__":
    args = get_args()
    if args.face_detector=="mtcnn":
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
    with open(args.inlist) as fd:
        inlist = fd.read().strip().split("\n")
    makedirs(args.outd, exist_ok=True)
    eye_detector = EyeDetector()
    if args.demo:
        eye_detector(inlist[args.demo_index])
    else:
        if args.njobs==1:
            for inf in tqdm(inlist):
                eye_detector(inf)
        else:
            pool = Pool(args.njobs)
            with tqdm(total=len(inlist)) as t:
                for _ in pool.imap_unordered(eye_detector, inlist):
                    t.update(1)



# 瞬き検出の基本系は完成：顔検出の実行速度はdlib>dlib_cnn（精度は逆かな）
# 追加機能として「デモ機能＝動画に対して瞬き検出結果やlandmarkを重畳」
# landmarkを使って黒目の重心を求めるのは未実装

# 参考
# EARの計算：https://qiita.com/mogamin/items/a65e2eaa4b27aa0a1c23#dlib%E7%89%88-%E5%AE%9F%E8%A3%85%E3%81%A8%E7%B5%90%E6%9E%9C
# 黒目の重心の取得：https://cppx.hatenablog.com/entry/2017/12/25/231121