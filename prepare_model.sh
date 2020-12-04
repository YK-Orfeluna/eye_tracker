#!/usr/bin/env bash
mkdir -p models
cd models
# dlib
mkdir -p dlib; cd dlib
URLs=(\
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
    )
for url in ${URLs[@]} ; do
    if [ ! -f ${url%*.bz2} ] ; then
        wet $url
        bunzip2 $url
    fi
done
cd ..
# opencv
mkdir -p opencv; cd opencv
URLs=(\
    "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml" 
    "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml"
    "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
)
for url in ${URLs[@]} ; do
    if [ ! -f ${url%*.bz2} ] ; then
        wet $url
    fi
done
cd ..
exit 0;