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
# hogehoge
cd ..
exit 0;