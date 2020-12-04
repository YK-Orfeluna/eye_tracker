# eye_tracker

## 概要

- 事前に録画した動画から，画像処理ベースで**瞬き**と**目線**を検出するツール
- 顔検出によって推定された顔領域からそれぞれ推定される
	- 瞬き: 目のlandmarkを用いて，目の開閉を判定する
	- 目線: 目の領域を二値化して黒目のピクセルを推定し，その重心座標を求める
- リアルタイム処理については特に考慮なし
	- 実験後に撮影しておいた動画を分析する用途に使う前提だったりする

## 必須環境

- Python 3.5
	- numpy
	- sicpy
	- tqdm
	- OpenCV
	- dlib
- 以下の機能を使う場合は，Pytorchが必要
	- `$face_detector`でmtcnnを利用

\# OpenCVとdlibはAnacondaではPython3.5までしか対応していないので要注意

## 事前準備

- モデルファイルのダウンロード
	- MacOS / Linuxの場合
		- `bash prepare_model.sh`と実行する
	- Windowsの場合（もしくは，prepare_model.shをうまく実行できない場合）
		1. `models`ディレクトリを作成
		2. 以下のファイルを`models`ディレクトリ以下にダウンロードする
			- `$face_detector`でdlib_cnnを利用
				- *`http://dlib.net/files/mmod_human_face_detector.dat.bz2`*
			- `landmark_detector`でdlibを利用
				-  *`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`*
			- `$face_detector`opencvを利用（下記のいずれか）
				- *`hoge`*
				- *`hoge`*
				- *`hoge`*
		3. `*.bz2`ファイルを解凍する

## 使い方

- `python EyeTracker.py movie.list output`

### 結果の見方

- $outdに結果がtsvで保存される
	- ファイル名は入力ファイルの拡張子がtsvに変換されたものとなる
- 以下，tsvファイルの中身を順に解説
	- Time
		- 動画ファイル内での時間
	- EAR
		- 両面のEARの平均値
	- Blink
		- 瞬きをしていたらTRUEと，していなければFALSEと，顔検出に失敗した場合はNoneと出力される
	- face-size
		- 顔の大きさ（ピクセル）
		- 以下の目線の動き幅を顔サイズで正規化するといいかもしれない
	- Left-eye_x
		- 左黒目の重心のx座標
		- 瞬き時と顔検出失敗時は-1
	- Left-eye_y
		- 左黒目の重心のy座標
		- 瞬き時と顔検出失敗時は-1
	- Right-eye_x
		- 右黒目の重心のx座標
		- 瞬き時と顔検出失敗時は-1
	- Right-eye_y
		- 右黒目の重心のy座標
		- 瞬き時と顔検出失敗時は-1

### 引数について

- **`inlist`**
	- inputの画像ファイルパスのリスト（テキスト形式）
- **`outd`**
	- 結果の出力先（なければ自動生成）
	- 動画と同じ名前の結果のファイルがtsv形式で出力される
- `--face_detector` (`-f`)
	- 顔検出の方法を{opencv, dlib, dlib_cnn, mtcnn}から選択（デフォルトはdlib）
		- opencv: OpenCVのHaar-like cascaedeによる検出機能を利用
		- dlib: dlibベースの検出モデルを利用（HOG+SVM）
		- dlib_cnn: dlibベースの検出モデルを利用（CNN）
		- mtcnn: Pytorchで実装されたCNNベースの顔検出モデル
	- 精度はmtcnn>dlib_cnn>dlib>opencv
	- 速度はopencv>dlib, mtcnn > dlib_cnn
		- GPUを使えるなら，mtcnnは高速化可能
	- Pytorch環境を構築できるのであればmtcnnを強く推奨，それ以外であればdlibかdlib_cnnを計算量に応じて選択
- `--opencv_face_detector`
	- `$face_detector`でopencvを選択した場合，モデルファイルを選択する
	- デフォルトでは，事前準備でDLしたモデルファイルを利用するようになっている
- `--dlib_cnn_face_detector`
	- `$face_detector`でdlib_cnnを選択した場合，モデルファイルを指定する
	- デフォルトでは，事前準備でDLしたモデルファイルを利用するようになっている
- `--landmark_detecotor` (`-l`)
	- 現在dlibベースのlandmark detectorのみ対応
- `--dlib_landmark_detector`
	- `$dlib_landmardk_detector`でdlibを指定した場合，モデルファイルを指定する
	- デフォルトでは，事前準備でDLしたモデルファイルを利用するようになっている
- `--face_size`
	- 検出した顔領域のリサイズ（正方形）
	- デフォルトでは224 pixelにリサイズされる
- `--ear_threshold`
	- EAR（eye accept rate）の閾値
		- EAR < `$ear_threshold`で目を瞑っていると判定
	- デフォルトは0.2
- `--demo`
	- 引数を実行すると，デモを生成する
	- `$inlist`に対して`$demo_index`番目のファイルに対して各種機能を実行し，その結果を描画した動画を$outdに生成
- `--demo_index` (`-i`)
- `--demo_frame`
	- デモを実行するフレームを指定する
	- 例えば，`--demo_frame 1 100`とすると，1フレーム目から100フレーム目までデモを実行して終了する
- `--skip`
	- `$skip`フレームごとに，フレームを飛ばす
	- デフォルトは0（飛ばさない）
- `--use_gpu`
	- ニューラルネットワークベースのモデル利用時（以下の条件），この引数を実行するとGPUを使うように指定可能
		- `$face_detector`: mtcnn
	- GPUが使えない場合は自動でCPUを使うように切り替わる
- `--binary_threshold` (`-b`)
	- 目線を推定するための黒目の重心を推定する際の，目領域の二値化の閾値
		- グレースケール値が閾値以下の領域を黒目とする
	- デフォルトは20
- `--njobs`
	- マルチプロセスの数を指定する
	- デフォルトは1
	- GPU利用時は1を推奨

### コツ

- 動画の余分な箇所を切り取る（高速化，精度向上）
	- 確実に顔が映らない領域を事前にffmepgなどを用いて落としておく
- 動画を小さくしてみる（高速化，精度悪化）
	- 顔領域が`$face_size`より小さくならなければ，精度は大きくは低下しない（はず）
- fpsを適度に下げる（高速化）
	- 30fps以上あるなら，ffmpegなどを用いて30fps程度に落としてみる
	- 60fspあるなら`$skip`を1にしてみる＝実質30fps
- `$face_detector`をmtcnnにしてみる（高速化，精度向上）
	- Anaconda環境なら簡単にPyTorchは導入可能
	- GPUでの演算ができるとさらに高速化
- スペックの高いマシンを使う（高速化）
	- 高性能なCPU
		- そもそもの演算速度を高速化する
	- マルチスレッドを持つCPU
		- `$njobs`の数字を上げて同時に複数の動画を分析する
	- ニューラルネットワークベースのモデルとGPUの演算
		- NNをGPUで実行して高速化
- 寝る前に計算を実行する（時間節約）

### デモについて

- mp4形式で書き出し
	- 動画の音声情報は消失
- パラメタの確認に使うと良い
- 処理時間が長いので，以下の方法を推奨
	- 動画そのものを短くしておく
	- `$demo_frame`を指定して，指定されたフレーム数までのみをデモ対象とするようにする
- 以下，描画内容の解説
	- 検出された顔領域が赤い四角形で表示される
		- 瞬き時は色が紫に変化
	- 画面右上にEARが表示される
		- 瞬き時はさらに**Blink!**と表示
	- 目のランドマーク（左右それぞれ6点ずつ）が緑の丸で表示される
		- 顔検出失敗時は表示されない
- 黒目の重心が黄色の丸で表示される
	- 顔検出失敗時，瞬き時は表示されない