# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import dlib
import cv2
import sys
import math
#for OSC communications
from pythonosc import udp_client

#テキストファイルの監視時間の制御
import time

#frameのコピーをする
import copy

"""
このファイルはOSC通信を可能にするためにデバッグしたものである。OSC通信では引数化して渡すことができないので注意せよ。
"""
####x_range---右端が640左端が0で200-400でやればよいな。
####y_range---0-360の間になっているはず


#パーティクルの定義
#particle = [x,y,weight]

#パーティクルフィルタ用の尤度関数
def likelihood(x,y,color_save,frame_H,w=30,h=30):
    """
    尤度関数
    x,y : 走査する中心の座標
    w,h : 走査する範囲
    HSV : array : [min_H, max_H]
    image_h : HSVに直され、値Hのみをもつ二次元配列
    return : float : そのx,yでの尤度
    """
    x1 = math.floor((max(0, x - w / 2)))
    y1 = math.floor((max(0, y - h / 2)))
    x2 = math.floor(min(frame_H.shape[1], x + w / 2))
    y2 = math.floor(min(frame_H.shape[0], y + h / 2))
    
    #ここでHSV空間の範囲を決定する。
    #uHeは[0,179], Saturationは[0,255]，Valueは[0,255]の範囲の値をとる
    H = color_save[0]
    HSV = [0,0]
    HSV[0] = int(H - 10)
    if H - 10 < 0:
        HSV[0] = 0
    
    HSV[1] = int(H + 10)
    if H + 10 >179:
        HSV[1] = 179
    
    #走査する範囲を決定する。
    region = frame_H[y1:y2,x1:x2]
    #条件を満たすピクセル数をカウントする。複数条件に注意。
    count_1 = np.count_nonzero((HSV[0] < region) & (region < HSV[1]))
    
    HSV[0] = int(H - 3)
    if H - 3 < 0:
        HSV[0] = 0
    
    HSV[1] = int(H + 3)
    if H + 3 > 179:
        HSV[1] = 179
    #条件を満たすピクセル数をカウントする。複数条件に注意。
    count_2 = np.count_nonzero((HSV[0] < region) & (region < HSV[1]))
    size = abs(x1-x2)*abs(y1-y2)
    
    return (count_1 + count_2)/ size

def resample(particles,x,y):
    """
    パーティクルフィルタのリサンプリングをする関数
    乱数を使って成績の悪いパーティクルを淘汰して成績の良いパーティクルで置き換え
    x,yは算出した予測位置。悪いパーティクルはこの値に置き換えられる。
    """
    #重みが平均より小さいものは消去するリスト入り確定
    weight_ave = np.mean(particles[:, 2])
    wei_list_1 = np.where(particles[:,2]< weight_ave)
    
    #重みが大きくても予測位置ｊから離れた場所のものはリスト入り。
    wei_list_2 = np.where(abs(particles[:,0] - x) > 20)
    wei_list_3 = np.where(abs(particles[:,1] - y) > 20)
    
    #置換の作業
    for i in range(len(wei_list_1)):
        particles[:,0][wei_list_1[i]] = x
        particles[:,1][wei_list_1[i]] = y
    for i in range(len(wei_list_2)):
        particles[:,0][wei_list_2[i]] = x
        particles[:,1][wei_list_2[i]] = y
    for i in range(len(wei_list_3)):
        particles[:,0][wei_list_3[i]] = x
        particles[:,1][wei_list_3[i]] = y
        
    return particles

def predict(particles,x,y,variance=100):
    """
    次のフレームに向けてパーティクルを実際に動かす
    varianceはランダム具合である。
    """
    particles[:, 0] += np.random.randn((particles.shape[0])) * variance
    particles[:, 1] += np.random.randn((particles.shape[0])) * variance
    
    #以下で画像からはみ出したパーティクルを元の座標へ戻す。|はwhwereの時の複数条件。
    wei_list_2 = np.where((particles[:,0] < 0) | (particles[:,0] > 640))
    wei_list_3 = np.where((particles[:,1] < 0) | (particles[:,1] > 360))
    
    #置換の作業
    for i in range(len(wei_list_2)):
        particles[:,0][wei_list_2[i]] = x
        particles[:,1][wei_list_2[i]] = y
    for i in range(len(wei_list_3)):
        particles[:,0][wei_list_3[i]] = x
        particles[:,1][wei_list_3[i]] = y
    
    return particles

def weight(particles,color_save,frame_H):
    """
    重み関数の更新
    パーティクルごとの尤度を判定する。この重みはlikelihood関数を利用
    """
    for i in range(particles.shape[0]):
        particles[i][2] = likelihood(particles[i][0], particles[i][1],color_save,frame_H)
    #以下で規格化をしている。
    sum_weight = np.sum(particles[:, 2])
    #最初とか、おかしなときに0で割らないように以下の例外を設けておこう。
    if sum_weight < 0.001:
        sum_weight = 0.001
    
    particles[:, 2] = particles[:, 2] /sum_weight
    
    return particles

def measure(particles):
    """
    成績のよいパーティクルが集中している場所を割り出す
    """
    x = np.sum((particles[:, 0] * particles[:, 2]))
    y = np.sum((particles[:, 1] * particles[:, 2]))
    return x,y

def particle_init(x,y):
    """
    パーティクルフィルタ用の初期化関数。
    """
    particles = np.ndarray((500, 3), dtype=np.float32)
    #初期値を代入する
    particles[:,0] = x
    particles[:,1] = y
    particles[:,2] = 0
    return particles


def particle_predict(color_save,x_save,y_save,OSC,frame,frame_H,particles):
    """
    パーティクルフィルタの本体。この関数を利用する前に初期化が必要。
    dlibで更新された場合にはparticleはたまたま近くにあったもの以外すべて削除されつようになっている。
    """
    x = x_save
    y = y_save
    
    #まずはリサンプリングして不要なパーティクルを除去
    particles = resample(particles,x,y)
    #ランダムに移動して更新
    particles = predict(particles,x,y)
    #重みの更新
    particles = weight(particles,color_save,frame_H)
    #予測値の算出
    x_save,y_save = measure(particles)
    
    #可視化するためにframeの中にparticleの点を表示させる(Debug modeの時のみ)
    X = particles[:,0]
    Y = particles[:,1]
    if OSC == 0 or OSC == 2:
        for i in range(0,len(X)):
            cv2.circle(frame, (int(X[i]), int(Y[i])), 1, (255,0, 255), -1)
            cv2.circle(frame, (int(x_save), int(y_save)), 10, (255,0, 125), -1)
    
    #ここでOSC通信を行うことはできない。
    return x_save,y_save,particles


def main():
    try:
        # カメラにアクセスできるかを確認。webcameraであればどれでもこのサイズで読み込むようだ。
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        print("Webcamera ready")
    except:
        print("Can't access to webcamera")#前回異常終了した場合などにはアクセスが拒否される。
        cap.release()
        sys.exit()
        return 0
    #モードを選択する。pyファイルにして実装するときには以下の二行をOSC = 1と書き換えればよい。
    
    print("choose mode:::  OSC_mode : 1 Debug_mode : 0 dlib_mode :2")
    OSC = int(input())
    #初回の通信のみ通信を確立するシーケンスが必要
    if OSC == 1:
        print("starting OSC mode...")
    elif OSC == 2:
        print("starting dlib mode...")
    else:
        #モードがデバッグかそれ以外のときにはclientには意味をなさない0を入れておく。
        print("starting debug mode...")
        #バグ防止のために他の入力がされたときには強制でデバッグモードへ移行する。
        OSC = 0
        
    #ウェブカメラとのアクセスを確認したら処理開始
    dlib_facial_landmarks(OSC)
    
def mouth_aspect_ratio(mouth):
    #口が開いているのかをはんていする
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

def dlib_facial_landmarks(OSC):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
    	help="path to facial landmark predictor")
    ap.add_argument("-w", "--webcam", type=int, default=0,
    	help="index of webcam on system")
    args = vars(ap.parse_args())
    
    # define one constants, for mouth aspect ratio to indicate open mouth
    MOUTH_AR_THRESH = 0.79
    
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    
    # grab the indexes of the facial landmarks for the mouth
    (mStart, mEnd) = (49, 68)
    
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)
    
    frame_width = 640
    frame_height = 360
    #webカメラの画像を動画ファイルにして保存して、待機する。
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    time.sleep(1.0)
    
    #時間の計測を開始
    time_st = time.time()
    
    #最初に見つからないときにはパーティクルフィルタ用にとりあえずの初期設定をする。
    x_save = 0
    y_save = 0
    color_save = [0,0,0]
    particles = particle_init(x_save,y_save)
    
    #OSC通信の初期化設定。引数clientを渡すことはできないので要注意。
    if OSC ==1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=5005, help="The port the OSC server is listening on")
        args = parser.parse_args()
        client = udp_client.SimpleUDPClient(args.ip, args.port)
        print("OSC is ready (from python)")
        
    # loop over frames from the video stream
    while True:
    	# grab the frame from the threaded video file stream, resize
    	# it, and convert it to grayscale channels)
        frame = vs.read()
        frame_ori = copy.copy(frame)
        frame = imutils.resize(frame, width = 640)
        #HSV形式のフレームを作成(パーティクルフィルタ用)
        frame_HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #二次元配列にする
        frame_H = frame_HSV[:, :, 0]
        #グレースケールのフレームを作成(dlib用)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    	# detect faces in the grayscale frame
        rects = detector(gray,0)
        # loop over the face detections
        #print(len(rects)) #lenによって顔の数を求めることができる
        
        face_number = len(rects)        
        #最初は口は閉じているとしましょう。
        mouth_ = 0
        
        #顔が見つからないときの処理をする。
        if face_number == 0 and OSC == 0:
            #particle filterで予測して値を更新し、通信も行う。予測結果をreturnもしてくれる。
            #frame_oriで元の画像を入れるのを忘れるな
            (x_new,y_new,particle_tmp) = particle_predict(color_save,x_save,y_save,OSC,frame,frame_H,particles)
            #値の更新
            x_save = x_new
            y_save = y_new
            particles = particle_tmp
        
        for rect in rects:
            mouth_ = 0
            #二人以上写っている場合にはこのループを抜ける。
            if face_number !=1:
                break
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            (x,y,w,h) = face_utils.rect_to_bb(rect)
            # extract the mouth coordinates, then use the
            # coordinates to compute the mouth aspect ratio
            mouth = shape[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            # compute the convex hull for the mouth, then
    		# visualize the mouth
            mouthHull = cv2.convexHull(mouth)
            
            if OSC == 0 or OSC == 2:
                #OSC mode の時には負荷を減らすためにこの処理を実行しない
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Draw text if mouth is open
            if mar > MOUTH_AR_THRESH:
                #口が開いていることを保存
                mouth_ = 1
                if OSC == 0 or OSC == 2:
                    #OSC mode の時には負荷を減らすためにこの処理を実行しない
                    cv2.putText(frame, "Mouth is Open!", (30,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            j = 1
            #ここに入る前にとりあえずHSVを入れておこう。
            H = 0 
            S = 0
            V = 0
            
            #ランドマークを表示する
            for (x, y) in shape:
                if OSC == 0 or OSC == 2:
                    #debug modeの時には表示のための処理をする。
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
                    if j == 67:
                        #口の中心位置であって、この座標をUnityへ伝える。
                        cv2.circle(frame, (int(x), int(y)), 1, (255,0, 0), -1)
                        #print(frame_ori[y][x])
                        #通信で送るための座標を更新
                        x_save = x
                        y_save = y
                    #test用
                    if j == 5 or j== 13 or j== 30 or j == 31:
                        #色のサンプリングはHSVでやることに注意！
                        color_ori = frame_HSV[y][x]
                        H = H + color_ori[0]/4
                        S = S + color_ori[1]/4
                        V = V + color_ori[2]/4
                if OSC == 1:
                    #5,13,30のlandmarkの色を抽出することにしよう。
                    #OSC modeの時には色の抽出が必要なので以下の処理をする。R,G,Bの値平均を格納する。
                    if j == 5 or j== 13 or j== 30 or j == 31:
                        #値のぶれがそれなりにあるのでこれは要修正かも。
                        color_ori = frame_ori[y][x]
                        H = H + color_ori[0]/4
                        S = S + color_ori[1]/4
                        V = V + color_ori[2]/4
                    if j == 67:
                        #口の中心位置であって、この座標をUnityへ伝える。
                        #通信で送るための座標を更新
                        x_save = x
                        y_save = y
                    
                j += 1
            
            #パーティクルフィルタのための色を更新。顔検出できたら毎回更新する。
            color_save = [int(H),int(S),int(V)]
            #print(color_save)
            # Write the frame into the file 'output.avi'
            out.write(frame)
        
        #以上で一枚のフレームに対する画像処理は終了。以下、表示か通信かで処理が異なる。
        
        # show the frame
        if OSC == 0 or OSC == 2:
            #OSC mode の時には負荷を減らすためにこの処理を実行しない
            print(x_save)
            cv2.imshow("Frame", frame)
        
        #OSC通信で送信する。送信はここで一括に行うので他の場所でやらない。
        if OSC ==1:
            #ここで型を明示して、intで統一する。
            mouth_ = int(mouth_)
            x_save = int(x_save)
            y_save = int(y_save)
            client.send_message("/volume", [mouth_,x_save,y_save])
        #print(color_save)
        # if the `q` key was pressed, break from the loop デバッグ用。
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("keyboard interpreted")
            break
        
        #5秒ごとに監視しているunityのResourcesフォルダの中にあるLog.txtファイルの中身が変更されたら強制終了する。
        time_now = time.time()
        if time_now - time_st > 5:
            #時間を5秒追加
            time_st += 5
            print("Logfile checked")
            #ファイルを確認する
            Logfilepath = "C:/unity2019/OSC/Gametest/TestGame/Assets/Resources/Logfile.txt"
            #読み取り専用で開くutf8ででコードすることに注意する。
            #さらにunityとの衝突時にはさらに5秒後にもう一度試すようにする。
            try:
                f = open(Logfilepath,'r',encoding="utf-8")
                data = f.read()
                #すぐに閉じてunityとの衝突を防ぐ
                f.close()
                data_list = data.split('\n')
                if len(data_list) > 1:
                    #実は上書き保存されてしまうので一行のみのファイルに戻すことができる。
                    #unityから行を追加されたらこちらも終了する仕組みである。速度が出ないのでゲームのコントロールには不向きだが
                    #通信をしなくていいので面倒な設定が不要である。
                    f = open(Logfilepath,'w',encoding="utf-8")
                    f.write("done")
                    f.close()
                    print("finish")
                    break
            except:
                #たまたまunityと衝突してしまった。
                print("collision occured")
                pass
    #不要なウィンドウを消去
    cv2.destroyAllWindows()
    vs.stop()
    

if __name__ == '__main__':
    main()
