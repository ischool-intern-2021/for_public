# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 08:04:25 2020

Copyright(c) 2020 Taichi Iizuka
Released under MIT license

AUTHORS:
MIT License
Copyright (c) 2018 Ross Mauck
"""

# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
#from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import sys
import math
#テキストファイルの監視時間の制御
import time


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

#opencvでgifをフレーム毎に読み込んで画像の配列にしている。

def vread(path, T):
    cap = cv2.VideoCapture(path)
    gif = [cap.read()[1][:,:,::1] for i in range(T)]
    gif = np.array(gif)
    cap.release()
    return gif

'''
#gifをnumpy配列にするための命令文
T = 13
gif = vread("test.gif",T)
np.save('np_data',gif)
data = np.load('np_data.npy')
print(data)
'''

#以下はgifを透過して背景に重ねて表示するプログラム。iによってgif画像の連番を制御。Tを与えて使用するgif画像の
#画像枚数を指定する必要がある。

def gousei(i,x,y,back,gifname,T):
    """
    i:gif画像の連番
    x,y:貼り付ける画像の左上の座標
    back:合成するバックの画像
    gifname:GIF画像のpath
    T : 使用するgif画像のフレーム数
    
    return:透過した画像を合成したもの.
    opencvでの表示は
    左上が(0,0)で右下が(image.shape[1],image.shape[0])になる。
    """
    #gif = vread(gifname,T)
    gif = np.load('np_data.npy')
    h = int(back.shape[0])#y600
    w = int(back.shape[1])#x800
    p_h = int(gif[i].shape[0])#y186
    p_w = int(gif[i].shape[1])#x248
    
    #以下ではみ出さないようにするための処理。
    if 0 < x and 0 < y and p_h + y < h and p_w + x < w:
        pass
    else:
        if x <= 0:
            x = 0
        if y <= 0:
            y = 0
        if  y + p_h >= h:
            y = h - p_h
        if x + p_w >= w:
            x = w - p_w
    
    #グレー画像を作成
    img2gray = cv2.cvtColor(gif[i],cv2.COLOR_BGR2GRAY)
    #反転させる
    mask_inv = cv2.bitwise_not(img2gray)
    #白塗りの画像を作る
    white_background = np.full(img2gray.shape,255,dtype=np.uint8)
    #キャラの部分を白抜きにしたマスク
    bk = cv2.bitwise_or(white_background,white_background,mask=mask_inv)
    #キャラの外側を白抜きにしたマスク
    ck = cv2.bitwise_not(bk)
    #キャラの外側を黒塗りに
    fg = cv2.bitwise_or(gif[i],gif[i],mask=mask_inv)
    #合成する領域をROIとして取り出す。
    roi = back[y:y+p_h,x:x+p_w,:]
    #キャラの場所を白でくりぬいておく
    new = cv2.bitwise_or(roi,roi,mask=ck)
    #最後に合成する
    final_roi = cv2.bitwise_or(new,fg)
    #外側も含めて合成
    back[y:y+p_h,x:x+p_w,:] = final_roi
    return back

def gousei_png(x,y,back,filename):
    """
    背景透明のpng画像を合成する関数。
    x,y:貼り付ける画像の左上の座標
    back:合成するバックの画像
    filename:画像のpath
    return:透過した画像を合成したもの.
    opencvでの表示は
    左上が(0,0)で右下が(image.shape[1],image.shape[0])になる。
    """
    gif = cv2.imread(filename)
    h = int(back.shape[0])#y600
    w = int(back.shape[1])#x800
    p_h = int(gif.shape[0])#y186
    p_w = int(gif.shape[1])#x248
    
    #以下ではみ出さないようにするための処理。
    if 0 < x and 0 < y and p_h + y < h and p_w + x < w:
        pass
    else:
        if x <= 0:
            x = 0
        if y <= 0:
            y = 0
        if  y + p_h >= h:
            y = h - p_h
        if x + p_w >= w:
            x = w - p_w
    img1 = back
    img2 = gif
    rows,cols,channels = img2.shape
    roi = img1[y:rows+y, x:cols+x ]
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    img1[y:rows+y, x:cols+x ] = dst
    return img1
"""
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
"""
def main():
    try:
        # カメラにアクセスできるかを確認。
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        print("Webcamera ready")
    except:
        print("Can't access to webcamera")#前回異常終了した場合などにはアクセスが拒否される。
        cap.release()
        sys.exit()
        return 0
    OSC = 0
    #ウェブカメラとのアクセスを確認したら処理開始
    dlib_facial_landmarks(OSC)
    

def dlib_facial_landmarks(OSC):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
    	help="path to facial landmark predictor")
    ap.add_argument("-w", "--webcam", type=int, default=0,
    	help="index of webcam on system")
    args = vars(ap.parse_args())
    
    # define one constants, for mouth aspect ratio to indicate open mouth
    #MOUTH_AR_THRESH = 0.79
    
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
    
    #最初に見つからないときにはパーティクルフィルタ用にとりあえずの初期設定をする。
    x_save = 0
    y_save = 0
    color_save = [0,0,0]
    particles = particle_init(x_save,y_save)
    
    #以下はゲームのための初期設定。gif画像の連番を制御する。
    player_i = 0
    #以下はHP
    HP = 3
    #以下は敵ないし回復アイテムが出現するインターバルの最大値。最初の敵はゲーム開始後この秒数で呼ばれる。
    int_random = 5
    #以下は敵と回復アイテムの格納場所
    Sweet = []
    Enemy = []
    #ダメージエフェクトを管理
    damage = 0
    
    #ゲームの進行を制御する変数
    status = 0
    while True:
        start_frame = cv2.imread("back_resize.png")
        cv2.putText(start_frame,"Press S to start", (10,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),2)
        gousei_png(150,250,start_frame,"start_resize.png")
        cv2.imshow("Start", start_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Finish")
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            #時間の計測を開始
            time_st = time.time()
            #状態遷移
            status = 1
            #スタート外面消去
            cv2.destroyAllWindows()
            break
       
    # loop over frames from the video stream
    while True and status ==1:
    	# grab the frame from the threaded video file stream, resize
    	# it, and convert it to grayscale channels)
        frame = vs.read()
        #左右反転
        frame = cv2.flip(frame, 1)
        #frame_ori = copy.copy(frame)
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
        #mouth_ = 0
        
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
            #mouth_ = 0
            #二人以上写っている場合にはこのループを抜ける。
            if face_number !=1:
                break
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            (x,y,w,h) = face_utils.rect_to_bb(rect)

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
                        #座標を更新
                        x_save = x
                        y_save = y
                    #test用
                    if j == 5 or j== 13 or j== 30 or j == 31:
                        #色のサンプリングはHSVでやることに注意！
                        color_ori = frame_HSV[y][x]
                        H = H + color_ori[0]/4
                        S = S + color_ori[1]/4
                        V = V + color_ori[2]/4
                    
                j += 1
            
            #パーティクルフィルタのための色を更新。顔検出できたら毎回更新する。
            color_save = [int(H),int(S),int(V)]
            #print(color_save)
            # Write the frame into the file 'output.avi'
            out.write(frame)
        
        #以上で一枚のフレームに対する画像処理は終了。
        time_now = time.time()
        
        player_i += 1
        if player_i == 13:
            player_i = 0
        if OSC == 0 or OSC == 2:
            #OSC mode の時には負荷を減らすためにこの処理を実行しない
            #ここでキャラの描画及びゲームの実装をする
            #背景をスタート画面と合成する
            back_gamen = cv2.imread("back_resize.png")
            ################################################3
            frame = cv2.addWeighted(src1=frame,alpha=0,src2=back_gamen,beta=0.7,gamma=0)
            #ゲーム画面
            gamen = "HP :" + str(HP) 
            cv2.putText(frame,gamen, (30,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),2)
            
            #プレーヤーの描画
            player = "test.gif"
            gousei(player_i,int(x_save-70),int(y_save-70),frame,player,13)
            
            
            if time_now - time_st > int_random:
                #時間を追加
                time_st += int_random
                int_random = np.random.randint(1,5)
                
                random_y = np.random.randint(400)
                if np.random.randint(100) > 80:
                    number = np.random.randint(1,8)
                    Sweet.append([550,random_y,number])
                else:
                    number = np.random.randint(1,3)
                    Enemy.append([550,random_y,number])
                
            #お菓子の描画
            for S in Sweet:
                sweet_x = S[0]
                sweet_y = S[1]
                filename = "sweets/sweet" + str(S[2]) + ".png"
                gousei_png(int(sweet_x),int(sweet_y),frame,filename)
                S[0] -= 13
                if S[0] < 20:
                    #左端へ行ったら自動で削除
                    Sweet.remove(S)
                #衝突判定    
                if np.sqrt((int(x_save-30) - sweet_x)**2 + (int(y_save) - sweet_y)**2) < 100:
                    HP += 1
                    Sweet.remove(S)
            #敵の描画
            for E in Enemy:
                enemy_x = E[0]
                enemy_y = E[1]
                filename = "enemy/enemy" + str(E[2]) + ".png"
                gousei_png(int(enemy_x),int(enemy_y),frame,filename)
                E[0] -= 20
                if E[0] < 20:
                    #左端へ行ったら自動で削除
                    Enemy.remove(E)
                #衝突判定    
                if np.sqrt((int(x_save) - enemy_x)**2 + (int(y_save) - enemy_y)**2) < 150:
                    HP -= 1
                    Enemy.remove(E)
                    damage = 10
            #ダメージの描画
            if damage >0:
                damage -= 1
                filename = "damage.png"
                gousei_png(int(x_save+40),int(y_save-30),frame,filename)    
            #ゲームオーバーかを判定する
            if HP < 0:
                status = 2
                break
            #描画
            cv2.imshow("Frame", frame)
        # if the `q` key was pressed, break from the loop デバッグ用。
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("keyboard interpreted")
            break
    while status ==2:
        cv2.putText(frame,"Game Over! press q to quit", (10,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("keyboard interpreted")
            break
    #不要なウィンドウを消去
    cv2.destroyAllWindows()
    vs.stop()

#実行する関数
main()

#if __name__ == '__main__':
#    main()