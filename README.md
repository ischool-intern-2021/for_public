# Face tracking game

以下のような顔追跡ゲームをpythonのインストールなしに遊ぶことができます。

You can play our face tracking game without installing python.

# Demo(デモ動画)

![video](https://user-images.githubusercontent.com/70675089/92320727-7fcdb080-f05e-11ea-894d-3267a61a8c0c.gif)

# How to play(ゲームの遊び方)

ゲームを起動してしばらくするとスタート画面になります。キーボードのsボタンを押すとゲーム画面になります。ゲーム画面では顔を動かすことでゲーム内のキャラクターを動かすことができます。3回敵に当たるとゲームオーバー、お菓子をゲットすると体力が回復します。

After starting this game,please wait until the game screen is popped up. You can play this game by pushing keyboard "s". The main character is tracking your face in game screen. If you hit enemies for three times, game will be over. If you can get sweets, you recover HP. 

# Requirement(システム要件)

* windows10 
* webcamera(ウェブカメラ)

# Usage(使い方)

すべてのファイルをダウンロードし、パソコンにウェブカメラが認識されていることを確認。その後game_folder.zipを展開し、install.cmdをダブルクリック。これで準備が完了するのでその後game.cmdをダブルクリック。

Please use windows10 with webcamera. Download all the files, unzip game_folder.zip, double click "install.cmd" and finally, double click "game.cmd".

* 詳細な説明

まずパソコンにウェブカメラが認識されていることを確認。その後下の画像を参考にzipをダウンロード

![1](https://user-images.githubusercontent.com/70675089/93488607-83f0ac80-f941-11ea-9c06-f7c0d50edca2.PNG)

次に下の画像を参考に展開


![2](https://user-images.githubusercontent.com/70675089/93488609-84894300-f941-11ea-93e0-6010bf869c83.PNG)

下記を展開


![3](https://user-images.githubusercontent.com/70675089/93488611-84894300-f941-11ea-93e2-df6715c542c1.PNG)

展開がきちんと終了してからinstall.cmdをダブルクリック


![4](https://user-images.githubusercontent.com/70675089/93488614-8521d980-f941-11ea-8729-99e83e5b2bdd.PNG)

game.cmdをダブルクリックしてゲーム開始


![5](https://user-images.githubusercontent.com/70675089/93488601-83581600-f941-11ea-9fda-f3f1260c46ba.PNG)

# Attention(注意点)

* Zipの展開するディレクトリの絶対パスに日本語が入っているとバグが生じます。よってドライブ直下に専用のフォルダを作ってそこで展開することをお勧めします。

Please do not unzip the folder whose absolute pathname contains Japanese character. 

* Windows10以前での動作保証はしていません。macやlinuxでは動作しません。

We do not support environments other than windows 10.

* ゲームの動作はcpuとウェブカメラの性能に大きく依存します。デモ動画のようにスムーズなトラッキングができない場合もあります。

Game performance strongly depends on cpu and webcamera.

* mac linuxでの動作には、

If you want to play the game on mac or linux,

1,python3.7の専用仮想環境を作成(このプログラムを動作させる目的でライブラリをインストールして既存の環境と衝突した場合、一切の責任を負いかねます)

Please make a new virtual python 3.7 environment on your computer with minimum libraries.

2,その仮想環境に必要なライブラリをpipでインストールする。

Then, install necessary libraries via pip.

3,pythonファイルを実行

Execute "gif.py".

# Bugs(不具合など)

* 何らかの不具合が生じた場合はwindowsの更新、再起動をしてみてください

If you have any trouble and can not play this game, please reboot your PC.

* 展開したファイルをエクスプローラー等で開いてパスを確認して日本語が入ってないかチェックしてみてください

Please do not unzip the folder whose absolute pathname contains Japanese character. 

# License(ライセンスや著作権)

* Unityアセットストアの素材を購入、利用しています。unityライセンスに準拠して作成しています。

We use unity asset to make this game under unity asset licence.

https://unity3d.com/legal/as_terms

* ライセンスはMITライセンスに準拠しています。詳細は以下で英語で示します。

Copyright(c) 2020 Kougaku-Hakurankai 2020.
Released under MIT license

https://opensource.org/licenses/mit-license.php

Authors:
MIT License  Copyright (c) 2018 Ross Mauck

https://opensource.org/licenses/mit-license.php

https://github.com/mauckc/mouth-open

* dlibを利用しています。ライセンスは以下になります。

license of dlib C++ library:

http://dlib.net/license.html

* このプログラムにおいてはdlibの商用利用に制限のある学習済みモデルを利用しています。詳細は以下のリンクを参照してください。

Do not use this game for commercial use. For more information,

http://dlib.net/face_landmark_detection.py.html


* このプログラムによって生じたいかなる不利益にも責任を負いかねます。

We assume no responsibility whatsoever for any direct or indirect damage, loss, or emotional distress caused by this game.

* このプログラムおよびアプリは工学博覧会開催の趣旨に準拠し,個人利用を目的に作成しています。

We made this game for Kougaku-Hakurankai 2020(online). It is for personal use, not for commercial use.
