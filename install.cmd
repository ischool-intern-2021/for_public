# このファイルの位置を作業ディレクトリに
cd /d %~dp0
move shape_predictor_68_face_landmarks.dat game_folder
del /f "%~dp0%~nx0"