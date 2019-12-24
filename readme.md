## 腎臓がんの自動識別用ライブラリ

## usage
ymlを書く  
python3 ./src/Keras/run_unet_3d_med.py -yml で学習  
pred.sh YML GPU_NUM で推論  

## caution
kldなどで生データじゃなくて加工後のバッチデータを使うときはymlに
NII_GZを書くこと。データフォルダを記載すること