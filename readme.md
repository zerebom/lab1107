## 腎臓がんの自動識別用ライブラリ

## usage
ymlを書く  
python3 ./src/Keras/run_unet_3d_med.py -yml で学習  
pred.sh YML GPU_NUM で推論  

## caution
kldなどで生データじゃなくて加工後のバッチデータを使うときはymlに
NII_GZを書くこと。データフォルダを記載すること

sudo rm -rf ./*/tumor*standard*
sudo rm -rf /home/kakeya/ssd/strange/*/tumor*

for i in `seq -w 000 160`; do
cd /home/kakeya/ssd/strange/00${i}
pwd
sudo python3 /home/kakeya/Desktop/higuchi/20191107/src/create_patch2.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --size 60 60 20 
sudo python3 /home/kakeya/Desktop/higuchi/20191107/src/create_patch2.py SE2.nii.gz SE3.nii.gz kidney.nii.gz CCRCC.nii.gz cyst.nii.gz --size 48 48 16 