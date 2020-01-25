#!/bin/bash

<< COMMENTOUT
deedsで患者Aと患者Bで位置合わせをするスクリプト
事前に位置合わせする患者のxyzすべてのshapeを合わせて置く必要がある。
また、deeds_pair.txtにどの患者同士をマッチングさせるか記載しておく必要がある。
このコードでは、保存先のディレクトリを新しく作成する。

usage:
sh deeds.sh /home/kakeya/ssd/deeds_data_130 /home/kakeya/Desktop/higuchi/20191107/output/txt/deeds_pair.txt SE2.nii.gz

参考: https://www.server-memo.net/shellscript/read-file.html

COMMENTOUT


data_dir=$1
pair_txt=$2
#SE2.nii.gz
# filename=$3
#ゼロ埋め
# dst_id=$(printf "%05d\n" "${dst_id}")
# src_id=$(printf "%05d\n" "${src_id}")

while read line
do  
    #set...空白区切りで文章を取得する
    set ${line}
    dst_id=${1}
    src_id=${2}
    save_dir=${data_dir}/deformed/${dst_id}_${src_id}
    for filename in SE2.nii.gz SE3.nii.gz ; do
        mkdir $save_dir
        #linearBCV -F $data_dir/$src_id/$filename -M $data_dir/$dst_id/$filename -O $save_dir/affine
        deedsBCV -F $data_dir/$src_id/$filename -M $data_dir/$dst_id/$filename -O $save_dir/$filename -S $data_dir/$dst_id/CCRCC.nii.gz $data_dir/$dst_id/kidney.nii.gz #-A save_dir/affine_matrix.txt
        done
done <$pair_txt