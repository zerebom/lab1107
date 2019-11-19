import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import SimpleITK as sitk
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


'''
ymlを書き換えるだけで適切な場所にplotが保存されるようにする。
出力DFをきれいに書き換える
'''
sns.set(font_scale=1.5)

pd.set_option('display.max_columns', 100)
plt.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = 800

def acc_plot(df,save_path=None,cols=['val_hcc_dice','val_cyst_dice','val_angioma_dice'],legends=['kidney','cyst','ccrcc']):
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    for col,legend in zip(cols,legends):
        ax.plot(df[col],label=legend)
    ax.set_xlabel('plot')
    ax.set_ylabel('Dice')
    plt.legend()
    plt.tight_layout()
    if save_path!=None:
        plt.savefig(save_path)
    plt.show()

def plot_whole_lesion_dice(df,save_dir=None):
    '''全症例に対してDiceスコアを出力する'''
    sns.set(style="whitegrid")
    plt.figure(figsize=(14,5))
    sns.set(font_scale=1.5) 

    # df[df['label_name']=='cyst'].sort_values('dice')
    sns.barplot(x='cid',y='dice',data=df,hue='label_name')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/whole_dice.png')
    plt.show()

def plot_each_lesion_dice(df,save_dir=None):
    for leison in ['kidney','CCRCC','Cyst']:
        plt.figure(figsize=(12,6))
        df2=df.query('existence==1 & label_name==@leison')[['dice','cid']].sort_values('dice').reset_index(drop=True)
        #df2=df.loc[df['label_name']==leison,['dice','cid']].sort_values('dice').reset_index(drop=True)
        plt.rcParams["font.size"] = 22
        sns.barplot(x='cid',data=df2,y='dice',order=df2['cid'],palette='GnBu_d')
        plt.title(f'{leison}_dice')
        plt.ylim(0,1)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/{leison}_dice.png')
        plt.show()

csv_path='/home/kakeya/Desktop/higuchi/20191107/experiment/single_channel/2019-11-09_16-37/epoch_results.csv'
df=pd.read_csv(csv_path)
df.head()

acc_plot(df,'/home/kakeya/Desktop/higuchi/20191107/experiment/single_channel/plot/dice.png')


csv_path='/home/kakeya/Desktop/higuchi/20191107/experiment/single_channel_HE05/lesion_evaluation.csv'
df=pd.read_csv(csv_path)
df=preprocess_lesion_df(df)
