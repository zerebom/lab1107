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
from pathlib import Path
import os
import argparse
import yaml
import plotly

'''
ymlを書き換えるだけで適切な場所にplotが保存されるようにする。
出力DFをきれいに書き換える
'''
sns.set(font_scale=1.5)

pd.set_option('display.max_columns', 100)
plt.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = 800

def ParseArgs():
    # from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml','--setting_yml_path',type=str,default='/home/kakeya/Desktop/higuchi/20191107/experiment/hist_equal05/setting.yml')
    args = parser.parse_args()
    return args

def preprocess_lesion_df(df):
    '''lesion_dfの前処理'''
    preprocess=lambda x:int(x.split('.')[0])
    df['cid']=df['filename'].apply(preprocess)
    df.loc[df['existence'].str.contains('0'),['existence','dice','recall','precision']]='0'
    df.loc[df['existence'].str.contains('1'),'existence']='1'
    df['existence']=df['existence'].astype(int)
    df['label_name']=df['label_name'].replace({'HCC': 'kidney', 'cyst': 'CCRCC','angioma':'Cyst'})
    df.iloc[:,3:6]=df.iloc[:,3:6].astype(float)
    return df

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
  

def plot_each_lesion_dice(df,save_dir=None):
    for leison,c_col in zip(['kidney','CCRCC','Cyst'],['count','count_CCR','count_cys']):
        plt.figure(figsize=(15,6))
        df2=df.query('existence==1 & label_name==@leison')[['dice','cid',c_col]].sort_values('dice').reset_index(drop=True)
        #df2=df.loc[df['label_name']==leison,['dice','cid']].sort_values('dice').reset_index(drop=True)
        plt.rcParams["font.size"] = 18
        g=sns.barplot(x='cid',data=df2,y='dice',order=df2['cid'],palette='GnBu_d')
        for index,row in df2.iterrows():
            g.text(index,row.dice+0.01,f'{str(row[c_col])[0]}e{len(str(int(row[c_col])))}',color='black',ha='center')
        plt.title(f'{leison}_dice')
        plt.ylim(0,1)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/{leison}_dice.png')


def scatter_plot(df,x,y,text,size,title):
    fig = px.scatter(df, x=x, y=y,text=text, size=size,size_max=30,hover_name='cid')
    fig.update_layout(title_text=title,titlefont={"size": 25},font={'size':18})
    fig.update_xaxes(title_font=dict(size=23, family='Courier'))
    fig.update_yaxes(tickfont=dict(family='Courier', size=20))
    fig.update_xaxes(tickfont=dict(family='Courier', size=20))
    fig.update_yaxes(title_font=dict(size=23, family='Courier'))
    plotly.offline.plot(fig, image_filename=title, image='jpeg')

def main(args):
    statistics_path='/home/kakeya/Desktop/higuchi/20191021/output/statistics.csv'
    if not Path(statistics_path).is_file():
        statistics_path='/home/higuchi/Desktop/higuchi/lab1107/output/csv/statistics.csv'
    with open(args.setting_yml_path) as file:
        yml = yaml.load(file)
        ROOT_DIR = yml['DIR']['ROOT']
        DATA_DIR = yml['DIR']['DATA']
        OWN_DIR=Path(yml['DIR']['OWN'])


    epoch_results_path= sorted(OWN_DIR.glob(f'**/epoch_results.csv'))[-1]
    print('use_epoch_result_csv:',epoch_results_path)
    epoch_df=pd.read_csv(epoch_results_path)
    if not Path(OWN_DIR/'plot').is_dir():
        Path(OWN_DIR/'plot').mkdir()
    acc_plot(epoch_df,OWN_DIR/'plot/dice.png')

    lesion_path=OWN_DIR/'lesion_evaluation.csv'
    lesion_df=pd.read_csv(lesion_path)
    lesion_df=preprocess_lesion_df(lesion_df)
    



    
    st_df=pd.read_csv(statistics_path,index_col=0).reset_index().rename(columns={'index':'cid'})

    lesion_df=lesion_df.merge(st_df,how='left',on='cid')
    plot_whole_lesion_dice(lesion_df,save_dir=OWN_DIR/'plot/')
    plot_each_lesion_dice(lesion_df,save_dir=OWN_DIR/'plot/')
    ccr_cols=[col for col in lesion_df.columns if  'CCR' in col]
    cys_cols=[col for col in lesion_df.columns if  'cys' in col]
    important_cols=['cid','dice','recall','precision','count','lumi_mean']


    ccr_df=lesion_df.query('existence==1 & label_name=="CCRCC"')[important_cols+ccr_cols]
    cys_df=lesion_df.query('existence==1 & label_name=="Cyst"')[important_cols+cys_cols]
    ccr_df['log_kid']=np.log10(ccr_df['count']+1)
    ccr_df['log_ccr']=np.log10(ccr_df['count_CCR']+1)
    ccr_df['round_dice']=np.round(ccr_df['dice'],3)
    cys_df['log_kid']=np.log10(cys_df['count']+1)
    cys_df['log_cys']=np.log10(cys_df['count_cys']+1)
    cys_df['round_dice']=np.round(cys_df['dice'],3)

    txt_path=OWN_DIR/'plot/accracy.txt'

    with open(txt_path,'w') as f:
        ccrcc_acc1=(ccr_df['dice']*ccr_df['count_CCR']).sum()/ccr_df['count_CCR'].sum()
        cys_acc1=(cys_df['dice']*cys_df['count_cys']).sum()/cys_df['count_cys'].sum()
        ccrcc_acc2=ccr_df['dice'].mean()
        cys_acc2=cys_df['dice'].mean()
        f.write(f'病変ごとの精度_ccrcc:{ccrcc_acc1}\n')
        f.write(f'病変ごとの精度_cyst:{cys_acc1}\n')
        f.write(f'患者ごとの精度_ccrcc:{ccrcc_acc2}\n')
        f.write(f'患者ごとの精度_cyst:{cys_acc2}\n')

    # scatter_plot(ccr_df,'log_kid','log_ccr','round_dice','dice','CCRCC_dice_with_dice')
    # scatter_plot(ccr_df,'log_kid','log_ccr','cid','dice','CCRCC_dice_with_cid')
    # scatter_plot(cys_df,'log_kid','log_cys','round_dice','dice','cyst_dice_with_dice')
    # scatter_plot(cys_df,'log_kid','log_cys','cid','dice','cyst_dice_with_cid')

if __name__== '__main__':
    args = ParseArgs()
    main(args)