from pathlib import Path
from ruamel.yaml import YAML, add_constructor, resolver
from collections import OrderedDict
import argparse

'''
学習後Yamlに同ディレクトリ内の、一番最新かつ、エポックの進んだ
重みファイルを書き込むコード。
順序を保持するためにruamle.yamlを使用する。
'''

def ParseArgs():
    parser=argparse.ArgumentParser()
    parser.add_argument('-yml',type=str)
    args = parser.parse_args()
    return args

args=ParseArgs()
yml=args.yml
hdf5=str(sorted(list(Path(yml).parent.glob("**/*.hdf5")))[-1])

# 入力時に順序を保持する
add_constructor(resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)))

yaml = YAML()
yaml.default_flow_style = False

# ファイルから入力
with open(yml,'r') as file:
    yml2=yaml.load(file)
    yml2['PRED_WEIGHT']=hdf5
    print(yml2)

with open(yml,'w') as file:
    yaml.dump(yml2, file)