import argparse
import pathlib
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import yaml
from pathlib import Path
'''
The purpose of this is to evaluate med of tsukuba data.
python3 /home/kakeya/Desktop/higuchi/20191021/Keras/src/test/evaluate_tsukuba_med.py /home/kakeya/Desktop/higuchi/20191021/Keras/experiments/1104_75epochs
'''


def get_yml(args):
    with open(args.setting_yml_path) as file:
        return yaml.load(file)


def ParseArgs():
    parser = argparse.ArgumentParser()
    # dile_parser.add_argument('file_dir')
    parser.add_argument('--output_file', default='lesion_evaluation.csv')
    parser.add_argument('-yml', '--setting_yml_path', type=str)
    parser.add_argument('--res_path', default=None)

    args = parser.parse_args()
    return args


def ValidateArgs(args, yml):
    OWN_DIR = Path(yml['DIR']['OWN'])

    if not OWN_DIR.is_dir():
        print(f'Evaluated directory({OWN_DIR}) is not found or not directory.')
        return False
    if not (OWN_DIR / 'ref').is_dir():
        print(f'Reference directory({OWN_DIR / "ref"}) is not found or not directory.')
        return False
    if not (OWN_DIR / 'res').is_dir():
        print(f'Resource directory({OWN_DIR / "res"}) is not found or not directory.')
        return False

    return True


def CalcMetrics(ref_array, res_array):
    share_array = np.logical_and(ref_array > 0, res_array > 0)
    dice = 2 * np.count_nonzero(share_array) / (np.count_nonzero(ref_array) + np.count_nonzero(res_array)
                                                ) if np.count_nonzero(ref_array) + np.count_nonzero(res_array) != 0 else 0.0
    recall = np.count_nonzero(share_array) / np.count_nonzero(ref_array) if np.count_nonzero(ref_array) != 0 else 0.0
    precision = np.count_nonzero(share_array) / np.count_nonzero(res_array) if np.count_nonzero(res_array) != 0 else 0.0
    return dice, recall, precision


def EvaluateMain(csv_file, reference_file, resource_file):
    ref_image = sitk.ReadImage(str(reference_file))
    res_image = sitk.ReadImage(str(resource_file))
    ref_array = sitk.GetArrayFromImage(ref_image)
    res_array = sitk.GetArrayFromImage(res_image)

    tqdm.write(f'Evaluate each island of {reference_file}.')
    label_names = ['bg', 'HCC', 'cyst', 'angioma']
    for i in range(1, len(label_names)):
        ref_i_array = np.where(ref_array == i, ref_array, 0)
        '''ref_i_image = sitk.GetImageFromArray(ref_i_array)
        ref_i_image.CopyInformation(ref_image)

        identified_ref_image = sitk.ConnectedComponent(ref_i_image)
        identified_ref_array = sitk.GetArrayFromImage(identified_ref_image)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(identified_ref_image, ref_image)'''

        res_i_array = np.where(res_array == i, res_array, 0)

        if np.count_nonzero(ref_i_array) == 0:
            csv_file.write(f'{reference_file.name},{label_names[i]}, "0", "", "", ""\n')

        else:
            dice, recall, precision = CalcMetrics(ref_i_array, res_i_array)
            csv_file.write(f'{reference_file.name},{label_names[i]}, "1",{dice},{recall},{precision}\n')

        '''res_i_image = sitk.GetImageFromArray(res_i_array)
        res_i_image.CopyInformation(res_image)

        identified_res_image = sitk.ConnectedComponent(res_i_image)
        identified_res_array = sitk.GetArrayFromImage(identified_res_image)

        if stats.GetLabels():
            tqdm.write(f'label: {label_names[i]}')
        # 腫瘍1つの正解領域（正解island）ごとに算出する
        for ref_island_label in tqdm(stats.GetLabels()):
            # 正解islandの体積
            size = stats.GetPhysicalSize(ref_island_label)
            # 正解islandの領域
            ref_island_array = identified_ref_array == ref_island_label

            # 正解islandと重なる予測領域のボクセル群
            share_array = identified_res_array[ref_island_array]
            # 正解islandと重なる予測領域のラベル値リスト
            share_island_labels = np.unique(share_array[share_array.nonzero()])
            # 正解islandの算出で関係する予測island群
            res_island_array = np.isin(identified_res_array, share_island_labels)

            dice, recall, precision = CalcMetrics(ref_island_array, res_island_array)
            res_labels = str(share_island_labels.tolist()).replace(',', '')
            island_type = np.unique(ref_array[ref_island_array])

            tqdm.write(f'  dice: {dice}, recall: {recall}, precision: {precision}')
            csv_file.write(f'{reference_file.name},{ref_island_label},{size},{dice},{recall},{precision},{res_labels},{island_type}\n')'''

    # identified_res_image = sitk.ConnectedComponent(res_image)
    # identified_res_array = sitk.GetArrayFromImage(identified_res_image)
    # identified_ref_image = sitk.ConnectedComponent(ref_image)
    # identified_ref_array = sitk.GetArrayFromImage(identified_ref_image)
    # dice_per_case, recall_per_case, precision_per_case = CalcMetrics(identified_ref_array, identified_res_array)
    #dice_per_case, recall_per_case, precision_per_case = CalcMetrics(ref_array, res_array)

    # sitk.WriteImage(identified_ref_image, 'identified/identified_ref_' + reference_file.name, True)
    # sitk.WriteImage(identified_res_image, 'identified/identified_res_' + resource_file.name,  True)
    csv_file.flush()

    return 0, 0, 0, 0


def main(args, yml):
    OWN_DIR = Path(yml['DIR']['OWN'])
    reference_file_list = sorted((OWN_DIR / 'ref').glob('*.nii.gz'))

    with open(OWN_DIR / args.output_file, 'w') as f:
        f.write('filename,label_name,existence,dice,recall,precision\n')
        score_per_case_list = []

        for reference_file in tqdm(reference_file_list):
            if args.res_path:
                resource_file = pathlib.Path(args.res_path) / reference_file.name
            else:
                resource_file = OWN_DIR / 'res' / reference_file.name
            if not resource_file.is_file():
                tqdm.write(f'Resource file({resource_file}) is not found or not file.')
                continue

            reference_file_name, dice_per_case, recall_per_case, precision_per_case = EvaluateMain(
                f, reference_file, resource_file)
            # score_per_case_list.append(f'{reference_file.name},{dice_per_case},{recall_per_case},{precision_per_case}\n')

        return
        f.write('\nfilename,dice_per_case,recall_per_case,precision_per_case\n')
        for score_per_case in score_per_case_list:
            f.write(score_per_case)


if __name__ == '__main__':
    args = ParseArgs()
    yml = get_yml(args)
    if ValidateArgs(args, yml):
        main(args, yml)
