import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings, time

warnings.filterwarnings("ignore")

class logger():
    def __init__(self, ):
        super(logger, self).__init__()
        self.result = []

    def updating(self, currn_dict):
        self.result.append(currn_dict)

    def show(self, columns):
        return pd.DataFrame(self.result, columns=columns)

    def save(self, path, columns):
        pd.DataFrame(self.result, columns=columns).to_csv(os.path.join(path, 'process_result.csv'), index=False)

def result_prediction(test_name, pred_y, save_root):
    name = test_name[0].split('_')[0]
    file_name = pd.DataFrame({'file_name': test_name})
    pred = pd.DataFrame({'Pred': pred_y})
    result_dataframe = file_name.join(pred)
    result_dataframe.to_csv(os.path.join(save_root, f'{name}prediction_dataframe.csv'), index=False)
    return result_dataframe

def interpolate(raw_data, outlier):
    row, group, group_skip_data = [], [], []
    if len(outlier) == 1:
        row.append(outlier[0])
        group.append(row)
        row = []

    elif len(outlier) >= 1:
        for i in range(1, len(outlier)):
            if outlier[i] - outlier[i - 1] == 1:
                row.append(outlier[i - 1])
            else:
                row.append(outlier[i - 1])
                group.append(row)
                row = []

        if outlier[-1] - outlier[-2] == 1:
            row.append(outlier[i])
            group.append(row)
            row = []
        else:
            row.append(outlier[i])
            group.append(row)
            row = []

    for sub_group in group:
        row = []
        for _ in range(len(sub_group)):
            if sub_group[0] == 0:
                if sub_group[-1] == len(raw_data)-1:
                    break
                else:
                    row.append(raw_data[sub_group[-1] + 1])

            elif sub_group[0] != 0:
                if sub_group[-1] == len(raw_data)-1:
                    row.append(raw_data[sub_group[0] - 1])
                else:
                    if raw_data[sub_group[-1] + 1] == raw_data[sub_group[0] - 1]: row.append(raw_data[sub_group[-1] + 1])
                    elif raw_data[sub_group[-1] + 1] == 1 or raw_data[sub_group[0] - 1] == 1: row.append(1)
                    elif raw_data[sub_group[-1] + 1] == 0 or raw_data[sub_group[0] - 1] == 0: row.append(0)
                    elif raw_data[sub_group[-1] + 1] == 3 or raw_data[sub_group[0] - 1] == 3: row.append(3)
                    elif raw_data[sub_group[-1] + 1] == 2 or raw_data[sub_group[0] - 1] == 2: row.append(2)
        group_skip_data.append(row)
    return group_skip_data, group

def plot_sleep_stage(total_prediction, pred_ahi, save_root, data_root):
    print('Plotting Sleep Stage................')
    name = total_prediction['file_name'].to_numpy()
    ind_file_name = name[0].split('_')[0]
    figure, ax = plt.subplots(1, 2, figsize=(15, 5),  gridspec_kw={'width_ratios': [10, 1]})
    total_length = int(len(np.load(f'{data_root}/{ind_file_name}.npy', allow_pickle=True)[4])/256/30)
    ind_data = total_prediction
    series_name = np.array([int(ind_data['file_name'].to_numpy()[i].split('_')[1]) for i in range(len(ind_data))])
    skip_series = np.array([series_name[i-1]+j for i in range(1, len(series_name)) if series_name[i]-series_name[i-1]>1 for j in range(1,series_name[i]-series_name[i-1])])

    if len(skip_series)<0.1*total_length:
        series_pred = ind_data['Pred'].to_numpy()
        new_ind_data = np.zeros(total_length)
        new_ind_data[series_name] = series_pred
        new_ind_data = np.array(new_ind_data)

        if pred_ahi < 5: status = 'Health'
        elif 5 <= pred_ahi < 15: status = 'Slight'
        elif 15 <= pred_ahi < 30: status = 'Moderate'
        elif pred_ahi > 30: status = 'Severe'

        group_skip_data, group = interpolate(new_ind_data, skip_series)
        ind_skip_data = np.array([y for sublist in group_skip_data for y in sublist])
        new_ind_data[skip_series] = ind_skip_data
        stage0 = len(np.where(new_ind_data == 0)[0]) / len(new_ind_data)
        stage1 = len(np.where(new_ind_data == 1)[0]) / len(new_ind_data)
        stage2 = len(np.where(new_ind_data == 2)[0]) / len(new_ind_data)
        stage3 = len(np.where(new_ind_data == 3)[0]) / len(new_ind_data)
        if len(new_ind_data)%120 >40: xtick = [x for x in range(int(len(new_ind_data)/120)+2)]
        else: xtick = [x for x in range(int(len(new_ind_data)/120)+1)]

        ax[0].plot(np.array([i/120 for i in range(len(new_ind_data))]), new_ind_data)
        for i in range(len(group)):
            subgroup = group[i]
            ax[0].plot(np.array([i/120 for i in subgroup]), group_skip_data[i], 'r')

        ax[0].set_title('Sleep Stage of ' + ind_file_name)
        ax[0].set_yticks([0,1,2,3])
        ax[0].set_xticks(xtick)
        ax[0].set_xlabel('Sleep Time (HR)')
        ax[0].set_ylabel('Sleep Stage')
        ax[0].text(0, 2.8, 'Predict AHI = ' + str(round(pred_ahi,3)), fontsize=14)
        ax[0].text(0, 2.5, 'Apnea = ' + str(status), fontsize=14)

        ax[1].bar(['prediction'], [stage0 + stage1 + stage2 + stage3])
        ax[1].bar(['prediction'], [stage0 + stage1 + stage2])
        ax[1].bar(['prediction'], [stage0 + stage1])
        ax[1].bar(['prediction'], [stage0])

        ax[1].set_title('Distribution of Sleep Stage')
        ax[1].legend(['REM', 'Deep Sleep', 'Light Sleep', 'Wake'], bbox_to_anchor=(1.05, 1))
        figure.savefig(f'{save_root}/'f'{ind_file_name}.jpg')
        ax[0].cla()
        ax[1].cla()

    print('\n')



