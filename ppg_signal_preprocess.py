from HRV_feature_measurement import Feature_extraction
from scipy.interpolate import interp1d
from peakdet import operations
from tqdm import tqdm
import numpy as np
import pandas as pd
import heartpy as hp
import heartpy

def preprocess(raw_ppg, sample_rate):
    filtered_ppg = heartpy.preprocessing.interpolate_clipping(raw_ppg, sample_rate=sample_rate, threshold=np.mean(raw_ppg)+5*np.std(raw_ppg))
    filtered_ppg = hp.filter_signal(filtered_ppg, cutoff=[0.5,8], sample_rate=sample_rate, order=2, filtertype='bandpass')
    filtered_ppg = hp.smooth_signal(filtered_ppg, sample_rate=256)
    PPG_peak = operations.peakfind_physio(filtered_ppg, thresh=0.03, dist=sample_rate*60/140).peaks

    RR_list = np.array([PPG_peak[i] - PPG_peak[i - 1] for i in range(1, len(PPG_peak))])
    RR_position = np.array([(PPG_peak[i] + PPG_peak[i - 1]) / 2 for i in range(1, len(PPG_peak))])
    return RR_list, RR_position, PPG_peak

def exlude_outlier(raw_data, outlier):
    row, group = [], []

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
        if sub_group[0] == 0:
            if sub_group[-1] != len(raw_data)-1:
                raw_data[sub_group] = (raw_data[sub_group[-1]+1])
            else:
                raw_data[sub_group] = np.mean(raw_data)
        if sub_group[-1] == len(raw_data)-1:
            if sub_group[0]-1 >=0:
                raw_data[sub_group] = (raw_data[sub_group[0]-1])
            else:
                raw_data[sub_group] = np.mean(raw_data)
        else:
            raw_data[sub_group] = (raw_data[sub_group[0]-1] + raw_data[sub_group[-1]+1]) / 2

    return raw_data

def signal_processing(peak, IBI, i, sampling_rate):
    peak_interval = np.array([x for x in np.intersect1d(np.where(peak > sampling_rate*30 * (i - 4)), np.where(peak < sampling_rate*30 * (i + 5)))])
    ind_IBI = IBI[peak_interval]
    ind_peak = peak[peak_interval] - ((i - 4) * sampling_rate * 30)
    index = np.intersect1d(np.where(ind_IBI<512)[0], np.where(ind_IBI>170)[0])
    ind_IBI = ind_IBI[index]
    ind_peak = ind_peak[index]

    if len(peak_interval) > 1 and ind_peak[-1] > sampling_rate * 30 * 8.5 and ind_peak[0] < sampling_rate * 30 * 0.5 and 480 > len(ind_peak) > 200:
        outlier = np.union1d(np.where(ind_IBI < np.mean(ind_IBI) - 32)[0], np.where(ind_IBI > np.mean(ind_IBI) + 32)[0])
        ind_IBI = exlude_outlier(ind_IBI, outlier)

        if len(outlier) < 0.2*len(ind_IBI):
            fe = interp1d(ind_peak, ind_IBI)
            interval = [(sampling_rate*8*30) * x / 480 + (sampling_rate*30*0.5) for x in range(480)]
            smo_IBI = np.array(fe(interval))

    return smo_IBI

def PPG_preprocessing(raw_ppg, file_name, sampling_rate):
    RR_list, RR_position,_ = preprocess(raw_ppg,sampling_rate)
    result = []
    total_length = int(len(raw_ppg)/sampling_rate/30)+1
    with tqdm(total=total_length-9) as pbar:
        for i in range(4, total_length - 5):
            try:
                smo_IBI = signal_processing(RR_position, RR_list, i, sampling_rate)
                df, columns = Feature_extraction(file_name.split('.')[0] + '_' + str(i), smo_IBI)
                result.append(df)
                pbar.update(1)
            except:
                pbar.update(1)
                continue

        if len(result) > 1:
            HRV_result = pd.DataFrame(result, columns=columns)
            return HRV_result
