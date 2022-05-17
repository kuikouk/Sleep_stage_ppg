from scipy.stats import entropy
from collections import Counter
from scipy.signal import welch
from numpy import log as ln
from numpy import diff
from pyhrv import hrv
import pandas as pd
import numpy as np
import pywt, warnings

warnings.filterwarnings("ignore")

columns =[
    'name','mean_RR', 'median_RR', 'mean_HR', 'median_HR', 'P5', 'P10', 'P25', 'P50', 'P75',
    'P90', 'P95','AVNN', 'MAD', 'RR_mod', 'RRdif_mod', 'RMSSD', 'SDNN', 'SD1', 'SD2', 'SD1_SD2_ratio', 'S', 'CV_RR',
    'NN50', 'pNN50', 'SDSD', 'VLF', 'TLF', 'MF', 'LF', 'HF', 'TF', 'LF_HF_ratio', 'MF_LF_ratio',
    'TLF_LF_ratio', 'HFmaxf', 'HFamp', 'LFn', 'HFn', 'MFn', 'TLFn', 'mean_LFf', 'mean_TLFf', 'mean_MFf',
    'mean_HFf', 'mean_TFf', 'SE_LF', 'SE_TLF', 'SE_MF', 'SE_HF', 'SE_TF', 'mZCIa', 'nsZCIa', 'mZCIl', 'nsZCIl',
    'mZCIh', 'nsZCIh', 'RR_min', 'RR_max', 'RR_diff_mean', 'RR_diff_min', 'RR_diff_max',
    'hr_min', 'hr_max', 'hr_std', 'nn20', 'pnn20', 'tinn_n', 'tinn_m', 'tinn', 'tri_index', 'tff_vlf_peak', 'tff_lf_peak',
    'tff_hf_peak', 'tff_vlf_abs', 'tff_lf_abs', 'tff_hf_abs', 'tff_vlf_rel', 'tff_lf_rel', 'tff_hlf_rel', 'tff_vlf_log',
    'tff_lf_log', 'tff_hlf_log', 'fft_vlf_norm', 'fft_hlf_norm', 'fft_ratio', 'fft_total', 'lomb_vlf_peak', 'lomb_lf_peak',
    'lomb_hf_peak', 'lomb_vlf_abs', 'lomb_lf_abs', 'lomb_hf_abs', 'lomb_vlf_rel', 'lomb_lf_rel', 'lomb_hlf_rel', 'lomb_vlf_log',
    'lomb_lf_log', 'lomb_hlf_log', 'lomb_vlf_norm', 'lomb_hlf_norm', 'lomb_ratio', 'lomb_total', 'ar_vlf_peak', 'ar_lf_peak',
    'ar_hf_peak', 'ar_vlf_abs', 'ar_lf_abs', 'ar_hf_abs', 'ar_vlf_rel', 'ar_lf_rel', 'ar_hlf_rel', 'ar_vlf_log', 'ar_lf_log',
    'ar_hlf_log', 'ar_vlf_norm', 'ar_hlf_norm', 'ar_ratio', 'ar_total', 'sd1', 'sd2', 'sd_ratio', 'ellipse_area', 'sample_entropy',
    'Shannon_entropy', 'dfa_short', 'dfa_long']

def Feature_extraction(name, IBI):
    SDNN, RMSSD, Ft, MAD, n = 0, 0, 0, 0, 0
    mean_RR = np.mean(IBI)
    median_RR = np.median(IBI)
    mean_HR = mean_RR * 60 / 125
    median_HR = median_RR * 60 / 125

    P5 = np.percentile(IBI, 5)
    P10 = np.percentile(IBI, 10)
    P25 = np.percentile(IBI, 25)
    P50 = np.percentile(IBI, 50)
    P75 = np.percentile(IBI, 75)
    P90 = np.percentile(IBI, 90)
    P95 = np.percentile(IBI, 95)

    AVNN = np.sum(IBI) / len(IBI)
    RRdif_mod = Counter([round(x) for x in diff(IBI)]).most_common()[0][0]
    RR_mod = Counter([round(x) for x in IBI]).most_common()[0][0]

    for x in IBI:
        SDNN += (x - mean_RR) ** 2
        MAD += abs(x - mean_RR)
    for j in range(len(IBI) - 1):
        RMSSD += (IBI[j + 1] - IBI[j]) ** 2
    MAD = MAD/ len(IBI)
    RMSSD = (RMSSD / len(IBI)) ** (1 / 2)
    SDNN = (SDNN / (len(IBI) - 1)) ** (1 / 2)
    CV_RR = SDNN / mean_RR
    NN50 = len([x for x in IBI if x > 50])
    pNN50 = NN50 / len(IBI)
    SDSD = diff(IBI).std()
    SD1 = (SDSD ** 2) / 2
    SD2 = 2 * (SDNN) ** 2 - SD1
    SD1_SD2_ratio = SD1 / SD2
    S = 3.1415926 * SD1 * SD2

    pd_series = pd.Series(IBI)
    counts = pd_series.value_counts()
    Shannon_entropy = entropy(counts)

    nni_results = hrv(nni=IBI, plot_ecg=False, plot_tachogram=False, show=False)
    RR_min = nni_results['nni_min']
    RR_max = nni_results['nni_max']
    RR_diff_mean = nni_results['nni_diff_mean']
    RR_diff_min = nni_results['nni_diff_min']
    RR_diff_max = nni_results['nni_diff_max']
    hr_min = nni_results['hr_min']
    hr_max = nni_results['hr_max']
    hr_std = nni_results['hr_std']
    nn20 = nni_results['nn20']
    pnn20 = nni_results['pnn20']
    tinn_n = nni_results['tinn_n']
    tinn_m = nni_results['tinn_m']
    tinn = nni_results['tinn']
    tri_index = nni_results['tri_index']

    tff_vlf_peak = nni_results['fft_peak'][0]
    tff_lf_peak = nni_results['fft_peak'][1]
    tff_hf_peak = nni_results['fft_peak'][2]
    tff_vlf_abs = nni_results['fft_abs'][0]
    tff_lf_abs = nni_results['fft_abs'][1]
    tff_hf_abs = nni_results['fft_abs'][2]
    tff_vlf_rel = nni_results['fft_rel'][0]
    tff_lf_rel = nni_results['fft_rel'][1]
    tff_hlf_rel = nni_results['fft_rel'][2]
    tff_vlf_log = nni_results['fft_log'][0]
    tff_lf_log = nni_results['fft_log'][1]
    tff_hlf_log = nni_results['fft_log'][2]
    fft_vlf_norm = nni_results['fft_norm'][0]
    fft_hlf_norm = nni_results['fft_norm'][1]
    fft_ratio = nni_results['fft_ratio']
    fft_total = nni_results['fft_total']

    lomb_vlf_peak = nni_results['lomb_peak'][0]
    lomb_lf_peak = nni_results['lomb_peak'][1]
    lomb_hf_peak = nni_results['lomb_peak'][2]
    lomb_vlf_abs = nni_results['lomb_abs'][0]
    lomb_lf_abs = nni_results['lomb_abs'][1]
    lomb_hf_abs = nni_results['lomb_abs'][2]
    lomb_vlf_rel = nni_results['lomb_rel'][0]
    lomb_lf_rel = nni_results['lomb_rel'][1]
    lomb_hlf_rel = nni_results['lomb_rel'][2]
    lomb_vlf_log = nni_results['lomb_log'][0]
    lomb_lf_log = nni_results['lomb_log'][1]
    lomb_hlf_log = nni_results['lomb_log'][2]
    lomb_vlf_norm = nni_results['lomb_norm'][0]
    lomb_hlf_norm = nni_results['lomb_norm'][1]
    lomb_ratio = nni_results['lomb_ratio']
    lomb_total = nni_results['lomb_total']

    ar_vlf_peak = nni_results['ar_peak'][0]
    ar_lf_peak = nni_results['ar_peak'][1]
    ar_hf_peak = nni_results['ar_abs'][2]
    ar_vlf_abs = nni_results['ar_abs'][0]
    ar_lf_abs = nni_results['ar_abs'][1]
    ar_hf_abs = nni_results['ar_abs'][2]
    ar_vlf_rel = nni_results['ar_rel'][0]
    ar_lf_rel = nni_results['ar_rel'][1]
    ar_hlf_rel = nni_results['ar_rel'][2]
    ar_vlf_log = nni_results['ar_log'][0]
    ar_lf_log = nni_results['ar_log'][1]
    ar_hlf_log = nni_results['ar_log'][2]
    ar_vlf_norm = nni_results['ar_norm'][0]
    ar_hlf_norm = nni_results['ar_norm'][1]
    ar_ratio = nni_results['ar_ratio']
    ar_total = nni_results['ar_total']

    sd1 = nni_results['sd1']
    sd2 = nni_results['sd2']
    sd_ratio = nni_results['sd_ratio']
    ellipse_area = nni_results['ellipse_area']
    sample_entropy = nni_results['sampen']
    dfa_short = nni_results['dfa_alpha1']
    dfa_long = nni_results['dfa_alpha2']

    LF,HF = [], []
    df = [x - mean_RR for x in IBI]
    zero_cross_point = np.where(np.diff(np.sign(df)))[0]
    ZCI = np.array([zero_cross_point[i]-zero_cross_point[i-1] for i in range(1,len(zero_cross_point))])
    mZCIa = ZCI.mean()
    nsZCIa = ZCI.std()
    db5,_ = pywt.dwt(IBI,'db5')
    db6, _ = pywt.dwt(IBI, 'db6')
    db3, _ = pywt.dwt(IBI, 'db3')
    db4, _ = pywt.dwt(IBI, 'db4')
    LF.append(db5), LF.append(db6)
    HF.append(db3), HF.append(db4)
    LF = [x for sublist in LF for x in sublist]
    HF = [x for sublist in HF for x in sublist]

    mean_LF = np.array(LF).mean()
    df = [x - mean_LF for x in LF]
    zero_cross_point = np.where(np.diff(np.sign(df)))[0]
    ZCI = np.array([zero_cross_point[i]-zero_cross_point[i-1] for i in range(1,len(zero_cross_point))])
    mZCIl = ZCI.mean()
    nsZCIl = ZCI.std()

    mean_HF = np.array(HF).mean()
    df = [x - mean_HF for x in HF]
    zero_cross_point = np.where(np.diff(np.sign(df)))[0]
    ZCI = np.array([zero_cross_point[i]-zero_cross_point[i-1] for i in range(1,len(zero_cross_point))])
    mZCIh = ZCI.mean()
    nsZCIh = ZCI.std()

    Fxx, Pxx = welch(IBI, fs=2.0)
    vlf_freq_band = (Fxx >= 0) & (Fxx <= 0.04)
    lf_freq_band = (Fxx >= 0.04) & (Fxx <= 0.15)
    tlf_freq_band = (Fxx >= 0.04) & (Fxx <= 0.1)
    mf_freq_band = (Fxx >= 0.1) & (Fxx <= 0.15)
    hf_freq_band = (Fxx >= 0.15) & (Fxx <= 0.4)
    tf_freq_band = (Fxx >= 0.04) & (Fxx <= 0.4)

    VLF = np.trapz(y=abs(Pxx[vlf_freq_band]))
    LF = np.trapz(y=abs(Pxx[lf_freq_band]))
    TLF = np.trapz(y=abs(Pxx[tlf_freq_band]))
    MF = np.trapz(y=abs(Pxx[mf_freq_band]))
    HF = np.trapz(y=abs(Pxx[hf_freq_band]))
    TF = np.trapz(y=abs(Pxx[tf_freq_band]))

    mean_LFf, mean_TLFf, mean_MFf, mean_HFf, mean_TFf = 0, 0, 0, 0, 0
    SE_LF, SE_TLF, SE_MF, SE_HF, SE_TF = 0, 0, 0, 0, 0
    table = []
    n_LF = len(np.where(lf_freq_band==True)[0])
    n_TLF = len(np.where(tlf_freq_band==True)[0])
    n_MF = len(np.where(mf_freq_band==True)[0])
    n_HF = len(np.where(hf_freq_band==True)[0])

    for n in range(n_LF):
        LFf = Fxx[lf_freq_band][n]
        LFp = Pxx[Fxx == LFf]
        mean_LFf += LFp * LFf
        SE_LF += (LFp * ln(LFp)) / ln(n_LF)

    for n in range(n_TLF):
        TLFf = Fxx[tlf_freq_band][n]
        TLFp = Pxx[Fxx == TLFf]
        mean_TLFf += TLFp * TLFf
        SE_TLF += (TLFp * ln(TLFp)) / ln(n_TLF)

    for n in range(n_MF):
        MFf = Fxx[mf_freq_band][n]
        MFp = Pxx[Fxx == MFf]
        mean_MFf += MFp * MFf
        SE_MF += (MFp * ln(MFp)) / ln(n_MF)

    for n in range(n_HF):
        row = []
        HFf = Fxx[hf_freq_band][n]
        HFp = Pxx[Fxx == HFf]
        mean_HFf += HFp * HFf
        SE_HF += (HFp * ln(HFp)) / ln(n_HF)
        row.append(HFp)
        row.append(HFf)
        table.append(row)

    for n in range(n_TLF):
        TFf = Fxx[tlf_freq_band][n]
        TFp = Pxx[Fxx == TFf]
        mean_TFf += TFp * TFf
        SE_TF += (TFp * ln(TFp)) / ln(n_TLF)

    mean_LFf = float(mean_LFf / LF)
    mean_TLFf = float(mean_TLFf / TLF)
    mean_MFf = float(mean_MFf / MF)
    mean_HFf = float(mean_HFf / HF)
    mean_TFf = float(mean_TFf / TF)

    SE_LF = float(-1 * SE_LF)
    SE_TLF = float(-1 * SE_TLF)
    SE_MF = float(-1 * SE_MF)
    SE_HF = float(-1 * SE_HF)
    SE_TF = float(-1 * SE_TF)

    table = np.array(table)
    HFmaxf = float(table[:, 1][np.where(table[:, 0] == np.max(table[:, 0]))[0]])
    HFamp = float(np.max(table[:, 0])/HF)

    LF_HF_ratio = LF / HF
    MF_LF_ratio = MF / LF
    TLF_LF_ratio = TLF / LF

    LFn = LF * 100 / (TF - VLF)
    HFn = HF * 100 / (TF - VLF)
    MFn = MF * 100 / (TF - VLF)
    TLFn = TLF * 100 / (TF - VLF)

    df = [name, mean_RR, median_RR, mean_HR, median_HR, P5, P10, P25, P50, P75, P90, P95, AVNN, MAD,
          RR_mod, RRdif_mod, RMSSD, SDNN, SD1, SD2, SD1_SD2_ratio, S, CV_RR, NN50, pNN50, SDSD, VLF,
          TLF, MF, LF, HF, TF, LF_HF_ratio, MF_LF_ratio, TLF_LF_ratio, HFmaxf, HFamp, LFn, HFn, MFn, TLFn,
          mean_LFf, mean_TLFf, mean_MFf, mean_HFf, mean_TFf, SE_LF, SE_TLF, SE_MF, SE_HF, SE_TF, mZCIa, nsZCIa,
          mZCIl, nsZCIl, mZCIh, nsZCIh, RR_min, RR_max, RR_diff_mean, RR_diff_min, RR_diff_max,
          hr_min, hr_max, hr_std, nn20, pnn20, tinn_n, tinn_m, tinn, tri_index, tff_vlf_peak, tff_lf_peak, tff_hf_peak,
          tff_vlf_abs, tff_lf_abs, tff_hf_abs, tff_vlf_rel, tff_lf_rel, tff_hlf_rel, tff_vlf_log, tff_lf_log, tff_hlf_log,
          fft_vlf_norm, fft_hlf_norm, fft_ratio, fft_total, lomb_vlf_peak, lomb_lf_peak, lomb_hf_peak, lomb_vlf_abs,
          lomb_lf_abs, lomb_hf_abs, lomb_vlf_rel, lomb_lf_rel, lomb_hlf_rel, lomb_vlf_log, lomb_lf_log, lomb_hlf_log,
          lomb_vlf_norm, lomb_hlf_norm, lomb_ratio, lomb_total, ar_vlf_peak, ar_lf_peak, ar_hf_peak, ar_vlf_abs,
          ar_lf_abs, ar_hf_abs, ar_vlf_rel, ar_lf_rel, ar_hlf_rel, ar_vlf_log, ar_lf_log, ar_hlf_log, ar_vlf_norm,
          ar_hlf_norm, ar_ratio, ar_total, sd1, sd2, sd_ratio, ellipse_area, sample_entropy, Shannon_entropy, dfa_short, dfa_long]
    return df, columns

