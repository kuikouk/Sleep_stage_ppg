from util import result_prediction, plot_sleep_stage, logger
from ppg_signal_preprocess import PPG_preprocessing
from model_process import evaluation
from SpO2_process import SpO2_preprocess
from time import process_time
import numpy as np
import warnings, os, torch
import lightgbm

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# =============================== parameter ===============================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
root = './Demo/'

sampling_rate = 256

save_root = os.path.join(root, 'result/')
if not os.path.exists(save_root): os.mkdir(save_root)

# =============================== data processing ==========================================
file_list = os.listdir(root)
file_list = [file_list[i] for i in range(len(file_list)) if '.npy' in file_list[i]]
for file_name in file_list:
    name = file_name.split('.')[0]
    raw_data = np.load(os.path.join(root,file_name), allow_pickle=True)
    raw_ppg = raw_data[4]
    raw_spo2 = raw_data[5]

    print('HRV measurement ...............')
    HRV_result = PPG_preprocessing(raw_ppg, file_name, sampling_rate)

# =============================== model setting ===========================================
    model = torch.load(f'{root}/model.pt', map_location=device)
    model.device = device
    log = logger()

# =============================== evaluate model =============================================
    print('Model prediction ...............')
    start = process_time()
    model.eval()
    prob, pred, test_name = evaluation(HRV_result, device, model)
    end = process_time()
    print('Process time is 'f'{round((end - start), 2)} seconds')

    model = lightgbm.Booster(model_file='./Demo/AHI_model_regression.txt')
    spo2_feature = SpO2_preprocess(raw_spo2)
    pred_ahi = model.predict(spo2_feature, num_iteration=model.best_iteration).item()

# ============= plot and save prediction ================
    total_prediction = result_prediction(test_name, pred, save_root)
    plot_sleep_stage(total_prediction, pred_ahi, save_root, root)
