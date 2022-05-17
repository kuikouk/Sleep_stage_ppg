import numpy as np
import torch

def feature_measurement(new_data):
    DesDuri3, DesArea3, DesSev3, DesDur3, start3, end3 = [], [], [], [], 0, 0
    DesDuri4, DesArea4, DesSev4, DesDur4, start4, end4 = [], [], [], [], 0, 0
    DesDuri90, DesArea90, DesSev90, DesDur90, start90, end90 = [], [], [], [], 0, 0
    Status3, Status4, Status90 = [], [], []

    baseline = np.mean(new_data)
    for i in range(1, len(new_data)):
        if new_data[i - 1] > baseline - 3 and new_data[i] < baseline - 3:
            start3 = i
        if new_data[i - 1] < baseline - 3 and new_data[i] > baseline - 3:
            end3 = i
        if start3 != 0 and end3 != 0:
            Area3 = 0
            duration3 = end3 - start3
            if duration3 > 3:
                DesDuri3.append(duration3)
                for j in range(end3 - start3): Area3 += new_data[j]
                DesArea3.append(Area3)
                DesSev3.append(Area3 / len(new_data))
                DesDur3.append(duration3 / len(new_data))
            start3, end3 = 0, 0

        if new_data[i - 1] > baseline - 4 and new_data[i] < baseline - 4:
            start4 = i
        if new_data[i - 1] < baseline - 4 and new_data[i] > baseline - 4:
            end4 = i
        if start4 != 0 and end4 != 0:
            Area4 = 0
            duration4 = end4 - start4
            if duration4 > 3:
                DesDuri4.append(duration4)
                for j in range(end4 - start4): Area4 += new_data[j]
                DesArea4.append(Area4)
                DesSev4.append(Area4 / len(new_data))
                DesDur4.append(duration4 / len(new_data))
            start4, end4 = 0, 0

        if new_data[i - 1] > 90 and new_data[i] < 90:
            start90 = i
        if new_data[i - 1] < 90 and new_data[i] > 90:
            end90 = i
        if start90 != 0 and end90 != 0:
            Area90 = 0
            duration90 = end90 - start90
            if duration90 > 3:
                DesDuri90.append(duration90)
                for j in range(end90 - start90): Area90 += new_data[j]
                DesArea90.append(Area90)
                DesSev90.append(Area90 / len(new_data))
                DesDur90.append(duration90 / len(new_data))
            start90, end90 = 0, 0

    odi3 = 3600 * len(DesDuri3) / len(new_data)
    odi4 = 3600 * len(DesDuri4) / len(new_data)
    odi90 = 3600 * len(DesDuri90) / len(new_data)

    DesDuri3 = np.sum(DesDuri3)
    DesArea3 = np.sum(DesArea3)
    DesSev3 = np.sum(DesSev3)
    DesDur3 = np.sum(DesDur3)
    Avg_DesArea3 = np.mean(DesArea3)
    Avg_DesDur3 = np.mean(DesDuri3)

    DesDuri4 = np.sum(DesDuri4)
    DesArea4 = np.sum(DesArea4)
    DesSev4 = np.sum(DesSev4)
    DesDur4 = np.sum(DesDur4)
    Avg_DesArea4 = np.mean(DesArea4)
    Avg_DesDur4 = np.mean(DesDuri4)

    DesDuri90 = np.sum(DesDuri90)
    DesArea90 = np.sum(DesArea90)
    DesSev90 = np.sum(DesSev90)
    DesDur90 = np.sum(DesDur90)
    Avg_DesArea90 = np.mean(DesArea90)
    Avg_DesDur90 = np.mean(DesDuri90)

    Status3.append(odi3)
    Status3.append(DesDuri3)
    Status3.append(DesArea3)
    Status3.append(DesSev3)
    Status3.append(DesDur3)
    Status3.append(Avg_DesArea3)
    Status3.append(Avg_DesDur3)

    Status4.append(odi4)
    Status4.append(DesDuri4)
    Status4.append(DesArea4)
    Status4.append(DesSev4)
    Status4.append(DesDur4)
    Status4.append(Avg_DesArea4)
    Status4.append(Avg_DesDur4)

    Status90.append(odi90)
    Status90.append(DesDuri90)
    Status90.append(DesArea90)
    Status90.append(DesSev90)
    Status90.append(DesDur90)
    Status90.append(Avg_DesArea90)
    Status90.append(Avg_DesDur90)

    return Status3, Status4, Status90

def SpO2_preprocess(raw_data):
    new_data = raw_data[np.where(raw_data > 65)[0]]
    Status3, Status4, Status90 = feature_measurement(new_data)

    feature = np.array(Status3)
    feature = np.append(feature, Status4)
    feature = np.append(feature, Status90)
    feature = torch.from_numpy(feature.astype(np.float32))
    return feature