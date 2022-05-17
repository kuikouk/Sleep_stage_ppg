import torch, random
import numpy as np
import pandas as pd
from queue import Queue
from threading import Thread
from sklearn import preprocessing

# Seed
seed = 22
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Dataset_LSTM():
    def __init__(self,file_list, padding_size, permute):
        self.file_list = file_list
        self.padding_size = padding_size
        self.permute = permute

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = pd.read_csv(self.file_list[idx])
        feature = data.iloc[:,2:]
        feature = np.array(feature)
        feature_mask = [~np.isnan(feature).any(axis=1)]
        sleep_stage = np.array(list(map(int,data.iloc[:,1])))
        name = np.array(data.iloc[:,0])
        file_name = np.array([int(x.split('-')[2].split('_')[0]) for x in name])
        series_name = np.array([int(x.split('-')[2].split('_')[1]) for x in name])

        feature = np.array(feature[feature_mask])
        sleep_stage = np.array(sleep_stage[feature_mask])
        file_name = np.array(file_name[feature_mask])
        series_name = np.array(series_name[feature_mask])

        sleep_stage_mask = np.intersect1d(np.where(sleep_stage!=6)[0],np.where(sleep_stage!=9)[0])
        feature = np.array(feature[sleep_stage_mask])
        sleep_stage = np.array(sleep_stage[sleep_stage_mask])
        file_name = np.array(file_name[sleep_stage_mask])
        series_name = np.array(series_name[sleep_stage_mask])

        normalize = preprocessing.StandardScaler()
        feature = normalize.fit_transform(feature)

        if self.permute ==True:
            sleep_stage[sleep_stage==1]=8
            sleep_stage[sleep_stage==2]=2
            sleep_stage[sleep_stage==3]=3
            sleep_stage[sleep_stage==4]=3
            sleep_stage[sleep_stage==5]=1
            sleep_stage[sleep_stage==8]=2
        else:
            sleep_stage[sleep_stage==2]=1
            sleep_stage[sleep_stage==3]=2
            sleep_stage[sleep_stage==4]=2
            sleep_stage[sleep_stage==5]=3

        sleep_stage = torch.tensor(sleep_stage)
        feature = torch.tensor(feature)
        file_name = torch.tensor(file_name)
        series_name = torch.tensor(series_name)

        length = len(feature)
        pad_1 = torch.nn.ConstantPad2d((0,0,0,self.padding_size-length),10000)
        pad_2 = torch.nn.ConstantPad1d((0,self.padding_size-length),-99999)

        feature = pad_1(feature)
        file_name = pad_2(file_name)
        series_name = pad_2(series_name)
        label = pad_2(sleep_stage)

        return feature, label, length, file_name, series_name

class CudaDataLoader:
    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        # The loop that will load into the queue in the background
        torch.cuda.set_device(self.device)
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def evaluation(test_data, device, model, num_classes=4, permute=True):
    feature = torch.tensor(test_data.iloc[:,1:].to_numpy())
    test_name = test_data.iloc[:,0].to_numpy()
    length = len(test_name)

    feature_mask = np.array(~np.isnan(feature).any(axis=1))
    feature = np.array(feature[feature_mask, :])

    normalize = preprocessing.StandardScaler()
    feature = normalize.fit_transform(feature)
    feature = torch.tensor(feature).float().to(device)
    feature = feature.reshape(1, length, 126).float().to(device)

    logits = model(feature, [length])
    logits = logits.reshape(-1, num_classes)

    Output = logits.detach().cpu()
    S = torch.nn.Softmax(dim=1)
    Prob = S(Output).numpy()

    if permute==True:
        Prob = Prob[:,[0,2,3,1]]

    Pred = np.array([list(x).index(max(x)) for x in Prob])

    return Prob, Pred, test_name