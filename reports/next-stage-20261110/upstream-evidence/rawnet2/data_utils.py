import torch
import collections
import os
import soundfile as sf
from torch.utils.data import Dataset
import numpy as np
from joblib import Parallel, delayed
from torch import Tensor


ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

## spoof=0,bonafie=1
Label_Dic={'fake':0, 'real':1}

def pad(x, max_len=400): #64600
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x	
    
class ASVDataset(Dataset):
    """ Utility class to load  train/dev/Eval datatsets """
    def __init__(self, datafile=None, transform=None, 
        sample_size=None, feature_name=None):  

        assert feature_name is not None, 'must provide feature name'
        self.sysid_dict = {
        '-': 0,  # bonafide speech
        'A01': 1, 
        'A02': 2, 
        'A03': 3, 
        'A04': 4, 
        'A05': 5, 
        'A06': 6,
        'A07': 7, 
        'A08': 8, 
        'A09': 9, 
        'A10': 10, 
        'A11': 11, 
        'A12': 12,
        'A13': 13, 
        'A14': 14, 
        'A15': 15, 
        'A16': 16, 
        'A17': 17, 
        'A18': 18,
        'A19': 19, 
        }

        self.datafile=datafile
        self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
        print('sysid_dict_inv',self.sysid_dict_inv)

        self.transform = transform
      
        self.files_meta = self.parse_protocols_file(self.datafile)
        data = list(map(self.read_file, self.files_meta))
        self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
        
        if self.transform:
            self.data_x = Parallel(n_jobs=4, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)
            
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
            self.files_meta= [self.files_meta[x] for x in select_idx]
            self.data_x = [self.data_x[x] for x in select_idx]
            self.data_y = [self.data_y[x] for x in select_idx]
            self.data_sysid = [self.data_sysid[x] for x in select_idx]
            
        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        data_x, sample_rate = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))
        

   
    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)


class Dataset_commom_np(Dataset):
    def __init__(self, data_file):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.data_file = data_file
            self.__process_datafile__(data_file)
            self.cut = 400    #64600 # take ~4 sec audio (64600 samples)
            
    def __process_datafile__(self, data_file):
        self.data = []
        self.labels = []
        with open(data_file,'r') as rf:
            for line in rf.readlines():
                id, path, key = line.strip().split('\t')
                self.data.append(path)
                self.labels.append(Label_Dic[key])
         
    def __len__(self):
        return len(self.data)
     
    def __getitem__(self, index):
        utt_path = self.data[index]
        x = np.load(utt_path) 
        X_pad= pad(x,self.cut)
        X_pad_3D=X_pad[np.newaxis,:,:]
        x_inp= Tensor(X_pad_3D)
        target = self.labels[index]
        return x_inp, target

class Dataset_commom_raw(Dataset):
    def __init__(self, data_file):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.data_file = data_file
            self.__process_datafile__(data_file)
            self.cut = 64600    #64600 # take ~4 sec audio (64600 samples)
            
    def __process_datafile__(self, data_file):
        self.data = []
        self.labels = []
        with open(data_file,'r') as rf:
            for line in rf.readlines():
                id, path, key = line.strip().split('\t')
                self.data.append(path)
                self.labels.append(Label_Dic[key])
         
    def __len__(self):
        return len(self.data)
     
    def __getitem__(self, index):
        utt_path = self.data[index]
        x , sample_rate = sf.read(utt_path)
        # print('------x.shape------')
        # print(x.shape)
        x_pad= pad(x,self.cut)
        # print('------x_pad.shape------')
        # print(x_pad.shape)
        # X_pad_3D=X_pad[np.newaxis,:,:]
        # x_inp= Tensor(X_pad_3D)
        target = self.labels[index]
        return x_pad, target

class MIXDataset(Dataset):
    """ Utility class to load  train/dev/Eval datatsets """
    def __init__(self, data_file=None,transform=None):        
        self.transform = transform

        self.data, self.labels= self.process_datafile(data_file)
        if self.transform:
            self.data_x = Parallel(n_jobs=4, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file=self.data[idx]
        x=self.read_file(file)
        x_pad= self.pad(x)
        y = float(self.labels[idx])
        return x_pad , y
    
    def process_datafile(self, data_file):
        data = []
        labels = []
        with open(data_file,'r') as rf:
            for line in rf.readlines():
                id, path, key = line.strip().split('\t')
                self.data.append(path)
                self.labels.append(Label_Dic[key])
        return data, labels

    def read_file(self, path):  
        try:      
            data_x, sample_rate = sf.read(path)
            return data_x
        except:
            print('ERROR!')
            print(path)


class Dataset_one_utt(Dataset): 
    def __init__(self, list_IDs):
            '''self.list_IDs    : list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
            return len(self.list_IDs)


    def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            x, fs = sf.read(utt_id)
            x_pad= self.pad(x,self.cut)
            return x_pad
    
    def pad(self, x, max_len=64600): #64600 96000
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x	