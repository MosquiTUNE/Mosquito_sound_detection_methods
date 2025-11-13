import pandas as pd
import librosa
import numpy as np
import librosa.display
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


# normalize the spectrogram
def spec_normalization(spec, err=1e-6):
    mean, std = spec.mean(), spec.std()
    spec = (spec - mean) / (std + err)
    return spec


# transfer from 2 channels spectrogram to 3 channels image
def spec_img(spec):
    spec = spec_normalization(spec)
    spec_min, spec_max = spec.min(), spec.max()
    spec = 255 * (spec - spec_min) / (spec_max - spec_min)
    spec = spec.astype(np.uint8)
    spec = spec[np.newaxis, ...]
    #print(spec.shape)
    return spec


# data augmentation on time domain and frequency domain in spectrogram image - have 3 channels
def specaug(mel_spectrogram, frequency_masking_para=1,
            time_masking_para=1, frequency_mask_num=1, time_mask_num=1):
    """
        Modified from SpecAugment
        Author: Demis TaeKyu Eom and Evangelos Kazakos
        License: https://github.com/DemisEom/SpecAugment/blob/master/LICENSE
        Code URL: https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
    """
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]
    # Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        mel_spectrogram[:, f0:f0 + f, :] = 0

    # Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        mel_spectrogram[:, :, t0:t0 + t] = 0
    return mel_spectrogram


# Store data into Dataset
class ListDataset(Dataset):
    def __init__(self, label_file, label_list, d_type=None):
        self.label_file = pd.read_csv(label_file)
        self.label_list = label_list
        self.transform = transforms.ToTensor()
        self.d_type = d_type
    
        self.specs = []
        self.labels = []
        self.fns = []

        for i in range(len(self.label_file)):
            audio_path = self.label_file.iloc[i]['Fname']
            self.fns.append(audio_path)
           
            spec = get_melspec(audio_path)
            spec = spec_img(spec)

            self.specs.append(spec)

            # get its label
            label_class = self.label_file.iloc[i]['Species']
            #print(label_class)
            label = torch.tensor(self.label_list[label_class])
            self.labels.append(label)

    def __getitem__(self, index):
        # data augumentation
        cur_spec = self.specs[index]
        if self.d_type == "train":
            cur_spec = specaug(cur_spec)

        return cur_spec, self.labels[index], self.fns[index]

    def __len__(self):
        #         length of the whole dataset
        return len(self.labels)

    
# transform data from raw audio to spectrogram
def get_melspec(file_path, sr=8000, top_db=80):
    #print(file_path)
    wav, sr = librosa.load(file_path, sr=sr)
    
    wav = np.pad(wav, int(np.ceil((2 * sr - wav.shape[0]) / 2)), mode='reflect')
 
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=256, hop_length=64)
   
    spec = librosa.power_to_db(spec, top_db=top_db)
    return spec




