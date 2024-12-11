from torch.utils.data import Dataset
import numpy as np
import torch
import os
import scipy.io as sio

def preprocess(displ_files,flag):

    row_delete = 40
    col_delete = 15
    gt_frame_range = np.arange(1,7)

    if flag == 'OPT':        
        files_processed = displ_files.astype(np.float32)
        files_processed[0], files_processed[1] = files_processed[1], files_processed[0]
        files_processed = files_processed[:, row_delete:-row_delete, col_delete:-col_delete]
    elif flag == 'GT':
        # 4d: a/l, strain size, y, x        
        files_processed = displ_files.astype(np.float32)
        # files_processed = files_processed[:,gt_frame_range, row_delete:-row_delete, 
        #                                   col_delete:-col_delete]
    elif flag == 'TRAD':
        files_processed = displ_files.astype(np.float32)
        

    return files_processed

class NpyDataset(Dataset):
    def __init__(self, data_folder_t,data_folder_o,data_folder_gt,flag='train'):
        self.data_folder_t = data_folder_t
        self.data_folder_o = data_folder_o
        self.data_folder_gt = data_folder_gt
        
        if flag == 'train':
            target_fields = ['model2','frame8','frame9','frame10']
        elif flag == 'test':
            target_fields = ['frame8','frame9','frame10']

        self.files_t = [f for f in os.listdir(data_folder_t) if f.endswith('.mat') 
                        and not any(field in f for field in target_fields)]
        self.files_o = [f for f in os.listdir(data_folder_o) if f.endswith('.npy')
                        and not any(field in f for field in target_fields)]
        self.files_gt = [f for f in os.listdir(data_folder_gt) if f.endswith('.npy')
                         and not any(field in f for field in target_fields)]

    def __len__(self):
        return len(self.files_t)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_folder_t, self.files_t[index])
        data_trad = sio.loadmat(file_path)['displ']
        trad = preprocess(data_trad,'TRAD')
        a_offset = np.max(np.abs(trad[0,:,:]))
        l_offset = np.max(np.abs(trad[1,:,:]))

        file_path = os.path.join(self.data_folder_o, self.files_o[index])
        data_opt = np.load(file_path)
        opt = preprocess(data_opt,'OPT')
        if np.max(np.abs(opt[0,:,:])) > a_offset:
            a_offset =np.max(np.abs(opt[0,:,:]))
        if np.max(np.abs(opt[1,:,:])) > l_offset:
            l_offset = np.max(np.abs(opt[1,:,:]))

        file_path = os.path.join(self.data_folder_gt, self.files_gt[index])
        data_gt = np.load(file_path)
        gt = preprocess(data_gt,'GT')

        offset = np.array([a_offset, l_offset])
        trad += offset[:, None, None]
        opt += offset[:, None, None]
        gt += offset[:, None, None]
        trad /= offset[:, None, None]
        opt /= offset[:, None, None]
        gt /= offset[:, None, None]

        return torch.tensor(trad, dtype=torch.float32),torch.tensor(opt, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32)

