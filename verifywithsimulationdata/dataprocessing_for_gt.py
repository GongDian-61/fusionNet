import os
import numpy as np
import scipy.io as sio

# preprocess for gt
row_delete = 40
col_delete = 15
gt_frame_range = np.arange(1,7)
# folder_path = r"/media/eadu/DATA/sim_groudtruth"
# save_folder = r"/media/eadu/DATA/sim_groudtruth/gt_in_slices_repeat"

save_folder = r"/media/eadu/DATA/displacment by NN/DeepUSE-main/sim_data"
folder_path = r"/media/eadu/DATA/displacment by NN/DeepUSE-main/sim_data"
files_list = os.listdir(folder_path)
i2=2

for file in files_list:
    if file.endswith('.mat'):
        file_path = os.path.join(folder_path, file)        
        data_gt = sio.loadmat(file_path)['RF']
        files_processed = data_gt.astype(np.float32)
        for scatter in range(10):            
            
            frames_rf = files_processed[scatter,:,:,:]
            file_name = f"model{i2}_scatter{scatter+1}"
            np.save(os.path.join(save_folder, file_name),frames_rf)
    i2 +=1
    
for file in files_list:
    if file.endswith('.mat'):
        file_path = os.path.join(folder_path, file)
        data_gt = np.stack((sio.loadmat(file_path)['axial'], 
                                    sio.loadmat(file_path)['lateral']), axis=0)

        files_processed = data_gt.astype(np.float32)
        files_processed = files_processed[:,gt_frame_range, row_delete:-row_delete, 
                                                col_delete:-col_delete]
        for scatter in range(10):
            for i in range(files_processed.shape[1]):
                gt_slice = files_processed[:,i,:,:]
                file_name = f"gt_model{i2}_scatter{scatter+1}_frame{i+2}"
                np.save(os.path.join(save_folder, file_name),gt_slice)
                
        i2+=1        



    
