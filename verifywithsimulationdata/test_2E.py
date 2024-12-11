import torch
from net_2E import Encoder, Decoder
from torch.utils.data import DataLoader
from dataset_self import NpyDataset

def nrmse(true, pred):
    rmse=torch.sqrt(torch.mean((true-pred)**2))
    range=torch.max(true) - torch.min(true)
    return rmse/range

# read weights
ckpt_path = "checkpoint_dir20241210_143420.pth"
# r"./runs/fcn_origin_20241031_140411/events.out.tfevents.1730379851.eadu-System-Product-Name.5637.0"
#
device = 'cuda'
E = Encoder(in_channels=2).to(device)
E2 = Encoder(in_channels=2).to(device)
D = Decoder(out_channels=2).to(device)
#
E.load_state_dict(torch.load(ckpt_path)['Encoder'])
E2.load_state_dict(torch.load(ckpt_path)['Encoder2'])
D.load_state_dict(torch.load(ckpt_path)['Decoder'])
#
E.eval()
E2.eval()
D.eval()


# read test img
data_folder_t = r"./data/trad"
data_folder_o = r"./data/opt"
data_folder_gt = r"./data/gt"
train_data = NpyDataset(data_folder_t,data_folder_o,data_folder_gt,flag='test')
train_loader = DataLoader(train_data,batch_size=1,
                            shuffle=True,num_workers=0)
loader = {'train_frames': train_loader, }   

with torch.no_grad():
    for data_trad, data_opt, data_gt in loader['train_frames']:
        
        data_trad, data_opt, data_gt = data_trad.cuda(), data_opt.cuda(), data_gt.cuda()       

        feature1_o, feature2_o,feature3_o,feature4_o = E(data_opt)
        feature1_t, feature2_t,feature3_t,feature4_t = E2(data_trad)
        feature1 = torch.cat((feature1_o,feature1_t),dim=1)
        feature2 = torch.cat((feature2_o,feature2_t),dim=1)
        feature3 = torch.cat((feature3_o,feature3_t),dim=1)
        feature4 = torch.cat((feature4_o,feature4_t),dim=1)
        pred = D(feature4, feature3,feature2,feature1)

        val_fuse = nrmse(data_gt, pred)
        val_opt = nrmse(data_gt,data_opt)
        val_t = nrmse(data_gt,data_trad)
        print("mean fuse:",val_fuse,", mean opt. flow:",val_opt,", mean trad:",val_t)