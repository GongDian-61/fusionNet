import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from net_2E import Encoder, Decoder, EncoderDecoder
from loss import GradNorm
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import time
import datetime
from os import path, makedirs
from dataset_self import NpyDataset


# 
logs_dir = r"./runs"
# path = 
# **Logging and Checkpointing**
dt_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir = path.join(logs_dir, f"2E1D_{dt_now}")
checkpoint_dir = path.join(logs_dir, "training_checkpoints")
makedirs(checkpoint_dir, exist_ok=True)
# checkpoint_path = path.join(checkpoint_dir, "ckpt.pt")
#
writer = SummaryWriter(logs_dir)
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir',logs_dir])
url = tb.launch()
print(f"tb is at: {url}")

# hyper-para
batch_size = 1
lr = 1e-4
learning_rate_decay = 1e-1
num_epochs = 5

#
device = 'cuda'
E = Encoder(in_channels=2).to(device)
E2 = Encoder(in_channels=2).to(device)
D = Decoder(out_channels=2).to(device)
optimizer_e = torch.optim.Adam(E.parameters(), lr=lr)
optimizer_e2 = torch.optim.Adam(E2.parameters(), lr=lr)
optimizer_d = torch.optim.Adam(D.parameters(),lr=lr)

l1Loss = nn.L1Loss()
mseLoss = nn.MSELoss()

data_folder_t = r"/media/eadu/DATA/fusionNet/displacementBYtradMethod/sim_result"
data_folder_o = r"/media/eadu/DATA/displacementBYoptflow"
data_folder_gt = r"/media/eadu/DATA/sim_groudtruth/gt_in_slices_repeat"
train_data = NpyDataset(data_folder_t,data_folder_o,data_folder_gt)
train_loader = DataLoader(train_data,batch_size=batch_size,
                            shuffle=True,num_workers=0)
loader = {'train_frames': train_loader, }                           

prev_time = time.time()
for epoch in range(num_epochs):
    # para_loader = pl.ParallelLoader(train_loader, [device])
    # for batch_idx, (inputs, targets) in enumerate(para_loader.per_device_loader(device)):
    batch_idx=0
    for data_trad, data_opt, data_gt in loader['train_frames']:
        E.train()
        E2.train()
        D.train()
        E.zero_grad()
        E2.zero_grad()
        D.zero_grad()
        data_trad, data_opt, data_gt = data_trad.cuda(), data_opt.cuda(), data_gt.cuda()          
        optimizer_e.zero_grad()
        optimizer_e2.zero_grad()
        optimizer_d.zero_grad()

        feature1_o, feature2_o,feature3_o,feature4_o = E(data_opt)
        feature1_t, feature2_t,feature3_t,feature4_t = E2(data_trad)
        feature1 = torch.cat((feature1_o,feature1_t),dim=1)
        feature2 = torch.cat((feature2_o,feature2_t),dim=1)
        feature3 = torch.cat((feature3_o,feature3_t),dim=1)
        feature4 = torch.cat((feature4_o,feature4_t),dim=1)

        pred = D(feature4, feature3,feature2,feature1)

        log_l1 = 5*l1Loss(pred, data_gt)
        log_mse = 5*mseLoss(pred, data_gt)
        log_grad = 2*GradNorm(pred)
        
        loss = log_l1 + log_mse + log_grad

        loss.backward()

        # Log parameters and gradients to TensorBoard
        for name, param in E.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'Encoder/{name}', param, epoch)
                writer.add_histogram(f'Encoder/{name}.grad', param.grad, epoch)
        
        for name, param in D.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'Decoder/{name}', param, epoch)
                writer.add_histogram(f'Decoder/{name}.grad', param.grad, epoch)

        # # Print parameters and gradients
        # for name, param in E.named_parameters():
        #     if param.requires_grad:
        #         print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Encoder Param {name}: {param.data}")
        #         print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Encoder Grad {name}: {param.grad}")


        optimizer_e.step()
        optimizer_e2.step()
        optimizer_d.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")
        batch_idx +=1

        # Determine approximate time left
        batches_done = epoch * len(loader['train_frames']) + batch_idx
        batches_left = num_epochs * len(loader['train_frames']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                batch_idx,
                len(loader['train_frames']),
                loss.item(),
                time_left,
            )
        )

    
    writer.add_scalar('Loss l1', log_l1, epoch) 
    writer.add_scalar('loss mse', log_mse, epoch)
    writer.add_scalar('loss grad', log_grad, epoch)

    # Save checkpoint after each epoch
    if (epoch+1)%100==0 or epoch == num_epochs-1:
        checkpoint = {
                        'Encoder': E.state_dict(),
                        'Encoder2': E2.state_dict(),
                        'Decoder': D.state_dict(),
                    }
        torch.save(checkpoint, path.join("checkpoint_dir" + dt_now + '.pth'))
    # xser.save([E.state_dict(), D.state_dict(),  optimizer_e.state_dict(),  optimizer_d.state_dict()], checkpoint_path)
    # print(f"Checkpoint saved at {checkpoint_path}")