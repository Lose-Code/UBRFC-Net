import os
import time
import torch
from Option import opt
from Loss import SSIMLoss
from CR import ContrastLoss
from Parameter import Parameter
from Dataset import CustomDatasetLoader
from GAN import Generator, Discriminator

lr = opt.lr
d_out_size = 30
device = "cuda:"+opt.device
epochs = opt.epochs

train_dataloader = CustomDatasetLoader(root_dir=opt.train_root, isTrain=True,batch_size=opt.batch_size,y_name='clear').load_data().dataloader
test_dataloader = CustomDatasetLoader(root_dir=opt.test_root, isTrain=False,batch_size=1,y_name='clear').load_data().dataloader

generator = Generator()
discriminator = Discriminator()  # 输出 bz 1 30 30
parameterNet = Parameter()

criterionSsim = SSIMLoss()
criterion = torch.nn.MSELoss()
criterionP = torch.nn.L1Loss()
criterionC = ContrastLoss(device,True)

if os.path.exists(opt.model_path) and opt.is_continue:
    lr = torch.load(opt.model_path)["lr"]

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_T = torch.optim.Adam([
    {'params': generator.parameters(), 'lr': lr},
    {'params': parameterNet.parameters(), 'lr': lr}
])

timeStamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

generator.to(device)
criterion.to(device)
criterionP.to(device)
parameterNet.to(device)
discriminator.to(device)
criterionSsim.to(device)

scheduler_T = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_T, T_max=epochs, eta_min=0, last_epoch=-1)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=0, last_epoch=-1)

if os.path.exists(opt.model_path) and opt.is_continue:
    print("加载模型")
    generator.load_state_dict(torch.load(opt.model_path)["generator_net"])
    discriminator.load_state_dict(torch.load(opt.model_path)["discriminator_net"])
    parameterNet.load_state_dict(torch.load(opt.model_path)["parameterNet_net"])
    optimizer_T.load_state_dict(torch.load(opt.model_path)["optimizer_T"])
    optimizer_D.load_state_dict(torch.load(opt.model_path)["optimizer_D"])
    iter_num = torch.load(opt.model_path)["epoch"]
    scheduler_T.step()
    scheduler_D.step()
else:
    iter_num = -1