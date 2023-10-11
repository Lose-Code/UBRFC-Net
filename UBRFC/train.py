import numpy as np
from tqdm import tqdm
from test import test
from Declaration import *

def train():
    global lr
    generator.train()
    discriminator.train()
    parameterNet.train()
    psnrs,ssims = [],[]
    for epoch in range(iter_num + 1, epochs):
        loss_total = 0
        for i, (x, y,_) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False,
                              desc="epoch is %d" % epoch):

            x = x.to(device)
            y = y.to(device)

            real_label = torch.ones((x.size()[0], 1, d_out_size, d_out_size), requires_grad=False).to(device)
            fake_label = torch.zeros((x.size()[0], 1, d_out_size, d_out_size), requires_grad=False).to(device)

            real_out = discriminator(y)
            loss_real_D = criterion(real_out, real_label)

            fake_img = generator(x)

            fake_out = discriminator(fake_img.detach())
            loss_fake_D = criterion(fake_out, fake_label)

            loss_D = (loss_real_D + loss_fake_D) / 2

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            #D_loss = loss_D.item()

            #fake_img_ = generator(x)
            output = discriminator(fake_img)
            haze, dehaze = parameterNet(x, fake_img)

            loss_G = criterion(output, real_label)
            loss_P = criterionP(haze, x)  # L1
            loss_Right = criterionP(fake_img, dehaze.detach())  # 右拉

            loss_ssim = criterionSsim(fake_img, dehaze.detach())  # 结构
            loss_C1 = criterionC(fake_img, dehaze.detach(), x, haze.detach())  # 对比下
            loss_C2 = criterionC(haze, x, fake_img.detach(), dehaze.detach())  # 对比上

            total_loss = loss_G + loss_P + loss_Right + 0.1 * loss_ssim + loss_C1 + loss_C2

            optimizer_T.zero_grad()
            total_loss.backward()
            optimizer_T.step()

            lr = scheduler_T.get_last_lr()[0]
            # G_loss = loss_G.item()
            # P_loss = loss_P.item()
            loss_total = total_loss.item()

        if (epoch % 10 == 0 and epoch != 0) or epoch>=30:
            ###测试
            with torch.no_grad():
                ssim_eval, psnr_eval = test(generator, parameterNet, test_dataloader)
                ssims = np.append(ssims, ssim_eval)
                psnrs = np.append(psnrs, psnr_eval)

            msg_clear = f"epoch: {epoch}|lr:{lr}|训练：[total loss: %4f]" % (loss_total) + "测试：" + f'ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}\n'
            print(msg_clear)

            ##保存log文件
            file = open('./Log/log_train/train_' + str(timeStamp) + '.txt', 'a+')
            file.write(msg_clear)
            file.close()
            model = {"generator_net": generator.state_dict(),"discriminator_net": discriminator.state_dict(),"parameterNet_net": parameterNet.state_dict(),
                     "optimizer_T": optimizer_T.state_dict(),"optimizer_D": optimizer_D.state_dict(),"lr": lr, "epoch": epoch}

            torch.save(model, "./model/model_%d" % epoch)
            torch.save(model, "./model/last_model.pth")
            if psnr_eval == psnrs[np.array(psnrs).argmax()]:
                max_msg = "epoch:%d,配对ssim:%f,最大psnr:%f\n" % (epoch, ssim_eval, psnr_eval)
                files = open('./Log/max_log/max_log' + str(timeStamp) + '.txt', 'a+')
                files.write(max_msg)
                files.close()
                torch.save(model, "./model/best_model.pth")

        scheduler_T.step()
        scheduler_D.step()





if __name__ == '__main__':
    train()
