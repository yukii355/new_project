#!/usr/bin/python
# -*- coding: utf-8 -*-

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import visdom, torch, torchvision
from cartoongan import Generator, discriminator
from dataloader import image_dataset
from dataloader import optical_flow
# from dataloader import video_dataset
from torch.optim import Adam
import argparse
import numpy as np
import cv2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_normal(m.weight.data)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--imgdir', type=str, default="/home/moriyama/", help='path of images')
    parser.add_argument('--videodir', type=str, default="/home/moriyama/", help='path of videos')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lamda', type=float, default=4, help='the weight of perceptual loss')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPU')

    opt = parser.parse_args()
    print(opt)

    # vis = visdom.Visdom()
    # lineg = vis.line(Y=np.arange(10), env="G_Loss")
    # lined = vis.line(Y=np.arange(10), env="D_Loss")



    '''
    Dataloaderは、訓練・テストデータのロード・前処理をするためのモジュール.
    結構実装が面倒だったり、適当にやりすぎると、メモリ周りできついことになる.
    Pytorchではデフォルトでdataloaderを用意しているが、ライブラリも用意されている,
    torchvision:画像周りのデータローダ、前処理、有名モデル(densenet,alex,resnet,vgg等)

    '''

    # cartoonGANでは画像を読み込み
    real_image_dataset = image_dataset(path=opt.imgdir)
    ani_image_dataset = image_dataset(path=opt.imgdir, type="ani_images/")
    real_image_loader = DataLoader(real_image_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=2)
    ani_image_loader = DataLoader(ani_image_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=2, drop_last=True)

    '''
    # videoデータセットを読み込み
    real_video_dataset = video_dataset(path=opt.videodir)
    ani_video_dataset = video_dataset(path=opt.videodir, type="ani_videos/")
    real_video_loader = DataLoader(real_video_dataset, batch_size=opt.batchsize,shuffle=True, num_workers=2)
    ani_video_loader = DataLoader(ani_video_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=2, drop_last=True)
    '''


    G_net = Generator(in_dim=3).to(device)
    D_net = discriminator(in_dim=3).to(device)
    ## G_net.load_state_dict(torch.load("weights_b/G_init.pkl"))
    # G_net.apply(weights_init)
    # D_net.apply(weights_init)


    # print(Vggmodel)でわかる通り、(features)と(classifier)の２つのSequentialモデルから成り立っている.
    # pretrained=TrueにするとImageNetで学習済みの重みがロードされる.
    Vgg_model = torchvision.models.vgg19(pretrained=True)
    # Vgg19モデルの中の、features内の、始めから26行分を取ってくる
    Vgg = nn.Sequential(*list(Vgg_model.features)[:26])
    Vgg.to(device)


    '''
    optimパッケージを使うと任意の最適化手法を用いてパラメータの更新を行うことができる。
    ex) optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum=0.9)
    上の例のように、最適化手法のパラメータを設定したら、backward計算をするたびにstep()を呼び出すことで更新を行える.

    # 順伝搬
    out = model(input)
    # ロス計算
    loss = criterion(out, target)
    # 勾配の初期化
    optimizer.zero_grad()
    # 勾配の計算
    loss.backward()
    # パラメータの更新
    optimizer.step()
    '''

    # 上のoptimを使わない場合、optimizer.step()の部分を以下のように書き換えることでパラメータの更新が可能
    for param in Vgg.parameters():
        param.requires_grad = False

    if opt.ngpu > 2:
        G_net = nn.DataParallel(G_net)
        D_net = nn.DataParallel(D_net)
        Vgg = nn.DataParallel(Vgg)
    criterion = nn.L1Loss().to(device) #Loss関数の定義
    criterionMSE = nn.MSELoss().to(device) #Loss関数の定義

    G_optimizer = Adam(G_net.parameters(), lr=opt.lr, betas=(0.5,0.999))
    D_optimizer = Adam(D_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    G_net.train()
    D_net.train()
    G_loss_list = []
    D_loss_list = []
    g_loss_list = []
    per_loss_list = []
    d_loss_fakeani_list = []
    d_loss_realani_list = []
    j = 0
    for epoch in range(opt.epochs):
        for i, (real_img, ani_img) in enumerate(zip(real_image_loader, ani_image_loader)):
            real_img = Variable(real_img).to(device)
            ani_img = Variable(ani_img).to(device)



            image_list1 = []
            image_list2 = []
            if j != 2:
                image_list1.append(real_img)
                image_list2.append(ani_img)
                j = j + 1
            else:
            # real_video = Variable(real_video).to(device)
            # ani_video = Variable(ani_video).to(device)
            #train discriminator
                '''
                prvs = cv2.cvtColor(image_list1[i], cv2.COLOR_BGR2GRAY)
                next = cv2.cvtColor(image_list1[i + 1], cv2.COLOR_BGR2GRAY)

                optical = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                '''

                j = 0
                abc = np.zeros(5, np.float32)
                optical_image = optical_flow(i)
                abc[0] = image_list1[0][:, :, 0]
                abc[1] = image_list1[0][:, :, 1]
                abc[2] = image_list1[0][:, :, 2]
                abc[3] = optical_image[:, :, 0]
                abc[4] = optical_image[:, :, 1]
                abc = torch.Tensor(abc)
                D_optimizer.zero_grad()

                #real_aniout = D_net(ani_img)
                real_aniout = D_net(image_list2)
                real_anilabel = Variable(torch.ones_like(real_aniout)).to(device)

                fake_anilabel = Variable(torch.zeros_like(real_aniout)).to(device)

                d_loss_realani = criterionMSE(real_aniout, real_anilabel)

               # fake_ani = G_net(real_img)
                fake_ani = G_net(abc)


                fake_aniout = D_net(fake_ani)
                d_loss_fakeani = criterionMSE(fake_aniout, fake_anilabel)

                d_loss = 0.5 * (d_loss_fakeani + d_loss_realani)

                d_loss.backward()
                D_optimizer.step()


                '''
                <shape>
                real_img = torch.Size([1, 3, 224, 224])
                ani_img = torch.Size([1, 3, 224, 224])
                real_aniout = torch.Size([1, 1, 56, 56])
                fake_ani = torch.Size([1, 3, 224, 224])
                fake_aniout = torch.Size([1, 1, 56, 56])

                '''

                #train generator
                G_optimizer.zero_grad()

                #fake_ani = G_net(real_img)
                fake_ani = G_net(image_list1)
                fake_aniout = D_net(fake_ani)
                g_loss = criterionMSE(fake_aniout, real_anilabel)
                #perceptual loss
                #per_loss = criterion(Vgg(fake_ani),Vgg(real_img))
                per_loss = criterion(Vgg(fake_ani), Vgg(image_list1))
                G_loss =  g_loss + opt.lamda * per_loss

                G_loss.backward()
                G_optimizer.step()

                G_loss_list.append(G_loss.item())
                g_loss_list.append(g_loss.item())
                per_loss_list.append(per_loss.item())
                D_loss_list.append(d_loss.item())
                d_loss_fakeani_list.append(d_loss_fakeani.item())
                d_loss_realani_list.append(d_loss_realani.item())

                if i % 40 == 0:
                    # torchvision.utils.save_image((fake_ani), 'samples/' +"genepoch" + str(epoch+1) + "batch" + str(i + 1) + '.jpg', normalize=True)
                    # torchvision.utils.save_image((real_img), 'samples/' +"realepoch" + str(epoch+1) + "batch" + str(i + 1) + '.jpg', normalize=True)
                    # torchvision.utils.save_image((ani_img), 'samples/' +"aniepoch" + str(epoch+1) + "batch" + str(i + 1) + '.jpg', normalize=True)

                    G_loss = sum(G_loss_list) / len(G_loss_list)
                    D_loss = sum(D_loss_list) / len(D_loss_list)
                    g_loss = sum(g_loss_list) / len(g_loss_list)
                    per_loss = sum(per_loss_list) / len(per_loss_list)
                    d_loss_realani = sum(d_loss_realani_list) / len(d_loss_realani_list)
                    d_loss_fakeani = sum(d_loss_fakeani_list) / len(d_loss_fakeani_list)

                    print("Epoch:{:.0f},Batch/Batchs: {:.0f}/{:.0f}, D_loss: {:.3f} d_loss_realani: {:.3f}  d_loss_fakeani: {:.3f} "
                          "G_loss: {:.3f} g_loss: {:.3f} per_loss: {:.3f}".format(
                            epoch+1,i,len(ani_image_loader),
                            D_loss, d_loss_realani, d_loss_fakeani,
                            G_loss, g_loss,per_loss))
                    '''
                    vis.line(Y=np.column_stack((np.array(G_loss_list), np.array(g_loss_list), np.array(per_loss_list))),
                              X=np.column_stack((np.arange(len(G_loss_list)), np.arange(len(G_loss_list)), np.arange(len(G_loss_list) ))),
                              update="new",
                              opts=dict(title="G_loss",legend=["G_loss", "g_loss", "per_loss"]),  win=lineg,
                              env="G_Loss")
                    #
                    vis.line(Y=np.column_stack((np.array(D_loss_list),np.array(d_loss_fakeani_list), np.array(d_loss_realani_list))),
                              X=np.column_stack((np.arange(len(G_loss_list)), np.arange(len(G_loss_list)),
                                                np.arange(len(G_loss_list)))),
                              update="new",
                              opts=dict(title="D_loss",legend=["D_loss", "d_fake", "d_real"]),  win=lined,
                    env="D_Loss")
                    '''
            if (epoch+1) % 5 == 0:
                torch.save(G_net.state_dict(), "weights/" + str(epoch) + ".pkl")



if __name__ == '__main__':
    main()













