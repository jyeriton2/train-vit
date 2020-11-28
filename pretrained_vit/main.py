import os, sys, json
import argparse

import torch
import torchvision
import torch.utils.data
from torchvision.transforms import transforms as T
import torch_optimizer as optim
from opts import opts
from VIT import VIT
from dataload import LoadImagesAndLabels
from trainer import Trainer


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch' : epoch,
            'state_dict' : state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def main(opt):
    torch.manual_seed(114)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(opt.device)
    
    transforms = T.Compose([T.ToTensor()])
    Dataset = LoadImagesAndLabels(path=opt.data_path, img_size=(512, 512), transforms=transforms)
    
    start_epoch = 0
    train_loader = torch.utils.data.DataLoader(
            Dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
            )
    
    model = VIT(img_size=512, patch_size=16)
    # optimizer = optim.Yogi(model.parameters(), opt.lr)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        _dict_train, _ = trainer.train(epoch, train_loader)
        
        if epoch % 5 == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)







if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    opt = opts().parse()
    main(opt)
