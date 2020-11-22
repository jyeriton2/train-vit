import os, sys
import torch
from progress.bar import Bar

class Trainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        if optimizer != None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), opt.lr)
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        
    def set_device(self, gpus, device):
        if len(gpus) < 1:
            print('device does not have GPU')
            sys.exit()
        self.model = self.model.to(device)
        self.loss = self.loss.to(device)

    def run_epoch(self, phase, epoch, data_loader):
        loss = self.loss
        if phase == 'train':
            loss.train()
        else:
            loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt

        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format('train', opt.exp_id), max=num_iters)
        
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            
            out, loss, loss_states = loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        bar.finish()

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)
    
    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)



