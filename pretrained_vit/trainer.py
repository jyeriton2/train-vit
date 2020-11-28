import os, sys
import time
import torch
from torch.autograd import Variable
from progress.bar import Bar
from image_utils import AverageMeter

class Trainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        if optimizer != None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), opt.lr)
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer.add_param_group({'params': self.loss.parameters()})
        self.device = None
        
    def set_device(self, gpus, device):
        if len(gpus) < 1:
            print('device does not have GPU')
            sys.exit()
        if self.opt.multi_gpu_use == True:
            self.model = torch.nn.DataParallel(self.model).cuda(device)   # for use multi-gpu
        else:
            self.model = self.model.to(device)
        self.loss = self.loss.to(device)
        self.device = device
        for state in self.optimizer.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        loss = self.loss
        self.model.train()
        if phase == 'train':
            loss.train()
        else:
            loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        data_time, batch_time = AverageMeter(), AverageMeter()
        losses = AverageMeter()
        accuracyes = AverageMeter()

        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format('train', opt.exp_id), max=num_iters)
        end = time.time()
        
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            
            if self.device is not None and self.device is not 'cpu':
                _in = batch['image'].cuda(self.device, non_blocking=True)
                _label = batch['label']
                _label = _label.cuda(self.device, non_blocking=True)
            else:
                _in = batch['image']
                _label = batch['label']

            model_result = self.model(_in)
            _loss = loss(Variable(model_result, requires_grad=True), Variable(_label))
            _loss = _loss.mean()
            _acc = self.cal_accuracy(model_result, _label, topk=(1,))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            Bar.suffix = '{phase}: [{epoch}][{iter_id}/{num_iters}]|Total: {total:} |ETA: {eta:}'.format(epoch=epoch, iter_id=iter_id, num_iters=num_iter, phase=phase, total=bar.elapsed_td, eta=bar.eta_td)
            losses.update(_loss.item(), batch['image'].size(0))
            Bar.suffix = Bar.suffix + '|loss : {:.4f}'.format(losses.avg)
            accuracyes.update(_acc[0], batch['image'].size(0))
            Bar.suffix = Bar.suffix + '|accu : {:.4f}'.format(int(accuracyes.avg))
            
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            bar.next()

        bar.finish()

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)
    
    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
    
    def cal_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            
            return res



