import argparse
import os
import time

import torch
import resnet
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.utils
import torchvision.datasets as datasets
import torch.utils.data

parser = argparse.ArgumentParser(description='Resnets for CIFAR10 in pytorch')
parser.add_argument('--arch', dest='arch', help='Choose the model', default='resnet20')
parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the models', default='temp_save',
                    type=str)
parser.add_argument('-b','--batch-size',dest='batch_size', default=128, type=int)
parser.add_argument('--workers', dest='workers', default=4, type=int)
parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float, dest='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('--start-epoch', default=0, type=int, dest='start_epoch')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--epochs', default=200, type=int, dest='epochs')
parser.add_argument('--save-every', default=10, type=int ,dest='save_every')
parser.add_argument('--print-freq', default=50, type=int, dest='print_freq')
parser.add_argument('--resume', default='', type=str)
best_pred=0


def main():
    global args, best_pred
    args = parser.parse_args()

    #create directory used to save models
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #use gpu to compute parallelly
    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #optimize the convolution operations
    cudnn.benchmark

    #set the normalizer (normalize the images from 3 tunnels)
    #usually use the imageNet dataset's mean and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    #load the data
    train_loader = torch.utils.data.DataLoader(
        #the CIFAR10 dataset is included in the torchvision.datasets package
        datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transforms.Compose([
                                #randomly processing some images to increase robustness
                                transforms.RandomHorizontalFlip(p=0.5),
                                ## 这里的图片是32*32，所以crop函数会自动根据 padding的值来选择切割的中心位置，保证padding以后和原大小一样。
                                transforms.RandomCrop(32,padding=4),
                                transforms.ToTensor(),
                                normalize
                            ])
                         ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    #load the validation_loader
    validation_Loader=torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    #define the loss function
    loss = torch.nn.CrossEntropyLoss().cuda()

    #define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #define the lr scheduler
    lr_scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150])
    lr_scheduler.last_epoch = args.start_epoch-1

    if args.evaluate:
        validate(validation_Loader, model, loss)

    for epoch in range(args.start_epoch, args.epochs):
        print('current lr{:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, loss, optimizer, epoch)
        lr_scheduler.step()

        #remember the best prediciton
        pred = validate(validation_Loader, model, loss)
        is_best = pred>best_pred
        best_pred = max(pred, best_pred)

        if epoch>0 and epoch % args.save_every ==0:
            save_checkpoint(
                {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'best_pred': best_pred},
                is_best,
                filename=os.path.join(args.save_dir,'check_point.th')
            )

        save_checkpoint(
            {'state_dict': model.state_dict(),
             'best_pred': best_pred,
             },
            is_best, filename=os.path.join(args.save_dir, 'model.th')
        )

def validate(validation_Loader, model, loss):
    batch_time=AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end =time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(validation_Loader):
            y_var = y.cuda()
            x_var = x.cuda()

            output = model(x_var)
            loss_output = loss(output, y_var)

            output = output.float()
            loss_output  = loss_output.float()

            pred = accuracy(output, y_var)

            losses.update(loss_output.item(),x.size(0))
            top1.update(pred.item(), x.size(0))

            #time
            batch_time.update(time.time() -end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test:[{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) \t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i,len(validation_Loader),batch_time=batch_time,loss=losses,top1=top1)
                )
    print('* pred {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train(train_loader, model, loss, optimizer, epoch):
    batch_time=AverageMeter()
    data_time=AverageMeter()
    losses =AverageMeter()
    top1=AverageMeter()

    model.train()
    end=time.time()
    for i ,(x,y) in enumerate(train_loader):
        data_time.update(time.time()-end)
        y = y.cuda()
        x = x.cuda()

        #compute the model
        y_hat = model(x)
        loss_y = loss(y_hat,y)

        #compute the gradient and do SGD step
        optimizer.zero_grad()
        loss_y.backward()
        optimizer.step()
        y = y.float()
        x = x.float()

        pred = accuracy(y_hat, y)
        losses.update(loss_y.item(), x.size(0))
        top1.update(pred.item(), x.size(0))

        batch_time.update(time.time()-end)
        end = time.time()

        if i% args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val =val
        self.sum += val *n
        self.count += n
        self.avg =self.sum/self.count

def accuracy(output, y):
    batch_size = y.size(0)
    # for output n*10 -> topk -> pred n*1 -> t() -> 1 * n
    # for y one-dimension n -> view(1,-1) -> 1 *n
    # for correct -> 1 *n -> view(-1) -> one-dimension n-> .float.sum(0) -> correct ones
    # correct ones / batch size *100(percentage)

    _, pred = output.topk(k=1, dim=1)
    pred = pred.t()
    correct = pred.eq(y.view(1,-1).expand_as(pred))
    correct_1 = correct.view(-1).float().sum(0)
    return correct_1.mul_(100/batch_size)



if __name__== '__main__':
    main()