import argparse
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.parse_config import *

# Hyperparameters: train.py --evolve --epochs 2 --img-size 320, Metrics: 0.204      0.302      0.175      0.234 (square smart)
hyp = {'xy': 0.1,  # xy loss gain  (giou is about 0.02)
       'wh': 0.1,  # wh loss gain
       'cls': 0.04,  # cls loss gain
       'conf': 4.5,  # conf loss gain
       'iou_t': 0.5,  # iou target-anchor training threshold
       'lr0': 0.0001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay

hyp_dis = {'lr0': 0.001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=100,  # 500200 batches at bs 4, 117263 images = 68 epochs
        batch_size=16,
        accumulate=4,  # effective bs = 64 = batch_size * accumulate
        freeze_backbone=False,
        transfer=False  # Transfer learning (train only YOLO layers)
):
    init_seeds()
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    bench = weights + 'bench.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = True  # possibly unsuitable for multiscale
    img_size_test = img_size  # image size for testing

    if opt.multi_scale:
        img_size_min = round(img_size / 32 / 1.5)
        img_size_max = round(img_size / 32 * 1.5)
        img_size = img_size_max * 32  # initiate with maximum multi_scale size

    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    target_train_path = data_dict['target_train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Net().to(device)
    dis = Dnet().to(device)

    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.conv3.parameters():
        param.requires_grad = False
    for param in model.bn3.parameters():
        param.requires_grad = False
#    for param in model.conv5.parameters():
#        param.requires_grad = False
#    for param in model.bn5.parameters():
#        param.requires_grad = False

    print(model.conv5.weight.requires_grad, model.conv7.weight.requires_grad, model.bn5.weight.requires_grad, model.bn5.running_mean.requires_grad)
    # Optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    optimizer_dis = optim.SGD(dis.parameters(), lr=hyp_dis['lr0'], momentum=hyp_dis['momentum'], weight_decay=hyp_dis['weight_decay'])



    #Discriminator loss
    criterion_d = nn.BCEWithLogitsLoss()

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    nf = 33  # yolo layer size (i.e. 255)
    
    #########################################################
    if resume:  # Load previously saved model
        chkpt = torch.load(latest, map_location=device)  # load checkpoint
        model.load_state_dict(chkpt['model'])
        dis.load_state_dict(chkpt['dis'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        if chkpt['optimizer_dis'] is not None:
            optimizer_dis.load_state_dict(chkpt['optimizer_dis'])
        del chkpt
    #########################################################################################
    else:
        chkpt = torch.load(bench,map_location=device)
        model.load_state_dict(chkpt['model'])
        del chkpt


    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    scheduler_dis = lr_scheduler.MultiStepLR(optimizer_dis, milestones=[round(opt.epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    scheduler_dis.last_epoch = start_epoch - 1

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.xlabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  rect=False)

    dataset_target = LoadTargetImages(target_train_path,
                                  img_size,
                                  batch_size,
                                  augment=True)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        dis = torch.nn.parallel.DistributedDataParallel(dis)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=True,  # disable rectangular training if True
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    dataloader_target = DataLoader(dataset_target,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=True,  # disable rectangular training if True
                            pin_memory=True,
                            collate_fn=dataset_target.collate_fn)


    # Remove old results
#    for f in glob.glob('*_batch*.jpg') + glob.glob('results*.txt'):
#        os.remove(f)

    # Start training
    model.hyp = hyp  # attach hyperparameters to model
    dis.hyp = hyp_dis
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model)
    model_info(dis)
    nb = len(dataloader)
    nbt = len(dataloader_target)
    count_batches = min(nb,nbt)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(count_batches / 5 + 1), 1000)  # burn-in batches
    t, t0 = time.time(), time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        dis.train()
        print(('\n%8s%12s' + '%10s' * 8) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'dis_loss' , 'targets', 'time'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # # Update image weights (optional)
        # w = model.class_weights.cpu().numpy() * (1 - maps)  # class weights
        # image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        # dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # random weighted index
        
        dataiter = iter(dataloader)
        dataiter_target = iter(dataloader_target)

        mloss = torch.zeros(6).to(device)  # mean losses
        
        lmbda = ((2/(1+np.exp(-10*((epoch+0.0)/epochs))))-1)
#        lmbda = 0
        for i in range(count_batches):
            total_loss = 0
            loss = 0
            loss_dis = 0

            imgs,targets,_,_ = dataiter.next()
            imgs = imgs.to(device)
            targets = targets.to(device)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                lr_dis = hyp_dis['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr
                for x in optimizer_dis.param_groups:
                    x['lr'] = lr_dis

            # Run model
            pred,dis_input = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            dis_out_source = dis(dis_input,lmbda)
            actual_d = torch.zeros(dis_out_source.shape)
            actual_d = actual_d.to(device)
            loss_d = criterion_d(dis_out_source,actual_d)
            loss_dis += loss_d 

            #target domain
            imgs_t,_,_ = dataiter_target.next()
            imgs_t = imgs_t.to(device)
            _,dis_input_t = model(imgs_t)
            dis_out_target = dis(dis_input_t,lmbda)
            actual_d_t = torch.ones(dis_out_target.shape)
            actual_d_t = actual_d_t.to(device)
            loss_d_t = criterion_d(dis_out_target,actual_d_t)
            loss_dis += loss_d_t            

            #Compute grads
            total_loss = loss + 2*loss_dis
            total_loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == count_batches:
                optimizer_dis.step()
                optimizer.step()
                optimizer.zero_grad()
                optimizer_dis.zero_grad()

            #Print batch results
            mloss = (mloss * i + torch.cat((loss_items.to(device),torch.Tensor([loss_dis]).to(device))).to(device)) / (i + 1)  # update mean losses
            s = ('%8s%12s' + '%10.3g' * 8) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss, len(targets), time.time() - t)
            t = time.time()
            print(s)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 10)) or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size_test, model=model,
                                          conf_thres=0.1)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 10)) or epoch == epochs - 1:
            with torch.no_grad():
                results_70, maps_70 = test.test(cfg, './data/sdd70.data', batch_size=batch_size, img_size=img_size_test, model=model,
                                          conf_thres=0.1)


        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        with open('results70.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results_70 + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        save = (not opt.nosave) or (epoch == epochs - 1)
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'dis': dis.state_dict(),
                     'optimizer_dis':optimizer_dis.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    dt = (time.time() - t0) / 3600
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch + 1, dt))
    return results
            ###make changes here
        
def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=68, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--accumulate', type=int, default=8, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco_64img.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt)

    if opt.evolve:
        opt.notest = True  # save time by only testing final epoch
        opt.nosave = True  # do not save checkpoints

    # Train
    results = train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=opt.resume or opt.transfer,
        transfer=opt.transfer,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
    )

    # Evolve hyperparameters (optional)
    if opt.evolve:
        best_fitness = results[2]  # use mAP for fitness

        # Write mutation results
        print_mutation(hyp, results)

        gen = 1000  # generations to evolve
        for _ in range(gen):

            # Mutate hyperparameters
            old_hyp = hyp.copy()
            init_seeds(seed=int(time.time()))
            s = [.3, .3, .3, .3, .3, .3, .3, .03, .3]  # xy, wh, cls, conf, iou_t, lr0, lrf, momentum, weight_decay
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 1.1  # plt.hist(x.ravel(), 100)
                hyp[k] = hyp[k] * float(x)  # vary by about 30% 1sigma

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay']
            limits = [(1e-4, 1e-2), (0, 0.90), (0.70, 0.99), (0, 0.01)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Determine mutation fitness
            results = train(
                opt.cfg,
                opt.data_cfg,
                img_size=opt.img_size,
                resume=opt.resume or opt.transfer,
                transfer=opt.transfer,
                epochs=opt.epochs,
                batch_size=opt.batch_size,
                accumulate=opt.accumulate,
            )
            mutation_fitness = results[2]

            # Write mutation results
            print_mutation(hyp, results)

            # Update hyperparameters if fitness improved
            if mutation_fitness > best_fitness:
                # Fitness improved!
                print('Fitness improved!')
                best_fitness = mutation_fitness
            else:
                hyp = old_hyp.copy()  # reset hyp to

            # # Plot results
            # import numpy as np
            # import matplotlib.pyplot as plt
            # a = np.loadtxt('evolve_1000val.txt')
            # x = a[:, 2] * a[:, 3]  # metric = mAP * F1
            # weights = (x - x.min()) ** 2
            # fig = plt.figure(figsize=(14, 7))
            # for i in range(len(hyp)):
            #     y = a[:, i + 5]
            #     mu = (y * weights).sum() / weights.sum()
            #     plt.subplot(2, 5, i+1)
            #     plt.plot(x.max(), mu, 'o')
            #     plt.plot(x, y, '.')
            #     print(list(hyp.keys())[i],'%.4g' % mu)
