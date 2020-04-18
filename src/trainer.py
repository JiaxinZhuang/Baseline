"""Trainer.

    Jiaxin Zhuang, lincolnz9511@gmail.com
"""
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from PIL import Image

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

from utils.function import init_logging, init_environment, get_lr, timethis
from utils.metrics import average_precision
import config
import dataset
import model


@timethis
def main():
    """Main.
    """
    # hyperparameter
    configs = config.Config()
    configs_dict = configs.get_config()
    exp = configs_dict["experiment_index"]
    cuda_id = configs_dict["cuda"]
    num_workers = configs_dict["num_workers"]
    seed = configs_dict["seed"]
    n_epochs = configs_dict["n_epochs"]
    log_dir = configs_dict["log_dir"]
    model_dir = configs_dict["model_dir"]
    batch_size = configs_dict["batch_size"]
    learning_rate = configs_dict["learning_rate"]
    eval_frequency = configs_dict["eval_frequency"]
    resume = configs_dict["resume"]
    optimizer = configs_dict["optimizer"]
    input_size = configs_dict["input_size"]
    re_size = configs_dict["re_size"]
    backbone = configs_dict["backbone"]
    dataset_name = configs_dict["dataset"]
    test_input_size = configs_dict["test_input_size"]
    bilinear = configs_dict["bilinear"]
    data_dir = configs_dict["data_dir"]

    # init environment and log
    init_environment(seed=seed, cuda_id=cuda_id)
    _print = init_logging(log_dir, exp).info
    configs.print_config(_print)
    tf_log = os.path.join(log_dir, exp)
    writer = SummaryWriter(log_dir=tf_log)
    try:
        os.mkdir(os.path.join(model_dir, exp))
    except FileExistsError:
        _print("{} has been created.".format(os.path.join(model_dir, exp)))

    # dataset
    _print(">> Dataset:{} - Input size: {}".format(dataset_name, input_size))
    if dataset_name == "CUB":
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.CenterCrop(test_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = dataset.CUB(root=data_dir, train=True,
                               transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
        valset = dataset.CUB(root=data_dir, train=False,
                             transform=test_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers)
    elif dataset_name == "cifar10":
        num_classes = 10
        mean = [0.489, 0.478, 0.446]
        std = [0.249, 0.245, 0.263]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.CenterCrop(test_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = CIFAR10(root=data_dir,
                           train=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
        valset = CIFAR10(root=data_dir,
                         train=False, transform=test_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers)
    elif dataset_name == "cifar100":
        num_classes = 100
        mean = [0.508, 0.478, 0.434]
        std = [0.264, 0.252, 0.275]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.CenterCrop(test_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = CIFAR100(root=data_dir,
                            train=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
        valset = CIFAR100(root=data_dir,
                          train=False, transform=test_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers)
    elif dataset_name == "SVHN":
        num_classes = 10
        mean = [0.444, 0.450, 0.478]
        std = [0.196, 0.198, 0.194]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.CenterCrop(test_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        data_dir = os.path.join(data_dir, "SVHN")
        trainset = SVHN(root=data_dir, split="train",
                        transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
        valset = SVHN(root=data_dir,
                      split="test", transform=test_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers)
    elif dataset_name == "mnist":
        num_classes = 10
        mean = [0.127]
        std = [0.304]
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((re_size, re_size),
                              interpolation=Image.BILINEAR),
            transforms.CenterCrop(test_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = MNIST(root=data_dir, train=True,
                         transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
        valset = MNIST(root=data_dir, train=False,
                       transform=test_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers)

    else:
        _print("Need dataset")
        sys.exit(-1)

    input_channel = len(mean)

    # define model
    net = model.Network(backbone=backbone, input_channel=input_channel,
                        num_classes=num_classes, bilinear=bilinear,
                        _print=_print)
    net = net.cuda()
    net.print_model()

    # Optimizer & loss function
    if optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(),
                              lr=learning_rate,
                              momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.1, patience=10,
                        verbose=True, threshold=1e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = None
    # Loss
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if resume:
        _print("Resume from model at epoch {}".format(resume))
        resume_path = os.path.join(model_dir, str(exp), str(resume))
        ckpt = torch.load(resume_path)
        net.load_state_dict(ckpt)
        start_epoch = resume + 1

    # whole Train process
    for epoch in range(start_epoch, n_epochs):
        net.train()
        losses = []
        for _, (data, targets) in enumerate(trainloader):
            data, targets = data.cuda(), targets.cuda()
            optimizer.zero_grad()
            embeddings = net(data)
            loss = criterion(embeddings, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            del data, targets, embeddings

        train_avg_loss = np.mean(losses)
        if scheduler:
            scheduler.step(train_avg_loss)

        _print("Epoch:{} - train loss: {:.3f}".format(epoch, train_avg_loss))
        writer.add_scalar("Lr", get_lr(optimizer), epoch)
        writer.add_scalar("Loss/train/", train_avg_loss, epoch)

        if epoch % eval_frequency == 0:
            with torch.no_grad():
                net.eval()
                y_pred = []
                y_true = []
                for _, (data, targets) in enumerate(trainloader):
                    data = data.cuda()
                    embeddings = net(data)
                    pred = torch.argmax(embeddings, dim=1).cpu().data.numpy().\
                        tolist()
                    true = (targets.cpu().data.numpy()).tolist()
                    y_pred.extend(pred)
                    y_true.extend(true)
                del data, embeddings
                acc = accuracy_score(y_true, y_pred)
                mca = balanced_accuracy_score(y_true, y_pred)
                mcp = average_precision(y_true, y_pred)
                _print("Epoch:{} - train acc:{:.4f}, mca:{:.4f}, mcp:{:.4f}".
                       format(epoch, acc, mca, mcp))
                writer.add_scalar("Metric/Acc", acc, epoch)
                writer.add_scalar("Metric/Mca", mca, epoch)
                writer.add_scalar("Metric/Mcp", mcp, epoch)

                y_pred = []
                y_true = []
                losses = []
                for _, (data, targets) in enumerate(valloader):
                    data, targets = data.cuda(), targets.cuda()
                    embeddings = net(data)
                    loss = criterion(embeddings, targets)
                    losses.append(loss.item())
                    pred = torch.argmax(embeddings, dim=1).cpu().data.numpy().\
                        tolist()
                    true = (targets.cpu().data.numpy()).tolist()
                    y_pred.extend(pred)
                    y_true.extend(true)
                del embeddings, data
                acc = accuracy_score(y_true, y_pred)
                mca = balanced_accuracy_score(y_true, y_pred)
                mcp = average_precision(y_true, y_pred)
                val_avg_loss = np.mean(losses)
                writer.add_scalar("Metric/Val_Acc", acc, epoch)
                writer.add_scalar("Metric/Val_Mca", mca, epoch)
                writer.add_scalar("Metric/Val_Mcp", mcp, epoch)
                writer.add_scalar("Loss/val/", val_avg_loss, epoch)
                _print("Epoch:{} - val acc: {:.4f}, mca: {:.4f}, mcp: {:.4f}".
                       format(epoch, acc, mca, mcp))
                # save model
                model_path = os.path.join(model_dir, str(exp), str(epoch))
                _print("Save model in {}".format(model_path))
                net_state_dict = net.state_dict()
                torch.save(net_state_dict, model_path)

    print("finishing training")


if __name__ == "__main__":
    main()
