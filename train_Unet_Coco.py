import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything,
                         show_config, worker_init_fn, save_config)
from utils.utils_fit import fit_one_epoch

def format_time(seconds):
    if seconds < 0:
        return "00:00:00"
    td = datetime.timedelta(seconds=int(seconds))
    days = td.days
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}天 {hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    
    from nets.MFDA_UNet import Unet

    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = True
    num_classes = 21
    backbone = "efficientnet-b5"
    pretrained = True
    model_path = ""
    downsample_factor = 16
    input_shape = [256, 256]

    Init_Epoch = 0
    Freeze_Epoch = 30
    Freeze_batch_size = 24
    UnFreeze_Epoch = 50
    Unfreeze_batch_size = 16
    input_seed = input("seed= : ")
    seed = int(input_seed)

    input_freeze = input("freeze=: ")
    Freeze_Train = True if input_freeze.lower() == 'y' else False

    input_opt = input("optimizer= : ")
    optimizer_type = input_opt.strip() if input_opt.strip() else "adamw"
    
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    momentum = 0.9
    if optimizer_type == 'adamw':
        weight_decay = 0.05
    elif optimizer_type == 'adam':
        weight_decay = 0
    else:
        weight_decay = 1e-4
    
    lr_decay_type = 'cos'
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 1

    VOCdevkit_path = r'VOC2012'
    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    aux_branch = True
    num_workers = 8

    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    model = Unet(num_classes=num_classes, backbone=backbone, pretrained=pretrained, use_attention=False)
    
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device, weights_only=True)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch import amp
        scaler = amp.GradScaler('cuda')
        print('使用fp16')
    else:
        scaler = None

    model_train = model.train()

    save_model_dir = os.path.join(log_dir, "model")
    os.makedirs(save_model_dir, exist_ok=True)

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(VOCdevkit_path, r"Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, r"Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        save_config(
            log_dir, num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (total_step, wanted_step, wanted_epoch))

    if True:
        UnFreeze_flag = False
        
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = Init_lr if optimizer_type in ['adam','adamw'] else 1e-1
        lr_limit_min = Init_lr if optimizer_type in ['adam','adamw'] else 5e-4 

        Init_lr_fit = Init_lr
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        backbone_lr_scale = 0.1
        lr_backbone = Init_lr_fit * backbone_lr_scale
        lr_decoder = Init_lr_fit

        if optimizer_type == 'adamw':
            if local_rank == 0:
                print("使用 AdamW 优化器，并进行参数分组")

            no_decay_keys = []
            if hasattr(model, 'backbone'):
                if hasattr(model.backbone, 'no_weight_decay_keywords'):
                    no_decay_keys = model.backbone.no_weight_decay_keywords()
                    if local_rank == 0:
                        print(f"从 backbone '{type(model.backbone).__name__}' 获取到 no_decay_keys: {no_decay_keys}")
                else:
                    if local_rank == 0:
                        print(f"'{type(model.backbone).__name__}' 没有 'no_weight_decay_keywords' 方法。将使用通用规则。")
            else:
                if local_rank == 0:
                    print("模型没有 'backbone' 属性。将使用通用规则对所有参数进行分组。")

            decoder_params_decay = []
            decoder_params_no_decay = []
            backbone_params_decay = []
            backbone_params_no_decay = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                is_no_decay = False
                if param.dim() <= 1 or "bias" in name or "norm" in name.lower() or "bn" in name.lower():
                    is_no_decay = True

                if no_decay_keys and any(ndk in name for ndk in no_decay_keys):
                    is_no_decay = True

                if hasattr(model, 'backbone') and "backbone" in name:
                    if is_no_decay:
                        backbone_params_no_decay.append(param)
                    else:
                        backbone_params_decay.append(param)
                else:
                    if is_no_decay:
                        decoder_params_no_decay.append(param)
                    else:
                        decoder_params_decay.append(param)

            if local_rank == 0:
                print("-" * 30)
                print("参数分组详情:")
                print(f"  - Backbone (Decay): {len(backbone_params_decay)} 个 tensors")
                print(f"  - Backbone (No Decay): {len(backbone_params_no_decay)} 个 tensors")
                print(f"  - Decoder (Decay): {len(decoder_params_decay)} 个 tensors")
                print(f"  - Decoder (No Decay): {len(decoder_params_no_decay)} 个 tensors")
                print("-" * 30)

            optimizer_grouped_parameters = [
                {'params': backbone_params_decay, 'lr': lr_backbone, 'weight_decay': weight_decay},
                {'params': backbone_params_no_decay, 'lr': lr_backbone, 'weight_decay': 0.0},
                {'params': decoder_params_decay, 'lr': lr_decoder, 'weight_decay': weight_decay},
                {'params': decoder_params_no_decay, 'lr': lr_decoder, 'weight_decay': 0.0}
            ]

            optimizer_grouped_parameters = [p for p in optimizer_grouped_parameters if p['params']]

            optimizer = optim.AdamW(optimizer_grouped_parameters, betas=(momentum, 0.999))

        else:
            if local_rank == 0:
                print(f"使用 {optimizer_type} 优化器，并应用差分学习率")

            params_groups = [
                {'params': model.backbone.parameters(), 'lr': lr_backbone},
                {'params': [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], 'lr': lr_decoder}
            ]

            if optimizer_type == 'adam':
                optimizer = optim.Adam(params_groups, lr_decoder, betas=(momentum, 0.999), weight_decay=weight_decay)
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(params_groups, lr_decoder, momentum=momentum, nesterov=True, weight_decay=weight_decay)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        train_dataset = UnetDataset(train_lines, input_shape, num_classes, train=True, dataset_path=VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, train=False, dataset_path=VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen_train = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                               worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        import time
        import datetime
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        start_time_path = os.path.join(log_dir, 'configurations.txt')
        with open(start_time_path, 'a', encoding='utf-8') as f:
            f.write(f"程序开始运行时间：{current_time}\n")

        lr_history = []
        start_time = time.time()
        loss_nan = False

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 16
                lr_limit_max = Init_lr if optimizer_type in ['adam','adamw'] else 1e-1
                lr_limit_min = Init_lr if optimizer_type in ['adam','adamw'] else 5e-4 
                Init_lr_fit = Init_lr
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                model.unfreeze_backbone()

                backbone_lr_scale = 0.1
                lr_backbone = Init_lr_fit * backbone_lr_scale
                lr_decoder = Init_lr_fit

                if optimizer_type == 'adamw':
                    if local_rank == 0:
                        print("解冻后，重新配置 AdamW 优化器")
                    no_decay_keys = model.backbone.no_weight_decay_keywords()
                    decoder_params, backbone_params_decay, backbone_params_no_decay = [], [], []
                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if "backbone" in name:
                            if param.dim() <= 1 or any(ndk in name for ndk in no_decay_keys) or "bias" in name:
                                backbone_params_no_decay.append(param)
                            else:
                                backbone_params_decay.append(param)
                        else:
                            decoder_params.append(param)
                    optimizer_grouped_parameters = [
                        {'params': backbone_params_decay, 'lr': lr_backbone, 'weight_decay': weight_decay},
                        {'params': backbone_params_no_decay, 'lr': lr_backbone, 'weight_decay': 0.0},
                        {'params': decoder_params, 'lr': lr_decoder, 'weight_decay': weight_decay}
                    ]
                    optimizer = optim.AdamW(optimizer_grouped_parameters, betas=(momentum, 0.999))
                else:
                    if local_rank == 0:
                        print(f"解冻后，重新配置 {optimizer_type} 优化器")
                    params_groups = [
                        {'params': model.backbone.parameters(), 'lr': lr_backbone},
                        {'params': [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], 'lr': lr_decoder}
                    ]
                    if optimizer_type == 'adam':
                        optimizer = optim.Adam(params_groups, lr_decoder, betas=(momentum, 0.999), weight_decay=weight_decay)
                    elif optimizer_type == 'sgd':
                        optimizer = optim.SGD(params_groups, lr_decoder, momentum=momentum, nesterov=True, weight_decay=weight_decay)

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen_train = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                       drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                                       worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            early_stop, loss_nan = fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                                 epoch_step, epoch_step_val,
                                                 gen_train, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_model_dir, local_rank)

            lr_path = os.path.join(log_dir, 'lr_history.txt')
            with open(lr_path, 'w') as f:
                for lr in lr_history:
                    f.write(f"{lr}\n")

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"当前梯度范数: {total_norm:.4f}")

            Gradient_norm_log_path = os.path.join(log_dir, "Gradient_norm_log.txt")
            with open(Gradient_norm_log_path, 'a') as f:
                f.write(f"\nEpoch {epoch} Gradient_norm = {total_norm:.4f}")

            if local_rank == 0 and not loss_nan:
                elapsed_time = time.time() - start_time
                completed_epochs = epoch - Init_Epoch + 1
                if completed_epochs > 0:
                    avg_time_per_epoch = elapsed_time / completed_epochs
                else:
                    avg_time_per_epoch = 0
                remaining_epochs = UnFreeze_Epoch - (epoch + 1)
                eta_seconds = avg_time_per_epoch * remaining_epochs
                eta_formatted = format_time(eta_seconds)
                elapsed_formatted = format_time(elapsed_time)
                print(f"总耗时: {elapsed_formatted} | 预计剩余时间 (ETA): {eta_formatted}")
                print("-" * 60)
            
            if total_norm > 1e2:
                print("警告：潜在梯度爆炸风险.")
                with open(Gradient_norm_log_path, 'a') as f:
                    f.write(f"潜在梯度爆炸风险\n")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            elif total_norm < 1e-6:
                print("警告：梯度消失风险.")
                with open(Gradient_norm_log_path, 'a') as f:
                    f.write(f"梯度消失风险\n")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 10
            elif total_norm > 1e5:
                print("错误:梯度完全爆炸.")
                with open(Gradient_norm_log_path, 'a') as f:
                    f.write(f"错误:梯度完全爆炸\n")

            if distributed:
                dist.barrier()

            if loss_nan:
                with open(start_time_path, 'a') as f:
                    f.write(f"loss = nan !!!")
                break

        if local_rank == 0:
            loss_history.writer.close()

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        end_times = f'{int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds'
        print(end_times)

