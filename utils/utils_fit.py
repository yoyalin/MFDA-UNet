import os
import torch
from tqdm import tqdm
import math

from utils.utils import get_lr
from utils.utils_metrics import f_score
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen_train, gen_val, 
                  UnFreeze_Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, 
                  local_rank=0):

    total_loss      = 0
    total_f_score   = 0


    val_loss        = 0
    val_f_score     = 0


    if local_rank == 0:
        print('Start Train')

        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3,position=0)

    model_train.train()

    for iteration, batch in enumerate(gen_train):

        if iteration >= epoch_step: 
            break

        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)


        optimizer.zero_grad()
        if not fp16:

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():

                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.amp import autocast
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):

                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():

                    _f_score = f_score(outputs, labels)


            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            


        total_loss      += loss.item()

        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)


            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice

            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('验证过程已经完成')

        early_stop = loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)

        eval_callback.on_epoch_end(epoch + 1, UnFreeze_Epoch,model_train)

        print('Epoch:'+ str(epoch+1) + '/' + str(UnFreeze_Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        if math.isnan(val_loss / epoch_step_val) or math.isnan(total_loss / epoch_step):

            print(f"\033[1;31mEpoch {epoch + 1}: 检测到 val_loss 为 NaN，终止训练！\033[0m")
            return early_stop, True 

        if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
            save_model_dir = os.path.join(save_dir, 'epo_%d-loss_%.3f-val_loss_%.3f.pth'%(epoch, total_loss / epoch_step, val_loss / epoch_step_val))

            torch.save(model.state_dict(), save_model_dir)
            print(f'第{epoch+1}轮定期保存模型成功(epoch+1)')


        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            

            record_file_path = os.path.join(save_dir, "best_epoch_record.txt")
            with open(record_file_path, 'w') as f:
                f.write(f'最佳模型的Epoch: {epoch}\n')
            
            print(f'保存性能最好的模型到 best_epoch_weights.pth 成功,当前Epoch为{epoch}')
            

        # 返回早停状态
        return early_stop,False

