import os
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_metrics_get25 import compute_mIoU
import time


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True, patience=5):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag
        self.patience = patience  

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []
        self.best_val_loss = float('inf')  
        self.epochs_without_improvement = 0  

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            device = next(model.parameters()).device
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to(device)
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"Epoch: {epoch}, loss: {loss}\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                # f.write(str(val_loss))
                f.write(f"Epoch: {epoch}, val_loss: {val_loss}\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)

        # 绘制loss图像
        self.loss_plot()

        # 早停策略
        if self.val_loss_flag and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0  
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {self.best_val_loss}")
                return True  
        return False

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        try:

            if len(self.losses) < 25:
                num = 5
            else:
                num = 15


            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')

            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2, label='smooth val loss')
        except:
            pass


        plt.grid(True)

        plt.xlabel('Epoch')

        plt.ylabel('Loss')

        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()

        plt.close("all")



class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                    miou_out_path=".temp_miou_out", eval_flag=True, period=5):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag 
        self.period = period 
        self.image_ids = [image_id.split()[0] for image_id in image_ids]

        self.mious = [0]
        self.epoches = [0]
        self.PA_recalls = [0]
        self.precisions = [0]
        self.accuracies = [0]
        self.F1_scores = [0]  
        self.Dice = [0]  
        self.HD95 = [0]
        self.error_rates = [0]  


    def get_miou_png(self, image):
        
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            pr = self.net(images)[0]
            if pr is None or pr.size == 0:
                print("Warning: The predicted image is empty.")
            
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    
    def on_epoch_end(self, epoch, UnFreeze_Epoch, model_eval):
        if epoch>=0 and (epoch % self.period == 0 or epoch == 1 or epoch == UnFreeze_Epoch-1) and self.eval_flag:
            self.net = model_eval
            
            gt_dir = os.path.join(self.dataset_path, r"masks")
            pred_dir = os.path.join(self.miou_out_path, r'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            else:
                
                shutil.rmtree(pred_dir)
                os.makedirs(pred_dir)
            print("开始获取 miou.")

            for image_id in tqdm(self.image_ids):
                
                image_path = os.path.join(self.dataset_path, "train/" + image_id + ".jpg")
                image = Image.open(image_path)
                
                with Image.open(image_path) as image:
                    image = self.get_miou_png(image)
                    image.save(os.path.join(pred_dir, image_id + ".png"))

            print("开始计算 miou.")
            _, IoUs, PA_Recall, Precision, accuracy,_,dice,temp_hd95 = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes,
                                                                    None)  
            temp_miou = np.nanmean(IoUs) * 100  
            temp_PA_Recall = np.nanmean(PA_Recall) * 100
            temp_Precision = np.nanmean(Precision) * 100
            temp_accuracy = np.nanmean(accuracy) * 100
            temp_dice = np.nanmean(dice) * 100
            # 计算 F1 Score
            if (temp_Precision + temp_PA_Recall) > 0:
                temp_F1 = 2 * (temp_Precision * temp_PA_Recall) / (temp_Precision + temp_PA_Recall)
            else:
                temp_F1 = 0 

            self.mious.append(temp_miou)  
            self.PA_recalls.append(temp_PA_Recall)
            self.precisions.append(temp_Precision)
            self.accuracies.append(temp_accuracy)
            self.F1_scores.append(temp_F1)  
            self.Dice.append(temp_dice)
            self.HD95.append(temp_hd95)
            self.epoches.append(epoch)  

            # 写入 mIoU
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, mIoU: {temp_miou}\n")

            # 写入 PA_Recall
            with open(os.path.join(self.log_dir, "epoch_PA_Recall.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, PA_Recall: {temp_PA_Recall}\n")

            # 写入 Precision
            with open(os.path.join(self.log_dir, "epoch_Precision.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, Precision: {temp_Precision}\n")

            # 写入 Accuracy
            with open(os.path.join(self.log_dir, "epoch_accuracy.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, Accuracy: {temp_accuracy}\n")
            
            # 写入 F1 Score
            with open(os.path.join(self.log_dir, "epoch_F1.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, F1: {temp_F1}\n")
                
            with open(os.path.join(self.log_dir, "dice.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, Dice: {temp_dice}\n")
            
            with open(os.path.join(self.log_dir, "HD95.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, HD95: {temp_hd95}\n")

            # 绘制 mIoU 曲线
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='Train mIoU')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla() 

            plt.close("all")
            print("Get miou done.")

            # 删除指定路径下的所有文件和文件夹
            for attempt in range(5):  
                try:
                    shutil.rmtree(pred_dir)
                    break  
                except Exception as e:
                    print(f"Attempt {attempt + 1}: Failed to remove {pred_dir}: {e}")
                    time.sleep(0.5)  

            # 检查目录是否存在
            if os.path.exists(pred_dir):
                # 遍历目录中的所有文件
                for filename in os.listdir(pred_dir):
                    file_path = os.path.join(pred_dir, filename)
                    try:
                        os.remove(file_path)  # 删除文件
                        print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")

                # 尝试删除空目录
                try:
                    os.rmdir(pred_dir)
                    print(f"Deleted directory: {pred_dir}")
                except Exception as e:
                    print(f"Failed to delete directory {pred_dir}: {e}")
        
    