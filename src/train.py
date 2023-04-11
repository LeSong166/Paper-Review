from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import glob
import itertools
import logging
import os,sys
import random
import shutil
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from scipy.stats.stats import pearsonr
from math import sqrt
import pandas as pd
from torch.autograd import Variable

from models import *



   

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='region785')
ap.add_argument('--sim_mat', type=str, default='region-adj')
ap.add_argument('--n_layer', type=int, default=1) 
ap.add_argument('--n_hidden', type=int, default=20) 
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--epochs', type=int, default=1500)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--weight_decay', type=float, default=5e-4)
ap.add_argument('--dropout', type=float, default=0.2)
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--check_point', type=int, default=1)
ap.add_argument('--shuffle', action='store_true', default=False)
ap.add_argument('--train', type=float, default=.5)
ap.add_argument('--val', type=float, default=.2)
ap.add_argument('--test', type=float, default=.3)
ap.add_argument('--mylog', action='store_false', default=True)
ap.add_argument('--cuda', action='store_true', default=False)
ap.add_argument('--window', type=int, default=20)
ap.add_argument('--horizon', type=int, default=5)
ap.add_argument('--save_dir', type=str,  default='save')
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--lamda', type=float, default=0.01)
ap.add_argument('--patience', type=int, default=100)
ap.add_argument('--k', type=int, default=8)
ap.add_argument('--hidR', type=int, default=64)
ap.add_argument('--hidA', type=int, default=64)
ap.add_argument('--hidP', type=int, default=1)
ap.add_argument('--hw', type=int, default=0)
ap.add_argument('--extra', type=str, default='')
ap.add_argument('--label', type=str, default='')
ap.add_argument('--pcc', type=str, default='')
ap.add_argument('--n', type=int, default=2)
ap.add_argument('--res', type=int, default=0)
ap.add_argument('--s', type=int, default=2)
ap.add_argument('--result', type=int, default=0)
ap.add_argument('--ablation', type=str, default=None)
ap.add_argument('--eval', type=str, default='')
ap.add_argument('--record', type=str, default='')
ap.add_argument('--model', type=str, default='EpiGNN')
args = ap.parse_args() 

# Print the parameters
from tabulate import tabulate

print("-------- Parameters --------")
#params = [[arg, getattr(args, arg)] for arg in vars(args)]
#print(tabulate(params, headers=["Parameter", "Value"], tablefmt="grid"))
print(args)
print("------------------------------")



# Set the CUDA environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Set the random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Check if CUDA is available and enable it if it is
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
#logger.info('CUDA enabled: %s', args.cuda)

# Set up tensorboard logging
time_token = str(time.time()).split('.')[0]
log_token = f'{args.model}.{args.dataset}.w-{args.window}.h-{args.horizon}'

if args.mylog:
    tensorboard_log_dir = f'tensorboard/{log_token}'
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    try:
        shutil.rmtree(tensorboard_log_dir)
    except PermissionError as e:
        err_file_path = str(e).split("\'",2)[1]
        if os.path.exists(err_file_path):
            os.chmod(err_file_path, stat.S_IWUSR)

class DataBasicLoader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # 20
        self.h = args.horizon # 1
        self.d = 0
        self.add_his_day = False
        self.save_dir = args.save_dir
        self.rawdat = np.loadtxt(open("data/{}.txt".format(args.dataset)), delimiter=',')
        print('data shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)
        if args.extra:
            self.load_external(args)
        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape # n_sample, n_group
        print(self.n, self.m)

        self.scale = np.ones(self.m)

        self._pre_train(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        print('size of train/val/test sets',len(self.train[0]),len(self.val[0]),len(self.test[0]))
    
    def load_label_file(self, filename):
        labelfile = pd.read_csv("data/"+filename+".csv", header=None)
        labelLen = len(labelfile)
        label = dict()
        for i in range(labelLen):
            label[labelfile.iloc[i,0]]=labelfile.iloc[i,1]
        return label, labelLen

    def load_external(self, args):
        label, label_num = self.load_label_file(args.label)
        files = os.listdir("data/{}".format(args.extra))
        filesLen = len(files)
        extra_adj_list = []
        for i in range(filesLen):
            snapshot = pd.read_csv("data/"+args.extra+"/"+files[i], header=None)
            snapshot_len = len(snapshot)
            extra_adj = np.zeros((label_num,label_num))
            for j in range(snapshot_len):
                extra_adj[label[snapshot.iloc[j,0]],label[snapshot.iloc[j,1]]] = snapshot.iloc[j,2]
            #print(extra_adj)
            extra_adj_list.append(extra_adj)
        extra_adj = torch.Tensor(np.array(extra_adj_list))
        print('external information', extra_adj.shape)
        self.external = Variable(extra_adj)
        if args.cuda:
            self.external = extra_adj.cuda()

    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(open("data/{}.txt".format(args.sim_mat)), delimiter=','))
        self.orig_adj = self.adj
        self.degree_adj = torch.sum(self.orig_adj, dim=-1)
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()
            self.orig_adj = self.orig_adj.cuda()
            self.degree_adj = self.degree_adj.cuda()

    def _pre_train(self, train, valid, test):
        self.train_set = train_set = range(self.P+self.h-1, train)
        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n)
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)
        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy() #199, 47
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        #np.save('%s/maxvalue.npy' % (self.save_dir), self.max)
        #np.save('%s/minvalue.npy' % (self.save_dir), self.min)
        self.peak_thold = np.mean(train_mx, 0)
        self.dat  = (self.rawdat  - self.min ) / (self.max  - self.min + 1e-12)
        print(self.dat.shape)
         
    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if (train == valid):
            self.val = self.test
 
    def _batchify(self, idx_set, horizon, useraw=False): ###tonights work

        n = len(idx_set)
        Y = torch.zeros((n, self.m))
        if self.add_his_day and not useraw:
            X = torch.zeros((n, self.P+1, self.m))
        else:
            X = torch.zeros((n, self.P, self.m))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P

            if useraw: # for narmalization
                X[i,:self.P,:] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i,:] = torch.from_numpy(self.rawdat[idx_set[i], :])
            else:
                his_window = self.dat[start:end, :]
                if self.add_his_day:
                    if idx_set[i] > 51 : # at least 52
                        his_day = self.dat[idx_set[i]-52:idx_set[i]-51, :] #
                    else: # no history day data
                        his_day = np.zeros((1,self.m))

                    his_window = np.concatenate([his_day,his_window])
                    # print(his_window.shape,his_day.shape,idx_set[i],idx_set[i]-52,idx_set[i]-51)
                    X[i,:self.P+1,:] = torch.from_numpy(his_window) # size (window+1, m)
                else:
                    X[i,:self.P,:] = torch.from_numpy(his_window) # size (window, m)
                Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    # original
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:]
            Y = targets[excerpt,:]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)
            #print('x shape', X.shape) # batch_size window_size region_num
            #print('y shape', Y.shape)

            data = [model_inputs, Variable(Y), index]
            yield data
            start_idx += batch_size

# Load and preprocess the data
data_loader = DataBasicLoader(args)

# Select the appropriate model based on the command line arguments
if args.ablation is None:
    model = EpiGNN(args, data_loader)
else:
    if args.ablation == 'woglobal':
        model = WOGlobal(args, data_loader)
    elif args.ablation == 'wolocal':
        model = WOLocal(args, data_loader)
    elif args.ablation == 'woragl':
        model = WORAGL(args, data_loader)
    elif args.ablation == 'baseline':
        model = baseline(args, data_loader)
    else:
        raise LookupError(f"Unknown ablation model: {args.ablation}")

logger.info('Model: %s', model)
if args.cuda:
    model.cuda()

# Initialize the optimizer and calculate the number of parameters in the model
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('# of model parameters: %s', pytorch_total_params)

def evaluate(data_loader, data, tag='val', show=0):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    x_value_mx = []

    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        output, _ = model(X, index)

        # Calculate the loss and accumulate performance metrics
        loss_train = F.mse_loss(output, Y)
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m)

        # Save the predicted and true values for later analysis
        x_value_mx.append(X.data.cpu())
        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    # Concatenate the predicted and true values into matrices
    x_value_mx = torch.cat(x_value_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx)
    x_value_states = x_value_mx.numpy() * (data_loader.max - data_loader.min) * 1.0 + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) * 1.0 + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) * 1.0 + data_loader.min
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)
    if not args.pcc:
        pcc_tmp = []
        for k in range(data_loader.m):
            pcc_tmp.append(pearsonr(y_true_states[:, k], y_pred_states[:, k])[0])
        pcc_states = np.mean(np.array(pcc_tmp))
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # Convert predicted and true values to 1D arrays and calculate performance metrics
    y_true = np.reshape(y_true_states, (-1))
    y_pred = np.reshape(y_pred_states, (-1))
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    if show == 1:
        print('x value:', x_value_states)
        print('ground truth:', y_true.shape)
        print('predicted values:', y_pred.shape)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 0.00001)))
    mape /= 10000000
    if not args.pcc:
        pcc = pearsonr(y_true, y_pred)[0]
    else:
        pcc = 1
        pcc_states = 1
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)

    # Save

    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples), mae,std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae

# Training function
def train(data_loader, data):
    model.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        optimizer.zero_grad()
        output, _ = model(X, index)
        if Y.size(0) == 1:
            Y = Y.view(-1)
        loss_train = F.mse_loss(output, Y)
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        n_samples += (output.size(0) * data_loader.m)
    return float(total_loss / n_samples)

# Training loop

import matplotlib.pyplot as plt

try:
    print('Begin training')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_losses, val_losses = [], []
    best_val = float("inf")
    bad_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train)
        val_loss, *_ = evaluate(data_loader, data_loader.val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #print(f"Epoch {epoch:3d} | Time: {time.time() - epoch_start_time:5.2f}s | Train loss: {train_loss:5.8f} | Val loss: {val_loss:5.8f}")

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = f"{args.save_dir}/{log_token}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Best validation epoch so far: {epoch} at {time.ctime()}")
            print(f"Epoch {epoch:3d} | Time: {time.time() - epoch_start_time:5.2f}s | Train loss: {train_loss:5.8f} | Val loss: {val_loss:5.8f}")
            test_metrics = evaluate(data_loader, data_loader.test, tag="test")
            mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = test_metrics[1:]

            print(f"TEST MAE: {mae:5.4f} std: {std_mae:5.4f} RMSE: {rmse:5.4f} RMSEs: {rmse_states:5.4f} PCC: {pcc:5.4f} PCCs: {pcc_states:5.4f} MAPE: {mape:5.4f} R2: {r2:5.4f} R2s: {r2_states:5.4f} Var: {var:5.4f} Vars: {var_states:5.4f} Peak: {peak_mae:5.4f}")

        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    # Plot the training and validation losses across epochs
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()

except KeyboardInterrupt:
    print('-' * 89)
    print(f"Exiting from training early, epoch {epoch}")

# Load the best saved model.
model_path = f"{args.save_dir}/{log_token}.pt"
model.load_state_dict(torch.load(model_path))

# Load the best saved model.
model_path = f"{args.save_dir}/{log_token}.pt"
with open(model_path, "rb") as f:
    model.load_state_dict(torch.load(f))

test_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.test, tag="test", show=args.result)
print("----------Final evaluation----------")

# Record test results.
if args.record != "":
    with open("result/result.txt", "a", encoding="utf-8") as f:
        f.write(f"Model: EpiGNN, dataset: {args.dataset}, window: {args.window}, horizon: {args.horizon}, seed: {args.seed}, MAE: {mae:5.4f}, RMSE: {rmse:5.4f}, PCC: {pcc:5.4f}, lr: {args.lr}, GNN_number: {args.n}, filter_num: {args.k}, res: {args.res}, hidR: {args.hidR}, hidP: {args.hidP}, hidA: {args.hidA}, dropout: {args.dropout}\n")

# Print evaluation results in table format.
print(f"{'-'*95}")
print(f"|{'Metric':^20s}|{'Overall':^20s}|{'States/Places Adjusted':^20s}|")
print(f"{'-'*95}")
print(f"|{'MAE':^20s}|{mae:20.4f}|{std_mae:20.4f}|")
print(f"|{'RMSE':^20s}|{rmse:20.4f}|{rmse_states:20.4f}|")
print(f"|{'PCC':^20s}|{pcc:20.4f}|{pcc_states:20.4f}|")
print(f"|{'MAPE':^20s}|{mape:20.4f}|{mape:20.4f}|")
print(f"|{'R2':^20s}|{r2:20.4f}|{r2_states:20.4f}|")
print(f"|{'Var':^20s}|{var:20.4f}|{var_states:20.4f}|")
print(f"|{'peak_mae':^20s}|{peak_mae:20.4f}|{peak_mae:20.4f}|")
print(f"{'-'*95}")

# test the trained model.
if args.eval != "":
    testdata = np.loadtxt(open(f"data/{args.eval}.txt"), delimiter=",")
    testdata = (testdata - data_loader.min) / (data_loader.max - data_loader.min)
    testdata = torch.Tensor(testdata)
    testdata = testdata.unsqueeze(0)
    testdata = Variable(testdata)

    if args.cuda:
        testdata = testdata.cuda()

    model.eval()
    with torch.no_grad():
        out_data, addinfo = model(testdata, None, isEval=True)
        out_data = out_data.cpu().numpy() * (data_loader.max - data_loader.min) * 1.0 + data_loader.min
        out_data = out_data.squeeze(0)

    # record
    out_data = out_data.tolist()
    adjacent = addinfo[0].squeeze(0).cpu().numpy()
    adjacent = adjacent.tolist()
    attn = addinfo[1].squeeze(0).cpu().numpy()
    attn = np.around(attn, 2)
    attn = attn.tolist()

    with open(f"save/{args.eval}result.txt", "a") as f:
        f.write(f"\nWindow: {args.window}, Horizon: {args.horizon}\n")
        f.write(str(out_data))
        f.write("\n")
    with open(f"save/{args.eval}adjacent.txt", "a") as f:
        f.write(f"\nWindow: {args.window}, Horizon: {args.horizon}\n")
        f.write(str(adjacent))
        f.write("\n")
    with open("save/{}.txt".format(args.eval+"attn"), "a") as f:
        f.write(f"\nWindow: {args.window}, Horizon: {args.horizon}\n")
        f.write(str(attn))
        f.write('\n')
        
