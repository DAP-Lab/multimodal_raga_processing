# In[1]:


import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import gc
from tqdm import tqdm  # Import tqdm
import datetime
import optuna
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from torch.autograd import Function
import configparser
import sys

# In[2]:


pth_files_dir='../models'
os.makedirs(pth_files_dir,exist_ok=True)

if len(sys.argv) != 2:
    raise ValueError("Usage: python script.py [split_1 | split_2 | split_3]")

split_arg = sys.argv[1].upper()
section = split_arg if split_arg in ['SPLIT_1', 'SPLIT_2', 'SPLIT_3'] else None

if section is None:
    raise ValueError("Invalid argument. Choose from: split_1, split_2, split_3")

config = configparser.ConfigParser()
config.read('config.ini')

# Load paths from config
train_X_path = config[section]['train_X_audio']
train_y_path = config[section]['train_y']
train_y_singer_path = config[section]['train_y_singer']
val_X_path = config[section]['val_X_audio']
val_y_path = config[section]['val_y']
val_y_singer_path = config[section]['val_y_singer']
db_name = config[section]['db_name_audio']
study_name = config[section]['study_name_audio']



train_X=np.load(train_X_path,allow_pickle=True)
train_y=np.load(train_y_path)
train_y_singer=np.load(train_y_singer_path)
val_X=np.load(val_X_path,allow_pickle=True)
val_y=np.load(val_y_path)
val_y_singer=np.load(val_y_singer_path)


print (train_X.shape,train_y.shape,val_X.shape,val_y.shape)


# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.data import TensorDataset, DataLoader


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna


# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from torch.autograd import Function

confusion_matrix_dict={}
trial_results={}

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, gamma):
        super(GradientReversal, self).__init__()
        self.gamma = gamma
        self.epoch = 0

    def forward(self, x):
        lambda_ = 2 / (1 + torch.exp(-self.gamma * self.epoch)) - 1
        return GradientReversalFunction.apply(x, lambda_)

    def update_epoch(self, epoch):
        self.epoch = epoch


   
class Custom1DNetwork(nn.Module):
    def __init__(self, in_features, n11, n21, n31, n32, n41, n42, n43, p, kernel_size=3, stride=1, padding='same', dropout_prob=0.5,pool_type='max'):
        super(Custom1DNetwork, self).__init__()

        # Convolutional layers
        self.conv11 = nn.Conv1d(in_channels=in_features, out_channels=n11, kernel_size=1, stride=stride, padding=padding)
        self.batch_norm11 = nn.BatchNorm1d(num_features=n11)
        
        self.conv21 = nn.Conv1d(in_channels=in_features, out_channels=n21, kernel_size=1, stride=stride, padding=padding)
        self.batch_norm21 = nn.BatchNorm1d(num_features=n21)


        self.conv31 = nn.Conv1d(in_channels=in_features, out_channels=n31, kernel_size=1, stride=1, padding=padding)
        self.batch_norm31=nn.BatchNorm1d(num_features=n31)
        self.conv32 = nn.Conv1d(in_channels=n31, out_channels=n32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm32=nn.BatchNorm1d(num_features=n32)

        
        self.conv41 = nn.Conv1d(in_channels=in_features, out_channels=n41, kernel_size=1, stride=1, padding=padding)
        self.batch_norm41=nn.BatchNorm1d(num_features=n41)

        self.conv42 = nn.Conv1d(in_channels=n41, out_channels=n42, kernel_size=1, stride=1, padding=padding)
        self.batch_norm42=nn.BatchNorm1d(num_features=n42)

        self.conv43 = nn.Conv1d(in_channels=n42, out_channels=n43, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm43=nn.BatchNorm1d(num_features=n43)

        
        # Pooling layer
        if pool_type=='max':
            self.pool = nn.MaxPool1d(kernel_size=3,stride=1,padding=1)
        else:
            self.pool = nn.AvgPool1d(kernel_size=3,stride=1,padding=1)

        
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Branch 1
        x1 = self.pool(x)
        x1 = F.relu(self.batch_norm11(self.conv11(x)))
        
        # Branch 2
        x2 = F.relu(self.batch_norm21(self.conv21(x)))

        # Branch 3
        x3 = F.relu(self.batch_norm31(self.conv31(x)))
        x3 = F.relu(self.batch_norm32(self.conv32(x3)))
        
        # Branch 4
        x4 = F.relu(self.batch_norm41(self.conv41(x)))
        x4 = F.relu(self.batch_norm42(self.conv42(x4)))
        x4 = F.relu(self.batch_norm43(self.conv43(x4)))

        # Match sequence lengths of all branches
        # min_len = min(x1.shape[-1], x2.shape[-1], x3.shape[-1],x4.shape[-1])
        # x1 = x1[:, :, :min_len]
        # x2 = x2[:, :, :min_len]
        # x3 = x3[:, :, :min_len]
        # x4 = x4[:, :, :min_len]
        
        # Concatenate along channel dimension
        x_concat = torch.cat((x1, x2, x3,x4), dim=1)
        
        # Apply dropout
        x_out = self.dropout(x_concat)
        
        return x_out


class MyModel(nn.Module):
    def __init__(self,num_filters,kernel_size,dropout_rate,n11,n21,n31,n32,n41,n42,n43\
                 ,num_convs,num_filters1,num_dense,num_dense_nodes,kernel_size1,pool_type,flatten_type,gamma,\
                    num_dense_singer,num_dense_singer_nodes):
        super(MyModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=num_filters, kernel_size=kernel_size, stride=1, padding='same')
        self.batch_norm = nn.BatchNorm1d(num_features=num_filters)
        self.activation=nn.ReLU()
        self.num_convs=num_convs
        self.num_dense=num_dense
        self.flatten_type=flatten_type

        if num_convs==2:
            self.conv1d_1 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters1, kernel_size=kernel_size1, stride=1, padding='same')
            self.batch_norm_1 = nn.BatchNorm1d(num_features=num_filters1)
            #
            self.inception = Custom1DNetwork(in_features=num_filters1,n11=n11,n21=n21,n31=n31,n32=n32,n41=n41,n42=n42,n43=n43,p=2,pool_type=pool_type)
            input_shape = (1, 2, 1200)
            dummy_input = torch.randn(input_shape)  # Example input
            x = self._calculate_output_shape(dummy_input)
        else:
            #
            self.inception = Custom1DNetwork(in_features=num_filters,n11=n11,n21=n21,n31=n31,n32=n32,n41=n41,n42=n42,n43=n43,p=2)
            input_shape = (1, 2, 1200)
            dummy_input = torch.randn(input_shape)  # Example input
            x = self._calculate_output_shape(dummy_input)
        # print (x.shape)


        self.dropout = nn.Dropout(p=dropout_rate)
        if pool_type=='max':
            self.pool=nn.MaxPool1d(kernel_size=x.shape[-1])
        else:            
            self.pool = nn.AvgPool1d(kernel_size=x.shape[-1])

        
        if flatten_type=='gap':
            self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
            flatten_shape=x.shape[1]
        elif flatten_type=='gmp':
            self.global_max_pool=nn.AdaptiveMaxPool1d(output_size=1)
            flatten_shape=x.shape[1]
        else:
            self.flatten = nn.Flatten()
            flatten_shape=x.shape[1]*x.shape[2]


        self.dropout_1 = nn.Dropout(p=dropout_rate)

        if num_dense==1:
            self.dense_hidden=nn.Linear(flatten_shape,num_dense_nodes)
            self.dense_activation=nn.ReLU()
            self.dropout_2 = nn.Dropout(p=dropout_rate)
            self.dense=nn.Linear(num_dense_nodes,9)
        else:
            self.dense = nn.Linear(flatten_shape, 9)

        self.gamma=gamma
        self.gr = GradientReversal(gamma)
        self.num_dense_singer_nodes=num_dense_singer_nodes
        # Dense layer and softmax output
        if num_dense_singer_nodes==1:
            self.dense_singer = nn.Linear(flatten_shape, num_dense_singer_nodes)  # Adjust input size if needed
            self.singer_output = nn.Linear(num_dense_singer_nodes, 11)  # 11 output classes
        else:
            self.singer_output = nn.Linear(flatten_shape, 11)  # 11 output classes

    def update_epoch(self, epoch):
        self.gr.update_epoch(epoch)

    def _calculate_output_shape(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        if self.num_convs==2:
            x = self.conv1d_1(x)
            x = self.batch_norm_1(x)
            x = self.activation(x)
        
        x = self.inception(x)
        return x

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        if self.num_convs==2:
            x = self.conv1d_1(x)
            x = self.batch_norm_1(x)
            x = self.activation(x)
        
        x = self.inception(x)
        # print (x.shape)
        x = self.dropout(x)
        # x = self.pool(x)
        if self.flatten_type=='flatten':
            x = self.flatten(x)
        elif self.flatten_type=='gap':
            x = self.global_avg_pool(x)
            x = x.squeeze(-1) 
        else:
            x = self.global_max_pool(x)
            x = x.squeeze(-1)

        x = self.dropout_1(x)
        flatten_output=x

        # print (x.shape)
        if self.num_dense==1:
            x=self.dense_hidden(x)
            x=self.dense_activation(x)
            x=self.dropout_2(x)
        x = self.dense(x)


        if self.num_dense_singer_nodes==1:
            x1=self.dense_singer(flatten_output)
            x=self.dense_activation(x)
            x=self.dropout_2(x)
            x1=self.singer_output(x1)

        x1=self.singer_output(flatten_output)
        

        return x,x1

def objective(trial):
    # Hyperparameters
    print ("Started running trial ",trial.number)
    D = 2

    num_filters=trial.suggest_int('num_filters',2,256)
    num_filters1=trial.suggest_int('num_filters1',2,256)
    num_convs=trial.suggest_categorical('num_convs',[1,2])
    kernel_size=trial.suggest_categorical('kernel_size',[3,5,7])
    kernel_size1=trial.suggest_categorical('kernel_size1',[3,5,7])
    dropout_rate=trial.suggest_uniform('dropout_rate',0,0.5)
    n11=trial.suggest_int('n11',2,256)
    n21=trial.suggest_int('n21',2,256)
    n31=trial.suggest_int('n31',2,256)
    n32=trial.suggest_int('n32',2,256)
    n41=trial.suggest_int('n41',2,256)
    n42=trial.suggest_int('n42',2,256)
    n43=trial.suggest_int('n43',2,256)
    pool_type=trial.suggest_categorical('pool_type',['max','avg'])
    flatten_type=trial.suggest_categorical('flatten_type',['flatten','gap','gmp'])
    num_dense=trial.suggest_categorical('num_dense',[1,2])
    num_dense_nodes=trial.suggest_int('num_dense_nodes',2,256)
    gamma=trial.suggest_uniform('gamma',0,1)
    num_dense_singer=trial.suggest_categorical('num_dense_singer',[0,1])
    num_dense_singer_nodes=trial.suggest_int('num_dense_singer_nodes',2,256)

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1)
    batch_size = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyModel(num_filters=num_filters,n11=n11,n21=n21,n31=n31,n32=n32,n41=n41,n42=n42,n43=n43\
                    ,kernel_size=kernel_size,dropout_rate=dropout_rate,num_convs=num_convs,num_filters1=num_filters1\
                        ,num_dense=num_dense,num_dense_nodes=num_dense_nodes,kernel_size1=kernel_size1,pool_type=pool_type\
                            ,flatten_type=flatten_type,num_dense_singer=num_dense_singer,num_dense_singer_nodes=num_dense_singer_nodes,gamma=gamma)
    model=model.to(device)


    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    # beta1 = trial.suggest_float("beta1", 0.8, 0.99)
    # beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    # # eps = trial.suggest_loguniform("eps", 1e-10, 1e-7)
    # weight_decay = trial.suggest_float("weight_decay", 0, 1e-3)
    # # amsgrad = trial.suggest_categorical("amsgrad", [True, False])

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=lr,
    #     betas=(beta1, beta2),
    #     # eps=eps,
    #     weight_decay=weight_decay
    #     # amsgrad=amsgrad
    # )

    # Suggest optimizer type
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])

    # Common learning rate
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    # Define optimizer-specific hyperparameters
    if optimizer_name == "Adam":
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", 0.9, 0.999)
        eps = trial.suggest_loguniform("eps", 1e-10, 1e-7)
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-3)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

    elif optimizer_name == "AdamW":
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", 0.9, 0.999)
        eps = trial.suggest_loguniform("eps", 1e-10, 1e-7)
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-2)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )

    elif optimizer_name == "RMSprop":
        alpha = trial.suggest_float("alpha", 0.8, 0.99)  # Smoothing constant
        eps = trial.suggest_loguniform("eps", 1e-10, 1e-7)
        momentum = trial.suggest_float("momentum", 0, 0.9)
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-3)
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay
        )

    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0, 0.9)
        weight_decay = trial.suggest_float("weight_decay", 0, 1e-3)
        nesterov = trial.suggest_categorical("nesterov", [True, False])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )

    criterion = torch.nn.CrossEntropyLoss()
    criterion_singer = torch.nn.CrossEntropyLoss()

    # Convert numpy arrays to PyTorch tensors
    train_X_tensor = torch.tensor(train_X, dtype=torch.float32).transpose(1, 2)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    val_X_tensor = torch.tensor(val_X, dtype=torch.float32).transpose(1, 2)
    val_y_tensor = torch.tensor(val_y, dtype=torch.long)
    train_y_tensor_singer=torch.tensor(train_y_singer, dtype=torch.long)
    val_y_tensor_singer=torch.tensor(val_y_singer, dtype=torch.long)


    # Create DataLoaders for both training and validation sets
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor,train_y_tensor_singer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_X_tensor, val_y_tensor,val_y_tensor_singer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_accuracy = 0
    epochs_without_improvement = 0

    loss_weight1=trial.suggest_uniform('loss_weight1',0,1)
    loss_weight2=trial.suggest_uniform('loss_weight2',0,1)
    
    num_epochs = 300  # Set the number of epochs as needed
    for epoch in tqdm(range(num_epochs),total=num_epochs, desc="Epochs"):
        model.update_epoch(epoch)
        model.train()
        for train_data, train_labels, train_labels_singer in train_loader:
            train_data, train_labels, train_labels_singer = train_data.to(device), train_labels.to(device),train_labels_singer.to(device)
            optimizer.zero_grad()
            output,output_singer = model(train_data)
            loss1 = criterion(output, train_labels)
            loss2 = criterion_singer(output_singer,train_labels_singer)
            loss=loss_weight1*loss1+loss_weight2*loss2
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        # with torch.no_grad():
        #     val_metrics = compute_validation_metrics(model, val_loader, device)
        #     conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(num_classes))
        num_classes=9
        num_classes_singer=11

        model.eval()
        correct = 0
        total = 0

        total_singer=0
        correct_singer=0
        conf_matrix = np.zeros((num_classes, num_classes))
        best_conf_matrix=conf_matrix

        conf_matrix_singer = np.zeros((num_classes_singer, num_classes_singer))
        # best_conf_matrix_singer=conf_matrix_singer

        with torch.no_grad(), tqdm(val_loader, unit="batch", disable=True) as vepoch:
            for features, labels,labels_singer in vepoch:
                features, labels,labels_singer = features.to(device), labels.to(device), labels_singer.to(device)
                outputs,outputs_singer = model(features)
                # Assume multimodal output is required for confusion matrix
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update confusion matrix
                conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(num_classes))

                


                # Handle singer output
                _, predicted_singer = torch.max(outputs_singer.data, 1)
                total_singer += labels_singer.size(0)
                correct_singer += (predicted_singer == labels_singer).sum().item()

                # Update singer confusion matrix
                conf_matrix_singer += confusion_matrix(
                    labels_singer.cpu().numpy(),
                    predicted_singer.cpu().numpy(),
                    labels=range(num_classes_singer)
                )

                

        val_accuracy = correct / total
        val_accuracy_singer = correct_singer / total_singer

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_conf_matrix=conf_matrix
            epochs_without_improvement = 0
            best_model=model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 50:
                trial_results[trial.number]=(best_val_accuracy,best_conf_matrix,conf_matrix_singer)
                break  # Early stopping

    # torch.save(best_model,os.path.join(pth_files_dir,'temp_model.pth'))
    confusion_matrix_dict[trial.number]=best_conf_matrix

    with open(os.path.join(pth_files_dir,'confusion_matrix_dict_gesture_split_1_GR.pkl'),'wb') as f:
        pickle.dump(confusion_matrix_dict,f)

    with open(os.path.join(pth_files_dir,'trial_results_gesture_split_1_GR.pkl'),'wb') as f:
        pickle.dump(trial_results,f)

    return best_val_accuracy

def compute_validation_metrics(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# db_name = 'gesture_classification_test_GR'  # Unique identifier of the study.
storage_name = 'sqlite:///{}.db'.format(db_name)

# study_name = 'gesture_classification1_split1_GR' # Unique identifier of the study.

study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)

# # Assuming train_X, train_y, val_X, val_y are already defined
# study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


print('Best trial:')
trial = study.best_trial
print('Value:', trial.value)







