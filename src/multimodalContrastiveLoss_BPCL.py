#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import optuna
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")
import configparser
import sys

if len(sys.argv)!=2:
    print ("Usage ",sys.argv[0]," <split_name>")
    exit(1)
else:
    split_name=sys.argv[1]
    view_name=sys.argv[2].lower()

if split_name not in ('split_1','split_2','split_3'):
    print ("Usage give split as split1,split2,split3")
    exit(1)

if view_name not in ('front','left','right'):
    print ("Usage give split as front,left,right")
    exit(1)

split_name=split_name.upper()

config = configparser.ConfigParser()
config.read('config.ini')
section=split_name

audio_split = config[section]['train_X_audio']
train_y_path = config[section]['train_y']
audio_split_path = config[section]['audio_unimodal_output']
video_split_path = config[section]['video_unimodal_output']
val_y_path = config[section]['val_y']
db_name = config[section]['db_name_audio']
study_name = config[section]['study_name_audio']
db_name = config[section]['db_name_multimodal']
study_name = config[section]['study_name_multimodal']


with open(audio_split_path,'rb') as f:
    audio_split=pickle.load(f)


with open(video_split_path,'rb') as f:
    video_split=pickle.load(f)


train_audio=audio_split['prefinal_layer_output_train_audio']
val_audio=audio_split['prefinal_layer_output_val_audio']

train_video=video_split['prefinal_layer_output_train_raga']
val_video=video_split['prefinal_layer_output_val_raga']


train_y=np.load(train_y_path)
val_y=np.load(val_y_path)


import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)


trial_results={}


num_classes=9



num_classes=9
best_model=0

class MyModel(nn.Module):
    def __init__(self, audio_input_dim, video_input_dim\
                 , common_embed_dim, intermediate_dim1, intermediate_dim2, \
                    output_softmax_dim, dropout_rate):
        super(MyModel, self).__init__()
        
        # Input to embedding layers with dropout
        self.audio_dense = nn.Sequential(
            nn.Linear(audio_input_dim, common_embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.video_dense = nn.Sequential(
            nn.Linear(video_input_dim, common_embed_dim),
            nn.Dropout(dropout_rate)
        )
       
        # Common embedding to multimodal output with dropout
        self.multimodal_dense = nn.Sequential(
            nn.Linear(2 * common_embed_dim, common_embed_dim),  # Modified to handle concatenated embeddings
            nn.Dropout(dropout_rate),
            nn.Linear(common_embed_dim, 9)  # Output is a 9-way softmax
        )
        
        # Intermediate and softmax output layers for unimodal tasks with dropout
        self.intermediate_audio = nn.Sequential(
            nn.Linear(common_embed_dim, intermediate_dim1),
            nn.Dropout(dropout_rate)
        )
        self.output_softmax_audio = nn.Linear(intermediate_dim1, output_softmax_dim)

        self.intermediate_video = nn.Sequential(
            nn.Linear(common_embed_dim, intermediate_dim2),
            nn.Dropout(dropout_rate)
        )
        self.output_softmax_video_front = nn.Linear(intermediate_dim2, output_softmax_dim)

    def audio_parameters(self):
        return list(self.audio_dense.parameters()) + \
            list(self.intermediate_audio.parameters()) + \
            list(self.output_softmax_audio.parameters())

    def non_audio_parameters(self):
        return [p for p in self.parameters() if p not in set(self.audio_parameters())]

    def forward(self, audio_layer, video_layer):
        # Embeddings
        audio_embed = F.relu(self.audio_dense(audio_layer))
        video_embed = F.relu(self.video_dense(video_layer))

        # Multimodal feature
        multimodal_feature = torch.cat([audio_embed, video_embed], dim=1)
        multimodal_output = self.multimodal_dense(multimodal_feature)
        multimodal_softmax = F.softmax(multimodal_output, dim=1)

        # Unimodal intermediate and output layers
        audio_intermediate = F.relu(self.intermediate_audio(audio_embed))
        audio_output = self.output_softmax_audio(audio_intermediate)
        audio_softmax = F.softmax(audio_output, dim=1)

        video_intermediate = F.relu(self.intermediate_video(video_embed))
        video_output = self.output_softmax_video_front(video_intermediate)
        video_softmax = F.softmax(video_output, dim=1)

        return {
            'multimodal_softmax': multimodal_softmax,
            'audio_softmax': audio_softmax,
            'video_softmax': video_softmax,
            'audio_intermediate' : audio_intermediate,
            'video_intermediate': video_intermediate,
            'audio_embed': audio_embed,
            'video_embed': video_embed,
        }
    
def negative_sample_loss(audio_vectors, negative_vectors, m):
    # Normalize the vectors to ensure cosine similarity is equivalent to dot product
    audio_vectors = F.normalize(audio_vectors, p=2, dim=1)
    negative_vectors = F.normalize(negative_vectors, p=2, dim=1)
    
    # Calculate cosine similarity
    cos_sim = torch.mm(audio_vectors, negative_vectors.t())
    
    # Apply the margin
    losses = F.relu(cos_sim - m)
    
    # Compute the loss as the mean of the maximum values across the cosine similarity matrix
    loss = losses.mean()
    
    return loss

def sample_hard_negative_vectors(audio_vectors, train_y):
    N, D = audio_vectors.shape
    
    # Normalize the vectors to get cosine similarity as dot product
    normalized_vectors = F.normalize(audio_vectors, p=2, dim=1).cuda()
    
    # Compute cosine similarity matrix
    cosine_sim = torch.matmul(normalized_vectors, normalized_vectors.t()).cuda()
    
    # Create a mask for valid hard negatives (different class and not the same vector)
    same_class_mask = (train_y.view(-1, 1) == train_y.view(1, -1)).cuda()
    identity_mask = torch.eye(N).to(train_y.device).bool().cuda()
    valid_negatives_mask = ~(same_class_mask | identity_mask)
    
    # Set cosine similarity of same class and identity to a very low number
    cosine_sim_masked = cosine_sim.masked_fill(~valid_negatives_mask, float('-inf')).cuda()
    
    # Get the index of the max cosine similarity (most similar but different class)
    _, negative_indices = torch.max(cosine_sim_masked, dim=1)
    
    # Use the indices to gather the hard negative samples
    hard_negative_vectors = audio_vectors[negative_indices]
    
    return hard_negative_vectors

def contrastive_loss(z_i, z_j, temperature):
    """
    Calculates the contrastive loss between two sets of vectors z_i and z_j.
    """
    N = z_i.size(0)  # Batch size
    device = z_i.device

    # Normalize the embeddings
    z_i_norm = F.normalize(z_i, p=2, dim=1)
    z_j_norm = F.normalize(z_j, p=2, dim=1)

    # Compute similarity with temperature scaling
    sim_matrix = torch.matmul(z_i_norm, z_j_norm.T) / temperature

    # Create labels for the positive pairs (diagonal elements of the similarity matrix)
    labels = torch.arange(N, device=device)
    
    # Calculate the log probabilities
    loss_i_j = F.cross_entropy(sim_matrix, labels)
    
    return loss_i_j

def compute_final_contrastive_loss(audio_embeddings, video_embeddings, temperature):
    """
    Computes the final contrastive loss as defined in Eq. (2) using audio and video embeddings.
    """
    # Assuming audio_embeddings and video_embeddings are of shape (N, D)
    # Ensure the embeddings are on the same device (GPU) before calculating the loss
    audio_embeddings = audio_embeddings.to('cuda')
    video_embeddings = video_embeddings.to('cuda')
    
    loss_audio_video = contrastive_loss(audio_embeddings, video_embeddings, temperature)
    loss_video_audio = contrastive_loss(video_embeddings, audio_embeddings, temperature)
    
    # Final loss is the average of audio->video and video->audio losses
    final_loss = (loss_audio_video + loss_video_audio) / 2
    
    return final_loss


def objective(trial):
    # Hyperparameters to be tuned by Optuna
    audio_input_dim=train_audio.shape[1]
    video_input_dim=train_video.shape[1]
    
    output_softmax_dim=num_classes
    
    
    common_embed_dim = trial.suggest_int('common_embed_dim', 2, 128)
    intermediate_dim1 = trial.suggest_int('hidden_dim1', 2, 128)
    intermediate_dim2 = intermediate_dim1

    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    
    lambda1=trial.suggest_uniform('lambda1',0.0,1.0)
    lambda2=trial.suggest_uniform('lambda2',0.0,1.0)
    lambda3=trial.suggest_uniform('lambda2',0.0,1.0)
    # m=trial.suggest_int('m',1,10)
    # m=float(m/10)
    # lambda1=lambda2=lambda3=1
    m=0.5
    temperature=trial.suggest_uniform('temperature',0,1)


    model = MyModel(
        audio_input_dim, 
        video_input_dim, 
        common_embed_dim, 
        intermediate_dim1,
        intermediate_dim2,
        output_softmax_dim,
        dropout_rate
    )
    
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_audio = optim.Adam(model.audio_parameters(), lr=lr)
    optimizer_non_audio = optim.Adam(model.non_audio_parameters(), lr=lr)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    # Calculate losses
    optimizer_audio.zero_grad()
    optimizer_non_audio.zero_grad()
    
    train_audio_tensor = torch.tensor(train_audio, dtype=torch.float32).to(device)
    train_video_tensor = torch.tensor(train_video, dtype=torch.float32).to(device)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long).to(device) 
    
    val_audio_tensor = torch.tensor(val_audio, dtype=torch.float32).to(device)
    val_video_tensor = torch.tensor(val_video, dtype=torch.float32).to(device)
    val_y_tensor = torch.tensor(val_y, dtype=torch.long).to(device)


    # Data loaders
    train_dataset = TensorDataset(train_audio_tensor,train_video_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_audio_tensor, val_video_tensor, val_y_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    best_val_accuracy = 0
    epochs_without_improvement = 0
    num_epochs=100
    
    for epoch in range(num_epochs):  # Max epochs
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for audio, video,  labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer_audio.zero_grad()
                optimizer_non_audio.zero_grad()
                
                outputs = model(audio, video)
                loss_multimodal_ce = criterion_ce(outputs['multimodal_softmax'], labels)
                loss_audio_ce = criterion_ce(outputs['audio_softmax'], labels)
                loss_video_ce = criterion_ce(outputs['video_softmax'], labels)
                audio_vectors=outputs['audio_embed']
                video_vectors=outputs['video_embed']
                multimodal_pairwise_contrastive_loss=compute_final_contrastive_loss(audio_vectors,video_vectors,temperature)
                loss_non_audio=multimodal_pairwise_contrastive_loss+loss_video_ce+loss_multimodal_ce
                loss_audio=multimodal_pairwise_contrastive_loss+loss_audio_ce+loss_multimodal_ce
                loss_audio.backward(retain_graph=True)
                loss_non_audio.backward()
                optimizer_audio.step()
                optimizer_non_audio.step()

                tepoch.set_postfix(loss=loss_non_audio.item())

        # Validation
        model.eval()
        correct = 0
        total = 0
        conf_matrix = np.zeros((num_classes, num_classes))
        best_conf_matrix=conf_matrix
        
        with torch.no_grad(), tqdm(val_loader, unit="batch") as vepoch:
            for audio, video, labels in vepoch:
                outputs = model(audio, video)
                # Assume multimodal output is required for confusion matrix
                _, predicted = torch.max(outputs['multimodal_softmax'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update confusion matrix
                conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(num_classes))

        val_accuracy = correct / total

        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_conf_matrix=conf_matrix
            epochs_without_improvement = 0
            best_model=model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 5:
                trial_results[trial.number]=(best_val_accuracy,best_conf_matrix)
                break  # Early stopping
                                                   
   
    trial_results[trial.number]=(best_val_accuracy,best_conf_matrix)
    return best_val_accuracy


# In[19]:


storage_name = 'sqlite:///{}.db'.format(db_name)

study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=200)  # Adjust n_trials as per requirement

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Best trial:")
trial = study.best_trial
print("    Value: ", trial.value)
print("    Params: ")
for key, value in trial.params.items():
    print(f"      {key}: {value}")


# In[ ]:


os.makedirs('../model',exist_ok=True)


best_model_output_file='model/'+split_name+'_'+study_name+'.pth'
torch.save(best_model,best_model_output_file)

