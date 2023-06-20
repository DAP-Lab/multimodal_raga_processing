#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import tensorflow as tf


# In[3]:


tf.config.list_physical_devices('GPU')


# In[ ]:


# !pip install keras_tuner


# In[ ]:


# pip install -q  --upgrade mlxtend


# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import re
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import KFold
import keras_tuner
from keras import backend as K
from sklearn.metrics import *
from keras.utils import to_categorical
import datetime
import pickle
from tensorflow.keras import  regularizers



# In[ ]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[5]:


os.chdir('/media/antpc/Data/SujoyRc/IITB/RnD_Project/')


# In[6]:


from keras_tuner.tuners import BayesianOptimization,Hyperband
import keras_tuner as kt
from keras_tuner import Objective


# In[7]:


from keras.layers import *
from keras import Model


# In[8]:


num_classes=9
num_classes_singer=11
input_shape=(600,16)


# In[ ]:


# for each_file in sorted(os.listdir('./output_np'))[0:1]:
#   if not re.search('npz',each_file):
#     continue
#   print ("Running for ",each_file," at ",datetime.datetime.now())
#   split_data=np.load(os.path.join('./output_np/',each_file),allow_pickle=True)
#   train_X,train_y_raga,train_y_singer,val_X,val_y_raga,val_y_singer,test_X,test_y_raga,test_y_singer=[split_data[p] for p in split_data.files]


# In[9]:


np.set_printoptions(precision=2,suppress=True)


# In[10]:


from keras.callbacks import EarlyStopping


# In[11]:


# @tf.custom_gradient
# def gradient_reversal(x,wt):
#     y = tf.identity(x)
#     def custom_grad(dy):
#         return -1*wt*dy
#     return y, custom_grad

@tf.custom_gradient
def gradient_reversal(x, wt):
    y = tf.identity(x)
    def custom_grad(dy):
        return -1 * wt * dy, None  # Return gradients for both inputs
    return y, custom_grad

tf.keras.backend.set_floatx('float32')

def build_model(hp):
    input_data = layers.Input(shape=(600,16))
    reg_lambda=hp.Choice('reg_lambda', [0.1,0.01,0.001,0.0001])
    regularizer = regularizers.l2(reg_lambda)  


    for i in range(hp.Int('conv_layers', 1, 2, default=1)):
        x=layers.Conv1D(filters=hp.Choice('conv_filters_'+str(i), [16,32,64,128], default=16),
                                kernel_size=hp.Int('conv_kernel_'+str(i), 3, 7, step=2, default=3),
                                activation='relu',
                                padding='same',kernel_regularizer=regularizer)(input_data)
        x=layers.BatchNormalization()(x)
        x=layers.ReLU()(x)
        x=layers.AveragePooling1D(pool_size=2)(x)
        
#     for i in range(hp.Int('conv_layers', 1, 2, default=1)):
#         x1=layers.Conv1D(filters=hp.Int('conv_filters_'+str(i), 16, 128, step=16, default=64),
#                                 kernel_size=hp.Int('conv_kernel_'+str(i), 3, 7, step=2, default=3),
#                                 activation='relu',
#                                 padding='same')(input_data)
#         x1=layers.BatchNormalization()(x1)
#         x1=layers.ReLU()(x1)
#         x1=layers.AveragePooling1D(pool_size=2)(x1)

    inception_filters = hp.Choice('inception_filters', [4,8,16,32], default=16)
    for i in range(1):
        layer_1 = Conv1D(inception_filters, 1, padding='same', activation='relu',kernel_regularizer=regularizer)(x)

        layer_2 = Conv1D(inception_filters, 3, padding='same', activation='relu',kernel_regularizer=regularizer)(x)
        layer_2 = Conv1D(inception_filters, 5, padding='same', activation='relu',kernel_regularizer=regularizer)(layer_2)

        #layer_3 = MaxPool1D(2,  padding='same')(x)
        layer_3 = Conv1D(inception_filters, 1, padding='same', activation='relu',kernel_regularizer=regularizer)(x)

        mid_1 = layers.concatenate([layer_1, layer_2, layer_3], axis = 2)
        
        
    # inception_filters1 = hp.Int('inception_filters1', 4, 32, step=4, default=64)
    # for i in range(1):
    #     layer_1_1 = Conv1D(inception_filters1, 1, padding='same', activation='relu',kernel_regularizer=regularizer)(x)

    #     layer_2_1 = Conv1D(inception_filters1, 3, padding='same', activation='relu',kernel_regularizer=regularizer)(x)
    #     layer_2_1 = Conv1D(inception_filters1, 5, padding='same', activation='relu',kernel_regularizer=regularizer)(layer_2)

    #     #layer_3 = MaxPool1D(2,  padding='same')(x)
    #     layer_3_1 = Conv1D(inception_filters1, 1, padding='same', activation='relu',kernel_regularizer=regularizer)(x)

    #     mid_1_1 = layers.concatenate([layer_1_1, layer_2_1, layer_3_1], axis = 2)

    x=layers.Flatten()(mid_1)
    # x1=layers.Flatten()(mid_1_1)
    gr_wt=hp.Choice('gr_weight',[0.05*(x) for x in range(1,20)])
    # gr_wt=1.0

    # reverse_layer = layers.Lambda(lambda x: gradient_reversal(x, gr_wt))(x) # apply gradient reversal to hidden_layer
    reverse_layer=x

    dropout_rate=hp.Choice('dropout_rate', [0.3,0.5,0.7])


    if hp.Boolean('dense_layer', default=False):
        x=layers.Dense(units=hp.Choice('dense_units', [16,32,64,128], default=64), activation='relu',kernel_regularizer=regularizer)(x)
        x=layers.Dropout(dropout_rate)(x)
        x=layers.BatchNormalization()(x)
        x=layers.ReLU()(x)

    if hp.Boolean('dense_layer_aux', default=False):
        x_aux=layers.Dense(units=hp.Choice('dense_units', [16,32,64,128], default=64), activation='relu',kernel_regularizer=regularizer)(reverse_layer)
        x_aux=layers.Dropout(dropout_rate)(x_aux)
        x_aux=layers.BatchNormalization()(x_aux)
        x_aux=layers.ReLU()(x_aux)
    else:
        x_aux=reverse_layer
            
    # if hp.Boolean('dense_layer1', default=False):
    #     x1=layers.Dense(units=hp.Int('dense_units1', 16, 128, step=16, default=64), activation='relu',kernel_regularizer=regularizer)(x1)
    #     x1=layers.Dropout(dropout_rate)(x1)
    #     x1=layers.BatchNormalization()(x1)
    #     x1=layers.ReLU()(x1)

    # if hp.Boolean('dense_layer_aux1', default=False):
    #     x1_aux=layers.Dense(units=hp.Int('dense_units', 16, 128, step=16, default=64), activation='relu',kernel_regularizer=regularizer)(x1)
    #     x1_aux=layers.Dropout(dropout_rate)(x1_aux)
    #     x1_aux=layers.BatchNormalization()(x1_aux)
    #     x1_aux=layers.ReLU()(x1_aux)
    # else:
    #     x1_aux=x1

    
    # Output Layer
    output1=layers.Dense(units=num_classes, activation='softmax',name='output1',kernel_regularizer=regularizer)(x)
        
    output1_aux=layers.Dense(units=num_classes_singer, activation='softmax',name='output1_aux',kernel_regularizer=regularizer)(x_aux)
    
    # output2=layers.Dense(units=num_classes_singer, activation='softmax' ,name='output2',kernel_regularizer=regularizer)(x1)
    #reverse_layer1 = layers.Lambda(lambda x: gradient_reversal(x, gr_wt))(x1_aux) # apply gradient reversal to hidden_layer 
    # output2_aux=layers.Dense(units=num_classes, activation='softmax' ,name='output2_aux',kernel_regularizer=regularizer)(x1_aux)
    
    print (output1.shape)
    model = Model([input_data], [output1,output1_aux])
    
    
    loss_weight1 = 1.0 #hp.Choice('loss_weight1',[0.05*(x) for x in range(1,20)])
    # loss_weight2 = hp.Float('loss_weight2',0.01,0.99)
    # loss_weight1=0.2

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-5,1e-4,1e-3,1e-2])),
                  loss={'output1': 'categorical_crossentropy',
                    'output1_aux': 'categorical_crossentropy'\
                        }\
                  ,loss_weights={'output1': loss_weight1
                                 ,'output1_aux': 1-loss_weight1},
                   metrics={'output1': 'accuracy',
                       'output1_aux': 'accuracy'})
    
    return model


# In[ ]:


from keras import backend as K
K.clear_session()

results_splitwise={}
counter=0

#each_file='split_1'
for each_file in sorted(os.listdir('./output_np_1')):
    #directory_name='./output_np_1/'+each_file
    train_data=np.load(os.path.join('./output_np_1/'+each_file,each_file+'_train.npz'))
    val_data=np.load(os.path.join('./output_np_1/'+each_file,each_file+'_val.npz')) #np.load('split_1_val.npz')
    test_data=np.load(os.path.join('./output_np_1/'+each_file,each_file+'_test.npz')) # np.load('split_1_test.npz')

    train_X_gesture,train_y_sing,train_y,train_filename=[train_data[p] for p in train_data.files]
    val_X_gesture,val_y_sing,val_y,val_filename=[val_data[p] for p in val_data.files]
    test_X_gesture,test_y_sing,test_y,train_filename=[test_data[p] for p in test_data.files]

    train_X_gesture=np.nan_to_num(train_X_gesture)
    val_X_gesture=np.nan_to_num(val_X_gesture)
    test_X_gesture=np.nan_to_num(test_X_gesture)

    train_X_gesture=train_X_gesture[:,:,:]
    val_X_gesture=val_X_gesture[:,:,:]
    test_X_gesture=test_X_gesture[:,:,:]

    tuner = BayesianOptimization(
        build_model,
        objective=keras_tuner.Objective('val_output1_accuracy',direction='max'),
        max_trials=25,
        executions_per_trial=1,
        directory='./processing_logs_no_GR_pva1/keras_tuner_gradient_reversal3_'+each_file,
        project_name='classification_9_classes',
        overwrite=False
    )

    tb_dir="./processing_logs_no_GR_pva1/tb_logs_peri_v3_2_1_"+each_file
    # Perform the hyperparameter search
    tuner.search(train_X_gesture, [np.asarray(train_y,dtype=np.float32),np.asarray(train_y_sing,dtype=np.float32)], epochs=300\
                , callbacks=[EarlyStopping(monitor='val_output1_accuracy', min_delta=0.02,patience=50)\
                            ,keras.callbacks.TensorBoard(tb_dir),keras.callbacks.ReduceLROnPlateau(
                monitor="val_output1_accuracy", min_delta=0.02, patience=10, min_lr=0.0001)]\
                , validation_data=(val_X_gesture, [np.asarray(val_y,dtype=np.float32),np.asarray(val_y_sing,dtype=np.float32)]))

    es=EarlyStopping( monitor="val_output1_accuracy",
        min_delta=0.02,
        patience=50,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )

    epochs = 300
    batch_size = 32

    num_batches=int(train_X_gesture.shape[0]/32)
    model_file_name='./processing_logs_no_GR_pva1/best_model_v3'+each_file+'.h5'
    

    callbacks = [
            keras.callbacks.ModelCheckpoint(
                model_file_name, save_best_only=True, monitor="val_output1_accuracy",save_freq='epoch'),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_output1_accuracy", min_delta=0.02, patience=10, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_output1_accuracy", patience=50,min_delta=0.02, verbose=1),
        ]
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    best_learning_rate = best_hps.get('learning_rate')
    loss_weight1=1.0 #best_hps.get('loss_weight1')



    best_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_learning_rate),
                    loss={'output1': 'categorical_crossentropy',
                        'output1_aux': 'categorical_crossentropy'\
                            }\
                    ,loss_weights={'output1': loss_weight1
                                    ,'output1_aux': 1-loss_weight1},
                    metrics={'output1': 'accuracy',
                        'output1_aux': 'accuracy'})
    
    results_splitwise[each_file]={}
    pred_y_val,pred_y_val_sing=best_model.predict(val_X_gesture)
    pred_y_val_classes=np.argmax(pred_y_val,axis=1)
    pred_y_val_classes_sing=np.argmax(pred_y_val_sing,axis=1)
    test_y_val_classes=np.argmax(val_y,axis=1)
    test_y_val_classes_sing=np.argmax(val_y_sing,axis=1)
    cm_val=confusion_matrix(pred_y_val_classes,test_y_val_classes)
    cm_val_sing=confusion_matrix(pred_y_val_classes_sing,test_y_val_classes_sing)
    results_splitwise[each_file]['val_y']=val_y
    results_splitwise[each_file]['pred_y_val']=pred_y_val
    results_splitwise[each_file]['val_y_sing']=val_y_sing
    results_splitwise[each_file]['pred_y_val_sing']=pred_y_val_sing
    results_splitwise[each_file]['accuracy_val']=accuracy_score(pred_y_val_classes,test_y_val_classes)
    results_splitwise[each_file]['confusion_matrix_val']=cm_val
    results_splitwise[each_file]['accuracy_val_sing']=accuracy_score(pred_y_val_classes_sing,test_y_val_classes_sing)
    results_splitwise[each_file]['confusion_matrix_val_sing']=cm_val_sing

    train_val_X_gesture=np.vstack([train_X_gesture,val_X_gesture])
    train_val_y=np.vstack([train_y,val_y])
    train_val_y_sing=np.vstack([train_y_sing,val_y_sing])
    
    history = best_model.fit(
        train_val_X_gesture,
        [np.asarray(train_val_y,dtype=np.float32),np.asarray(train_val_y_sing,dtype=np.float32)],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(test_X_gesture, [np.asarray(test_y,dtype=np.float32),np.asarray(test_y_sing,dtype=np.float32)]),
        verbose=1,
    )

    best_model=keras.models.load_model(model_file_name)    
    pred_y,_=best_model.predict(test_X_gesture)
    pred_y_classes=np.argmax(pred_y,axis=1)
    test_y_classes=np.argmax(test_y,axis=1)
    cm=confusion_matrix(pred_y_classes,test_y_classes)
    results_splitwise[each_file]['test_y']=test_y
    results_splitwise[each_file]['pred_y']=pred_y
    best_hp = tuner.get_best_hyperparameters()[0]
    model_best_hp = tuner.hypermodel.build(best_hp)   
    results_splitwise[each_file]['best_hp']=best_hp
    results_splitwise[each_file]['accuracy']=accuracy_score(pred_y_classes,test_y_classes)
    results_splitwise[each_file]['confusion_matrix_test']=cm
    #results_splitwise[each_file]['accuracy_sing']=accuracy_score(pred_y_classes_sing,test_y_classes_sing)
    #results_splitwise[each_file]['confusion_matrix_test_sing']=cm_sing


    pickle_filename='./processing_logs_no_GR_pva1/results_splitwise_'+re.sub('.npz','',each_file)+'_multitask_va_GR_v4.pkl'

    with open(pickle_filename,'wb') as f:
        pickle.dump(results_splitwise,f)

    counter=counter+1


# In[14]:

print (results_splitwise)
