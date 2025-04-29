
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import logging.config
from gym import  spaces
import gym
import json
import sys
import time
import torch_geometric
from torch_geometric.utils import from_networkx, add_self_loops, degree
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborSampler
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

X_test = pd.read_csv(r"") #upload path of X_test.csv
cols_to_norm = ['LONGEST_FLOW_PKT', 'RETRANSMITTED_IN_PKTS', 'MAX_IP_PKT_LEN', 'TCP_WIN_MAX_IN', 'RETRANSMITTED_OUT_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'OUT_PKTS', 'DURATION_OUT', 'IN_PKTS', 'FLOW_DURATION_MILLISECONDS', 'MIN_IP_PKT_LEN', 'NUM_PKTS_512_TO_1024_BYTES', 'RETRANSMITTED_IN_BYTES', 'DURATION_IN', 'SHORTEST_FLOW_PKT', 'IN_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_SECOND_BYTES', 'TCP_WIN_MAX_OUT', 'SRC_TO_DST_AVG_THROUGHPUT', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'OUT_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_UP_TO_128_BYTES', 'MAX_TTL', 'MIN_TTL']

from sklearn.preprocessing import LabelEncoder

test_le = LabelEncoder()
X_test['Attack'] = test_le.fit_transform(X_test['Attack'])
test_class_map = test_le.classes_

X_test['h'] = X_test[ cols_to_norm ].values.tolist()

G_nx_test = nx.from_pandas_edgelist(X_test, "saddr", "daddr", ['h', 'Label', 'Attack'], create_using=nx.MultiDiGraph())

G_pyg_test = from_networkx(G_nx_test)

test_num_nodes = G_pyg_test.num_nodes
test_num_edges = G_pyg_test.num_edges

G_pyg_test.x = th.ones(test_num_nodes, len(X_test['h'].iloc[0]))

test_edge_attr_list = []
test_edge_label_list = []
test_edge_class_list = []

for u, v, key, data in G_nx_test.edges(keys=True, data=True):
    test_edge_attr_list.append(data['h']) 
    test_edge_label_list.append(data['Label']) 
    test_edge_class_list.append(data['Attack'])

G_pyg_test.edge_attr = th.tensor(test_edge_attr_list, dtype=th.float32)
benign_class_index = test_le.transform(['Benign'])[0]
edge_labels_binary = [0 if attack == benign_class_index else 1 for attack in test_edge_class_list]
G_pyg_test.edge_label = th.tensor(edge_labels_binary, dtype=th.long)

G_pyg_test.edge_class = th.tensor(test_edge_class_list, dtype=th.long)

print("Number of edges in G_pyg_test:", G_pyg_test.num_edges)
print("Number of node in G_pyg_test:", G_pyg_test.num_nodes)
print("Shape of node in G_pyg_test:", G_pyg_test.x.shape)
print("Shape of edge attr in G_pyg_test:", G_pyg_test.edge_attr.shape)
print("Shape of edge label in G_pyg_test:", G_pyg_test.edge_label.shape)
print("Shape of edge class in G_pyg_test:", G_pyg_test.edge_class.shape)

class data_cls:
    def __init__(self, train_test, attack_map, **kwargs):
        self.train_test = train_test
        
        if self.train_test == 'train':
            self.train_path = r"" #upload your train_gnn.csv file
        else:
            self.test_path = r"" #upload your test_gnn.csv file

        self.attack_map =   attack_map 
        self.attack_types = list(attack_map.keys())
        
        self.loaded = False
        
    
    def get_batch(self, batch_size=100):
        
        if self.loaded is False:
            self._load_df()
            
        indexes = list(range(self.index,self.index+batch_size))    
        if max(indexes)>self.data_shape[0]-1:
            dif = max(indexes)-self.data_shape[0]
            indexes[len(indexes)-dif-1:len(indexes)] = list(range(dif+1))
            self.index=batch_size-dif
            batch = self.df.iloc[indexes]
        else: 
            batch = self.df.iloc[indexes]
            self.index += batch_size    
        
        map_type = pd.Series(index=self.attack_types,data=np.arange(len(self.attack_types))).to_dict()
        labels = batch[label_col].map(self.attack_map).map(map_type).values
        del(batch[label_col])
            
        return np.array(batch),labels
    
    def get_full(self):

        self._load_df()
        
        batch = self.df
        map_type = pd.Series(index=self.attack_types,data=np.arange(len(self.attack_types))).to_dict()
        labels = batch[label_col].map(self.attack_map).map(map_type).values
        
        del(batch[label_col])
        
        return np.array(batch), labels
    
    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        
        self.data_shape = self.df.shape
        return self.data_shape
    
    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.train_path) 
        else:
            self.df = pd.read_csv(self.test_path)
            
        self.index=np.random.randint(0,self.df.shape[0]-1,dtype=np.int32)
        self.loaded = True


class NetworkClassificationEnv(gym.Env, data_cls):
    def __init__(self,train_test, attack_map, **kwargs):
        data_cls.__init__(self,train_test, attack_map,**kwargs)
        self.data_shape = self.get_shape()
        self.batch_size = kwargs.get('batch_size', 1) 
        self.fails_episode = kwargs.get('fails_episode', 10) 
        
        # Gym spaces
        self.action_space = spaces.Discrete(len(self.attack_types))
        self.observation_space = spaces.Discrete(self.data_shape[0])
        
        self.observation_len = self.data_shape[1]-1
        
        self.counter = 0

    def _update_state(self):
        self.states,self.labels = self.get_batch(self.batch_size)
        

    def reset(self):
        self.states,self.labels = self.get_batch(self.batch_size)
        self.counter = 0
        
        return self.states
    def _get_rewards(self,actions):
        self.reward = 0
        if actions == self.labels:
            self.reward = 1
        else: 
            self.counter += 1

    def step(self,actions):
        self._get_rewards(actions)
            
        self._update_state()

        if self.counter >= self.fails_episode:
            self.done = True
        else:
            self.done = False
            
        return self.states, self.reward, self.done
    

import tensorflow as tf
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable(package="Custom", name="huber_loss")
def huber_loss(y_true, y_pred, clip_value=1.0):
    assert clip_value > 0.0, "clip_value must be positive."

    x = y_true - y_pred
    
    # Handle the case when clip_value is infinity
    if np.isinf(clip_value):
        return 0.5 * tf.square(x)

    # Define condition for the quadratic part
    condition = tf.abs(x) < clip_value

    # Compute squared and linear losses
    squared_loss = 0.5 * tf.square(x)
    linear_loss = clip_value * (tf.abs(x) - 0.5 * clip_value)

    # Use TensorFlow's conditional logic to return the appropriate loss
    return tf.where(condition, squared_loss, linear_loss)


# Check if the weights and model files exist
weights_path = r"" #upload class_weights.pth path
json_path = r"" #upload gnn_model.json file

if os.path.exists(weights_path) and os.path.exists(json_path):
    print("Model files exist.")
else:
    print("Model files are missing.")


import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import  confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


with open(json_path) as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights(weights_path)
label_dict = r"" #upload label_dict.json file

with open(label_dict, "r") as f:
    label_dict = json.load(f)


model.compile(loss=huber_loss, optimizer="sgd")

env = NetworkClassificationEnv('test', label_dict) 

total_reward = 0

label_col = 'Label'
    
true_labels = np.zeros(len(env.attack_types),dtype=int)
estimated_labels = np.zeros(len(env.attack_types),dtype=int)
estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)

states, labels = env.get_full()
q = model.predict(states)
actions = np.argmax(q, axis=1)        


labs, true_labels = np.unique(labels, return_counts=True)

for indx,a in enumerate(actions):
    estimated_labels[a] +=1              
    if a == labels[indx]:
        total_reward += 1
        estimated_correct_labels[a] += 1


Accuracy = estimated_correct_labels / true_labels
Mismatch = estimated_labels - true_labels
print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
      len(states),float(100*total_reward/len(states))))
outputs_df = pd.DataFrame(index = env.attack_types,columns = ["Estimated","Correct","Total","Acuracy"])
for indx,att in enumerate(env.attack_types):
    outputs_df.iloc[indx].Estimated = estimated_labels[indx]
    outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
    outputs_df.iloc[indx].Total = true_labels[indx]
    outputs_df.iloc[indx].Acuracy = Accuracy[indx]*100
    outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])


outputs_df

fig, ax = plt.subplots()
width = 0.35
pos = np.arange(len(true_labels))
p1 = plt.bar(pos, estimated_correct_labels,width,color='g')
p1 = plt.bar(pos+width,
             (np.abs(estimated_correct_labels-true_labels)),width,
             color='r')
p2 = plt.bar(pos+width,np.abs(estimated_labels-estimated_correct_labels),width,
             bottom=(np.abs(estimated_correct_labels-true_labels)),
             color='b')


ax.set_xticks(pos+width/2)
ax.set_xticklabels(env.attack_types,rotation='vertical')

ax.set_title('Test set scores, Acc = {:.2f}'.format(100*total_reward/len(states)))
plt.legend(('Correct estimated','False negative','False positive'))
plt.tight_layout()
plt.show()

aggregated_data_test =labels

print('Performance measures on Test data')
print('Accuracy =  {:.4f}'.format(accuracy_score( aggregated_data_test,actions)))
print('F1 =  {:.4f}'.format(f1_score(aggregated_data_test,actions, average='weighted')))
print('Precision_score =  {:.4f}'.format(precision_score(aggregated_data_test,actions, average='weighted')))
print('recall_score =  {:.4f}'.format(recall_score(aggregated_data_test,actions, average='weighted')))

cnf_matrix = confusion_matrix(aggregated_data_test,actions)
np.set_printoptions(precision=2)
plt.figure()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=env.attack_types, normalize=True,
                      title='Normalized confusion matrix')


import seaborn as sns
cm=confusion_matrix(aggregated_data_test,actions)
f,ax=plt.subplots(figsize=(10,8))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax,cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()