import numpy as np 
import pandas as pd 
import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error, roc_auc_score, accuracy_score, confusion_matrix

data_path = '../data/'
output_path = '../mod_data/'
result_path = '../results/'

train = pd.read_csv(output_path+'combined_train.csv')

train_columns = ['user_id', 'content_id', 'content_type', 'bundle_id_x', 'answer',
       'explan', 'days', 'elapsed_days', 'lecture', 'ability', 'ques_id',
       'ques_part', 'wrong', 'right', 'hard', 'easy', 'ques_cluster', 'tag_1',
       'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'correct']

N = 20000
x = train.iloc[:N, 1:-1].values 
y = train.iloc[:N, -1].values 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

train_auc_list = []
test_auc_list = []
train_acc_list = []
test_acc_list = []


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_yt = lr.predict(x_train)
lr_y = lr.predict(x_test)

lr_auc_t = roc_auc_score(y_train, lr_yt)
lr_auc = roc_auc_score(y_test, lr_y)
print('lr: train %s, test %s'%(lr_auc_t, lr_auc))
lr_acc_t = accuracy_score(y_train, lr_yt)
lr_acc = accuracy_score(y_test, lr_y)
print('lr: train %s, test %s'%(lr_acc_t, lr_acc))

train_auc_list.append(lr_auc_t)
test_auc_list.append(lr_auc)
train_acc_list.append(lr_acc_t)
test_acc_list.append(lr_acc)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(x_train, y_train)
rf_yt = rf.predict(x_train)
rf_y =rf.predict(x_test)

rf_auc_t = roc_auc_score(y_train, rf_yt)
rf_auc = roc_auc_score(y_test, rf_y)
print('rf: train %s, test %s'%(rf_auc_t, rf_auc))
rf_acc_t = accuracy_score(y_train, rf_yt)
rf_acc = accuracy_score(y_test, rf_y)
print('rf: train %s, test %s'%(rf_acc_t, rf_acc))

train_auc_list.append(rf_auc_t)
test_auc_list.append(rf_auc)
train_acc_list.append(rf_acc_t)
test_acc_list.append(rf_acc)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn_yt = knn.predict(x_train)
knn_y = knn.predict(x_test)
knn_auc_t = roc_auc_score(y_train, knn_yt)
knn_auc = roc_auc_score(y_test, knn_y)
print('knn: train %s, test %s'%(knn_auc_t, knn_auc))
knn_acc_t = accuracy_score(y_train, knn_yt)
knn_acc = accuracy_score(y_test, knn_y)
print('knn: train %s, test %s'%(knn_acc_t, knn_acc))


train_auc_list.append(knn_auc_t)
test_auc_list.append(knn_auc)
train_acc_list.append(knn_acc_t)
test_acc_list.append(knn_acc)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
svc_yt = knn.predict(x_train)
svc_y = knn.predict(x_test)
svc_auc_t = roc_auc_score(y_train, svc_yt)
svc_auc = roc_auc_score(y_test, svc_y)
print('svc: train %s, test %s'%(svc_auc_t, svc_auc))
svc_acc_t = accuracy_score(y_train, svc_yt)
svc_acc = accuracy_score(y_test, svc_y)
print('svc: train %s, test %s'%(svc_acc_t, svc_acc))

train_auc_list.append(svc_auc_t)
test_auc_list.append(svc_auc)
train_acc_list.append(svc_acc_t)
test_acc_list.append(svc_acc)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree_yt = knn.predict(x_train)
tree_y = knn.predict(x_test)
tree_auc_t = roc_auc_score(y_train, tree_yt)
tree_auc = roc_auc_score(y_test, tree_y)
print('tree: train %s, test %s'%(tree_auc_t, tree_auc))
tree_acc_t = accuracy_score(y_train, tree_yt)
tree_acc = accuracy_score(y_test, tree_y)
print('tree: train %s, test %s'%(tree_acc_t, tree_acc))
train_auc_list.append(tree_auc_t)
test_auc_list.append(tree_auc)
train_acc_list.append(tree_acc_t)
test_acc_list.append(tree_acc)

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim 
class Dataset_py(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.long)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


y_train_array = y_train.reshape((len(y_train), 1))
y_test_array = y_test.reshape((len(y_test), 1))

batch_size = 128
train_ds = Dataset_py(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size)
xtrain_t = torch.tensor(x_train, dtype=torch.float32)
xtest_t = torch.tensor(x_test, dtype=torch.float32)

class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.net = nn.Sequential(
		nn.Linear(input_size, 64), 
		nn.ReLU(), 
		nn.Linear(64, 32), 
		nn.Linear(32, output_size),
		# nn.Softmax(),
		)

	def forward(self, x):
		pred = self.net(x)
		return pred 


nn_model = MLP(x_train.shape[1], 2)
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
loss_list = []

epoch_num = 100
for epoch in range(epoch_num):
  for x_batch, y_batch in train_dl:
      pred = nn_model(x_batch)
      loss = cost(pred, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  print('epoch %s, error %s'%(epoch, loss.item()))
  loss_list.append(loss.item())

nn_model.eval()
nn_yt = torch.log_softmax(nn_model(xtrain_t), dim=1).detach().numpy()
nn_y = torch.log_softmax(nn_model(xtest_t), dim=1).detach().numpy()
nn_yt = np.argmax(nn_yt, axis=1)
nn_y = np.argmax(nn_y, axis=1)

nn_auc_t = roc_auc_score(y_train, nn_yt)
nn_auc = roc_auc_score(y_test, nn_y)
print('nn: train %s, test %s'%(nn_auc_t, nn_auc))
nn_acc_t = accuracy_score(y_train, nn_yt)
nn_acc = accuracy_score(y_test, nn_y)
print('nn: train %s, test %s'%(nn_acc_t, nn_acc))
train_auc_list.append(nn_auc_t)
test_auc_list.append(nn_auc)
train_acc_list.append(nn_acc_t)
test_acc_list.append(nn_acc)

plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('Epcoh num', fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'nn_learning_curve'+'.png', dpi=100)
plt.close()


import xgboost as xgb
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
dvalid = xgb.DMatrix(x_val, label=y_val)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

params = {}
params['objective'] = 'multi:softprob'
params['eval_metric'] = 'mlogloss'
params['num_class'] = 2
params['max_depth'] = 10
params['eta'] = 0.1
params['min_child_weight'] = 20


print('Training XGBoost')
model = xgb.train(params, dtrain, 200, watchlist, early_stopping_rounds=50, verbose_eval=True)

xgb_yt = np.argmax(model.predict(dtrain), axis=1)
xgb_y = np.argmax(model.predict(dtest), axis=1)

xgb_auc_t = roc_auc_score(y_train, xgb_yt)
xgb_auc = roc_auc_score(y_test, xgb_y)
print('xgb: train %s, test %s'%(xgb_auc_t, xgb_auc))
xgb_acc_t = accuracy_score(y_train, xgb_yt)
xgb_acc = accuracy_score(y_test, xgb_y)
print('xbg: train %s, test %s'%(xgb_acc_t, xgb_acc))
train_auc_list.append(xgb_auc_t)
test_auc_list.append(xgb_auc)
train_acc_list.append(xgb_acc_t)
test_acc_list.append(xgb_acc)


model_list = ['lr', 'rf' , 'knn', 'svc', 'tree', 'nn', 'xgb']

plt.figure(figsize=(6,4))
plt.bar(np.arange(len(model_list)), train_auc_list, width=0.2,  color='lightblue', align='center', label='Train')
plt.bar(np.arange(len(model_list))-0.2, test_auc_list, width=0.2, color='y', align='center', label='Test')
plt.xticks(np.arange(len(model_list)), model_list, rotation=90)
# plt.xlabel('Models', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.ylim([0,1])
plt.legend(loc=2, fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'models_auc'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(6,4))
plt.bar(np.arange(len(model_list)), train_acc_list, width=0.2,  color='lightblue', align='center', label='Train')
plt.bar(np.arange(len(model_list))-0.2, test_acc_list, width=0.2, color='y', align='center', label='Test')
plt.xticks(np.arange(len(model_list)), model_list, rotation=90)
# plt.xlabel('Models', fontsize=12)
plt.ylim([0,1])
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc=2, fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'models_acc'+'.png', dpi=100)
plt.show()
