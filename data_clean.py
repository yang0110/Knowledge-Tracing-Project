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

train = pd.read_csv(output_path+'train.csv')
ques = pd.read_csv(output_path+'question.csv')
lec = pd.read_csv(output_path+'lecture.csv')

ques = ques.fillna(1)
ques_columns = ['ques_id', 'bundle_id', 'cor_answer', 'ques_part', 'wrong',
       'right', 'hard', 'easy', 'ques_cluster', 'tag_1', 'tag_2', 'tag_3',
       'tag_4', 'tag_5', 'tag_6']
ques = ques[ques_columns]

train_columns = ['row_id', 'user_id', 'content_id', 'content_type',
       'bundle_id', 'answer', 'explan', 'days', 'elapsed_days',
       'lecture', 'ability', 'correct']
train = train[train_columns]
train['ques_id'] = train['content_id'][train.content_type.values==0]
# train['lec_id'] = train['content_id'][train.content_type.values==1]
train = train.fillna(0)
train.ques_id = train.ques_id.astype(int)
# train.lec_id = train.lec_id.astype(int)
train = train.fillna(0)

train = train.merge(ques, on='ques_id', how='left')
# train = train.merge(lec, on='lec_id', how='left')

train = train.fillna(0)
train = train.drop('bundle_id_y', axis=1)

train_columns = ['user_id', 'content_id', 'content_type', 'bundle_id_x',
       'answer', 'explan', 'days', 'elapsed_days', 'lecture', 'ability',
        'ques_id',  'ques_part', 'wrong',
       'right', 'hard', 'easy', 'ques_cluster', 'tag_1', 'tag_2', 'tag_3',
       'tag_4', 'tag_5', 'tag_6', 'correct']

train = train[train_columns]
train  = train[train.correct !=-1]
train.to_csv(output_path+'combined_train.csv', index=False)

# train = train.sample(frac=1).reset_index(drop=True)
# x = train.iloc[:, 1:-1].values 
# y = train.iloc[:, -1].values

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(x_train, y_train)
# rf_yt = rf.predict(x_train)
# rf_y =rf.predict(x_test)

# auc_t = roc_auc_score(y_train, rf_yt)
# auc = roc_auc_score(y_test, rf_y)
# print('train %s, test %s'%(auc_t, auc))
# acc_t = accuracy_score(y_train, rf_yt)
# acc = accuracy_score(y_test, rf_y)
# print('train %s, test %s'%(acc_t, acc))
# cfm_t = confusion_matrix(y_train, rf_yt)
# cfm = confusion_matrix(y_test, rf_y)


# plt.matshow(cfm_t)
# plt.show()

# plt.matshow(cfm)
# plt.show()




