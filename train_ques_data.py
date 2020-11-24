import numpy as np 
import pandas as pd 
import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



data_path = '../data/'
output_path = '../mod_data/'

N = 500000
train = pd.read_csv(data_path + 'train.csv',nrows=N)
ques = pd.read_csv(data_path+'questions.csv')

train_columns = ['row_id', 'timestamp', 'user_id', 'content_id', 'content_type',
       'bundle_id', 'answer', 'correct',
       'elapsed_time', 'explan']

ques_columns= ['ques_id', 'bundle_id', 'cor_answer', 'ques_part', 'ques_tags']

train.columns = train_columns
ques.columns = ques_columns

train = train[['row_id',  'user_id', 'timestamp', 'content_id', 'content_type',
       'bundle_id', 'answer', 
       'elapsed_time', 'explan', 'correct']]

train['days'] = train.timestamp.values / (31536000000/365)
train['elapsed_days'] = train.elapsed_time.values/(31536000000/365)
train['explan'] = train.explan.astype('bool')
train['lecture'] = 1.0*(train['content_type'].values == 1)

train['ability'] = 0
unique_users = np.unique(train.user_id.values)
for user in unique_users:
	user_train = train[train.user_id.values==user]
	num = user_train.shape[0]
	correct = len(user_train[user_train.correct ==1])
	ratio = correct/num 
	train.loc[train.user_id==user, 'ability'] = ratio

train_columns = ['row_id', 'user_id', 'timestamp', 'content_id', 'content_type',
       'bundle_id', 'answer', 'elapsed_time', 'explan',  'days',
       'elapsed_days', 'lecture', 'ability', 'correct']
train = train[train_columns]
train.to_csv(output_path+'train.csv', index=False)


tags_list = [x.split() for x in ques.ques_tags.values.astype(str)]
ques['ques_tags'] = tags_list

correct = train[train.correct != -1].groupby(["content_id", 'correct'], as_index=False).size()
correct = correct.pivot(index= "content_id", columns='correct', values='size')
correct.columns = ['wrong', 'right']
correct = correct.fillna(0)
correct[['wrong', 'right']] = correct[['wrong', 'right']].astype(int)
ques = ques.merge(correct, left_on = "ques_id", right_on = "content_id", how = "left")
ques['hard'] = ques.wrong/(ques.wrong+ques.right)
ques['easy'] = 1- ques.hard
ques.head()

ques.ques_tags.fillna(-1, inplace=True)
all_tags = []
for x in ques.ques_tags.values:
	all_tags += x

unique_tags = np.unique(all_tags)
ques_num = len(ques)
tag_matrix = np.zeros((ques_num, len(unique_tags)+1))
for i in range(ques_num):
	print('user id %s'%(i))
	tags = ques.ques_tags.values[i]
	for t in tags:
		if t == 'nan':
			x = len(unique_tags)
		else:
			x = int(t)
		tag_matrix[i, x] = 1.0

matrix = np.hstack((tag_matrix, ques.ques_part.values.reshape(-1,1), ques.bundle_id.values.reshape(-1,1)))


sc = StandardScaler()
matrix = sc.fit_transform(matrix)
km = KMeans(n_clusters = 100)
ques_clusters = km.fit(matrix).labels_
ques['ques_cluster'] = ques_clusters


tag_matrix = np.zeros((ques_num, 6))
for i in range(ques_num):
	tags = ques.ques_tags.values[i]
	tag_num = len(tags)
	print('tag_num', tag_num)
	for j in range(tag_num):
		tag = tags[j]
		if tag == 'nan':
			pass 
		else:
			tag = int(tag)
		tag_matrix[i, j] = tag

ques[['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6']]  = tag_matrix
ques.to_csv(output_path+'question.csv', index=False)
