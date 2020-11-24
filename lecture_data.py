import numpy as np 
import pandas as pd 
import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


data_path = '../data/'
output_path = '../mod_data/'

lec = pd.read_csv(data_path+'lectures.csv')
lec_columns = ['lec_id', 'lec_tag', 'lec_part', 'lec_type']
lec.columns = lec_columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lec['lec_type'] = le.fit_transform(lec['lec_type'].values)
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 20)
x = lec[['lec_type', 'lec_tag', 'lec_part']].values 
sc = StandardScaler()
x = sc.fit_transform(x)
clusters = km.fit(x).labels_
lec['lec_cluster'] = clusters
lec.to_csv(output_path+'lecture.csv', index=False)
