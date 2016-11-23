from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random

data0=np.loadtxt("a20161013_0425.tsv",delimiter=" ",usecols=(2,5))
data1=np.loadtxt("a20161013_0426.tsv",delimiter=" ",usecols=(2,5))
data2=np.loadtxt("a20161013_0427.tsv",delimiter=" ",usecols=(2,5))
data3=np.loadtxt("a20161013_0428.tsv",delimiter=" ",usecols=(2,5))
data4=np.loadtxt("a20161013_0429.tsv",delimiter=" ",usecols=(2,5))
data5=np.loadtxt("a20161014_0219.tsv",delimiter=" ",usecols=(2,5))
data6=np.loadtxt("a20161014_0220.tsv",delimiter=" ",usecols=(2,5))
data7=np.loadtxt("a20161014_0221.tsv",delimiter=" ",usecols=(2,5))
data8=np.loadtxt("a20161014_0222.tsv",delimiter=" ",usecols=(2,5))
data9=np.loadtxt("a20161014_0223.tsv",delimiter=" ",usecols=(2,5))
data10=np.loadtxt("a20161014_0224.tsv",delimiter=" ",usecols=(2,5))
data11=np.loadtxt("a20161014_0225.tsv",delimiter=" ",usecols=(2,5))
data12=np.loadtxt("a20161014_0226.tsv",delimiter=" ",usecols=(2,5))
data13=np.loadtxt("a20161014_0227.tsv",delimiter=" ",usecols=(2,5))
data14=np.loadtxt("a20161014_0228.tsv",delimiter=" ",usecols=(2,5))
data15=np.loadtxt("a20161014_0229.tsv",delimiter=" ",usecols=(2,5))
data16=np.loadtxt("a20161014_0230.tsv",delimiter=" ",usecols=(2,5))
data17=np.loadtxt("a20161014_0231.tsv",delimiter=" ",usecols=(2,5))
data18=np.loadtxt("a20161014_0232.tsv",delimiter=" ",usecols=(2,5))
data19=np.loadtxt("a20161014_0233.tsv",delimiter=" ",usecols=(2,5))
data20=np.loadtxt("a20161014_0234.tsv",delimiter=" ",usecols=(2,5))
data21=np.loadtxt("a20161014_0235.tsv",delimiter=" ",usecols=(2,5))

#data=np.loadtxt("cat_file.tsv",delimiter=" ")
test0=np.loadtxt("a20161014_0107.tsv",delimiter=" ",usecols=(2,5))
test1=np.loadtxt("a20161014_0201.tsv",delimiter=" ",usecols=(2,5))
true_result=np.loadtxt("a20161014_0107.tsv",delimiter=" ",usecols=(3,6))
#true_result1=np.loadtxt("a20161014_0201.tsv",delimiter=" ",usecols=(6))


online_data            = np.concatenate([data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19])
data     = np.concatenate([data20,data21])
test            = np.concatenate([test0])
len_data        = len(data)
len_test        = len(test)
len_online_data = len(online_data)
number_cluster  = 100

print("len_data=%d" %len_data)
print("len_test=%d" %len_test)
print("center number=%d" %number_cluster)
print("len_online_data=%d" %len_online_data)



colorlist    = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf','#9400D3','#7CFC00']
clusters     = number_cluster*[2*[0]]
test_datas   = number_cluster*[2*[0]]
centroids    = number_cluster*[2*[0]]
distance     = number_cluster*[0]
counter      = number_cluster*[0]
belongs		 = len_online_data*[0]
m_distance	 = 10000000

online_data=random.sample(online_data,len(online_data))
online_data=np.array(online_data)
t0 = time()
k_means = KMeans(n_clusters=number_cluster).fit(data)
print (time()-t0)

for i in range(number_cluster):
	clusters[i]=data[k_means.labels_==i] 
	counter[i]=len(clusters[i])

for i in range(number_cluster):
	centroids[i]=k_means.cluster_centers_[i,:]

for i in range(len_online_data):
 	for j in range(number_cluster):
		distance[j] = pow(online_data[i][1]-centroids[j][1],2)+pow(online_data[i][0]-centroids[j][0],2)

		if m_distance>=distance[j]:
			m_distance=distance[j]
			m_index=j
	
	belongs[i]=m_index	
	m_distance=10000000
	counter[m_index]+=1
	centroids[m_index]=centroids[m_index]+(online_data[i]-centroids[m_index])/counter[m_index]


ndarray_value=np.array(belongs)
print(counter)
print(ndarray_value)

test_k_means = k_means.predict(test)

#print(test_k_means)
#print(k_means.labels_)


for i in range(number_cluster):
	clusters[i]=online_data[ndarray_value==i]
	
#plt.ylim([-1.5,0.5])

for i in range(number_cluster):
	plt.scatter(clusters[i][:,1],clusters[i][:,0],color=colorlist[(i%10)],s=1)

"""
for i in range(number_cluster):
	plt.scatter(centroids[i][1],centroids[i][0],c='black',s=10,marker="^")
"""

for i in range(number_cluster):
	plt.hlines([min(clusters[i][:,0]),max(clusters[i][:,0])],min(clusters[i][:,1]),max(clusters[i][:,1]),linestyle="dashed")
	plt.vlines([min(clusters[i][:,1]),max(clusters[i][:,1])],min(clusters[i][:,0]),max(clusters[i][:,0]),linestyle="dashed")

count_normal=0
count_abnormal=0
count=0
decide_normal=len_test*[0]

for i in range(len_test):
		if test[i][0]>=min(clusters[test_k_means[i]][:,0]) and test[i][0]<=max(clusters[test_k_means[i]][:,0]) and test[i][1]>=min(clusters[test_k_means[i]][:,1]) and test[i][1]<=max(clusters[test_k_means[i]][:,1]):
			count_normal+=1
			count+=1
			decide_normal[i]=0
		else:
			count+=1
			count_abnormal+=1
			decide_normal[i]=1
			print(count)

print(count_normal)
print(count_abnormal)
#print(true_result[:,1])

tp=0
tn=0
fp=0
fn=0

for i in range(len_test):
	if true_result[i][1]==0 and decide_normal[i]==0:   
		tn+=1
	if true_result[i][1]==1 and decide_normal[i]==1:
		tp+=1
	if true_result[i][1]==0 and decide_normal[i]==1:
		fp+=1
	if true_result[i][1]==1 and decide_normal[i]==0:
		fn+=1

print("tp=%d" %tp)
print("tn=%d" %tn)
print("fp=%d" %fp)
print("fn=%d" %fn)

for i in range(len_test):
	if decide_normal[i]==1:
		plt.scatter(test[i][1],test[i][0],c='yellow')

#plt.show()
plt.savefig("hitachi5_online_training6000_k=100_1_10.png")

