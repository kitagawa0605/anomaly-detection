from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


data0=np.loadtxt("a20161013_0425.tsv",delimiter=" ",usecols=(2,5))
data1=np.loadtxt("a20161013_0426.tsv",delimiter=" ",usecols=(2,5))
data2=np.loadtxt("a20161013_0427.tsv",delimiter=" ",usecols=(2,5))
data3=np.loadtxt("a20161013_0428.tsv",delimiter=" ",usecols=(2,5))
data4=np.loadtxt("a20161013_0429.tsv",delimiter=" ",usecols=(2,5))

test0=np.loadtxt("a20161014_0107.tsv",delimiter=" ",usecols=(2,5))
test1=np.loadtxt("a20161014_0201.tsv",delimiter=" ",usecols=(2,5))
true_result=np.loadtxt("a20161014_0107.tsv",delimiter=" ",usecols=(3,6))
#true_result1=np.loadtxt("a20161014_0201.tsv",delimiter=" ",usecols=(6))


data=np.concatenate([data0,data1,data2,data3,data4])
test=np.concatenate([test0])
len_test=len(test)
print(len_test)


colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf','#9400D3','#7CFC00']
clusters=10*[2*[0]]
test_datas=10*[2*[0]]
centroids=10*[2*[0]]


t0 = time()
k_means = KMeans(n_clusters=10).fit(data)
print (time()-t0)

print(k_means.fit_predict(data))
print(k_means.labels_)

#print(k_means.fit_predict(data))
#print(k_means.fit_predict(data))

test_k_means = k_means.predict(test)

print(test_k_means)


for i in range(10):
	clusters[i]=data[k_means.labels_==i]    #cluster[0][0][0] is cluster0 of y clusters[0][0][1] is cluster0 of x
	#test_datas[i]=test[test_k_means==i]

for i in range(10):
	centroids[i]=k_means.cluster_centers_[i,:]

#plt.ylim([-1.5,0.5])

for i in range(10):
	plt.plot(clusters[i][:,1],clusters[i][:,0],color=colorlist[(i%10)])
	plt.scatter(centroids[i][1],centroids[i][0],c='black',s=40,marker="^")

for i in range(10):
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




plt.plot(test[:,1],test[:,0],c='black',linewidth=2)
#plt.show()
plt.savefig("hitachi7.png")



