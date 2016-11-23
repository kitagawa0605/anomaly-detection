from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy.random as rd
from sklearn.datasets import make_blobs
import random
import numpy.random as rd


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

test0          = np.loadtxt("a20161014_0107.tsv",delimiter=" ",usecols=(2,5))
test1          = np.loadtxt("a20161014_0201.tsv",delimiter=" ",usecols=(2,5))
true_result    = np.loadtxt("a20161014_0107.tsv",delimiter=" ",usecols=(3,6))
#true_result1  = np.loadtxt("a20161014_0201.tsv",delimiter=" ",usecols=(6))
#data=data0
data           = np.concatenate([data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19])
test           = np.concatenate([test0])
#np.savetxt('cat_file_20.tsv',data,delimiter=' ')
len_data       = len(data)
len_test       = len(test)
number_cluster = 10			  #cluster number


data=random.sample(data,len(data))    #input randomly
data=np.array(data)

print("len_data=%d" %len_data)
print("len_test=%d" %len_test)
print("center_number=%d" %number_cluster)


colorlist        = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf','#9400D3','#7CFC00']

cdef int i,j
clusters         = number_cluster*[2*[0]]
test_datas       = number_cluster*[2*[0]]
centroids        = number_cluster*[2*[0]]
candidate_center = number_cluster*[0]
counter          = number_cluster*[0]
distance         = number_cluster*[0]
belongs			 = len_data*[0]
m_distance		 = 10000000
m_index=0

potential        = len_data*[0]
s_potential      = 0
center_id        = 0
center_order     = len_data*[0]
center_count     = 0
kyori            = number_cluster*[0]
m_kyori          = 100000000


t0 = time()
for i in range(len_data):			#initialized first center
	center_order[i]=i

center_id 		   = rd.choice(center_order)
candidate_center[0]=data[center_id]
center_count	  += 1

while center_count<number_cluster:        #select 2~k center
	for i in range(len_data):
		for j in range(center_count):
			kyori[j] = pow(data[i][1]-candidate_center[j][1],2)+pow(data[i][0]-candidate_center[j][0],2)

			if(m_kyori>=kyori[j]):
				m_kyori=kyori[j]
				potential[i]=kyori[j]

		m_kyori =10000000
	s_potential = sum(potential)

	for k in range(len_data):
		potential[k] = potential[k]/s_potential
	
	center_id		   			   = rd.choice(center_order,p=potential)
	candidate_center[center_count] = data[center_id]
	center_count				  += 1


#print(candidate_center)
#candidate_center=random.sample(data,number_cluster)



for i in range(len_data):
	for j in range(number_cluster):
		distance[j] = pow(data[i][1]-candidate_center[j][1],2)+pow(data[i][0]-candidate_center[j][0],2)

		if m_distance>=distance[j]:
			m_distance=distance[j]
			m_index=j
			
	belongs[i]=m_index
	m_distance=10000000
	counter[m_index]+=1
	candidate_center[m_index]=candidate_center[m_index]+(data[i]-candidate_center[m_index])/counter[m_index]

#print(candidate_center)
ndarray_value=np.array(belongs)
print(counter)
#print(ndarray_value)
print (time()-t0)

for i in range(number_cluster):
	clusters[i]=data[ndarray_value==i]

for i in range(number_cluster):
	plt.scatter(clusters[i][:,1],clusters[i][:,0],color=colorlist[(i%10)],s=0.5)
	plt.scatter(candidate_center[i][1],candidate_center[i][0],c='black',s=20,marker="^")

for i in range(number_cluster):
	plt.hlines([min(clusters[i][:,0]),max(clusters[i][:,0])],min(clusters[i][:,1]),max(clusters[i][:,1]),linestyle="dashed")
	plt.vlines([min(clusters[i][:,1]),max(clusters[i][:,1])],min(clusters[i][:,0]),max(clusters[i][:,0]),linestyle="dashed")


#plt.plot(data[0][1],data[0][0])
#plt.plot(data[1][1],data[1][0])

#plt.ylim([-1.2,0.5])
#plt.savefig("hitachi4_online_n=20sample_k=100.png")
print("plt end")