import lrLMM
from dataGeneration import dataGeneration
import numpy as np
import AdaptiveLasso
from NeighborhoodSelection import neighborhoodSelection_singleNode
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib import pyplot as plt
np.random.seed(1)

num_cluster = 1
p = 1000
X, Ps = dataGeneration(seed=10, n=500, p=p, num_cluster=num_cluster, k=2, visualFlag=False, quiet=False)
X = np.asmatrix(X)

for i in range(p):
   for j in range(p):
       if(Ps[0, i, j] != 0):
           Ps[0, i, j] = 1
#print(Ps[0])
np.savetxt('../sample.csv', X, delimiter = ',')
np.savetxt('../precision.csv', Ps[0])
X = np.genfromtxt('../sample.csv', delimiter=',')

samples = X.shape[0]
features = X.shape[1]

print(X.shape)
#print(Ps.shape)

model = LowRankLinearMixedModel()
#ada = AdaLasso(method='Uni')
 
l0 = np.zeros(samples)
c = 0
for i in range(samples):
    l0[i] = c

K = np.dot(X,X.T)
#print(K.shape)

Result = np.zeros(shape = (features, features))

#print(y.shape)
for i in range(features):
    y = X[:, i]
    SUX = model.train(X,K,Kva=None,Kve=None,y=y,mode = 'non-linear')
    rs = neighborhoodSelection_singleNode(SUX, l0, i, 3, 1)
    Rs = np.asmatrix(rs)
    Result[i] = Rs
    '''
    beta = np.asmatrix(beta)
    beta = beta.T
    beta[i] = 1
    beta[beta!=0] = 1
    #print(beta)
    Result[i] = beta.T
    '''


#post analysis
#print(Result.shape)
for i in range(Result.shape[0]):
    for j in range(Result.shape[1]):
        if(Result[i][j] == 1):
            Result[j][i] = 1
#print(Result)


#visualization
'''
G = nx.Graph()
fig = plt.figure(figsize=(12,12))
list_nodes = [node for node in range(features)]
G.add_nodes_from(list_nodes)

edge = []
edge_lists = np.copy(Result)
for i in range(features):
    for j in range(features):
        if(edge_lists[i][j] == 1):
            edge.append([i,j])

G.add_edges_from(edge)

#nx.draw_networkx((G), pos= nx.circular_layout(G), node_size = 50)
#nx.draw_networkx((G), pos= nx.spring_layout(G), node_size = 50)
'''

#validation
tp = fn = fp = tn = 0

P = np.genfromtxt('month_precision.csv')
#print(P.shape)
for i in range(features):
    for j in range(features):
        if(Result[i][j] == 1 and P[i][j] == 1):
            tp+=1
        if(Result[i][j] == 0 and P[i][j] == 1):
            fn+=1
        if(Result[i][j] == 1 and P[i][j] == 0):
            fp+=1
        if(Result[i][j] == 0 and P[i][j] == 0):
            tn+=1
            
print(tp, fn, fp, tn)
TPR = tp/(tp + fn)
FPR = fp/(fp + tn)
print(TPR)
print(FPR)
