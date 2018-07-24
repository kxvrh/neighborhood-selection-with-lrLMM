
__author__ = 'Haohan Wang'

import numpy as np
import scipy
np.set_printoptions(suppress=True)

def randomPermutate(M, threshold=0.9):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.random.random() > threshold:
                M[i,j] = np.abs(1-M[i,j])
    return M

def is_pos_def(x):
    try:
        if np.all(np.linalg.eigvals(np.linalg.inv(x)) > 0):
            return True
        else:
            return False
    except:
        return False

    # return np.all(np.linalg.eigvals(x) > 0)

def increaseCount(count):
    #return count/2.0
    return count*2.0

def fillPrecisionMatrix_StarNetwork(P, k):
    [p, p] = P.shape
    # a = np.random.choice(range(0, p), 1, replace=False)[0]
    a = 0
    ml = np.random.choice(range(0, a) + range(a+1, p), k-1, replace=False)
    for m in ml:
        # r = np.random.random()
        r = 1
        P[a, m] = r
        P[m, a] = r
    return P

def fillPrecisionMatrix_AverageEdge(P, k):
    # todo: check standard method to do this
    [p, p] = P.shape
    for a in range(p):
        ml = np.random.choice(range(0, a) + range(a+1, p), k-1, replace=False)
        P[a, ml] = 1
        P[ml, a] = 1
    return P

def fillPrecisionMatrix_Predefined(P, p):
    '''
    if i == 0:
        P[0, 2] = 0.5
        P[2, 0] = 0.5
        P[1, 3] = 0.5
        P[3, 1] = 0.5  
    elif i == 1:
        P[0, 3] = 0.5
        P[3, 0] = 0.5
        P[1, 2] = 0.5
        P[2, 1] = 0.5
    '''
    '''
    P[0, 2] = 0.5
    P[2, 0] = 0.5
    P[1, 3] = 0.5
    P[3, 1] = 0.5
    P[p-1, p-3] = 0.5
    P[p-3, p-1] = 0.5
    P[p-4, p-2] = 0.5
    P[p-2, p-4] = 0.5
    '''
    
    
    d = int(p*p*0.0025/2)
    for x in range(d):
        b = np.random.randint(0,p)
        a = np.random.randint(0,p)
        P[a, b] = 0.5
        P[b, a] = 0.5
    
    return P

def dataGeneration(seed, n=500, p=1000, num_cluster=5, k=10,  visualFlag=True, quiet=False):
    np.random.seed(seed)
    plt = None
    if visualFlag:
        from matplotlib import pyplot as plt
    s = n/num_cluster
    I = np.zeros([p, p])
    X = None
    for j in range(p):
        I[j,j] = 1
    Ps = []

    # k0 = k/2
    P = np.zeros([p, p])
    P = fillPrecisionMatrix_Predefined(P, p)
    indexMapping = {}

    for i in range(num_cluster):
        if not quiet:
            print ('cluster', i, 'checking with coefficient:',)
        #P = np.zeros([p, p])

        # # this is a more careful data generation procedure, but less variant
        # for j in range(p-k0):
        #     if len(np.where(P[j,:]!=0)[0]) < k0:
        #         ind = np.random.choice(range(j+1, p), k0 - len(np.where(P[j,:]!=0)[0]), replace=False)
        #         r = np.random.random()
        #         P[j, ind] = r
        #         P[ind, j] = r

        # P = fillPrecisionMatrix_AverageEdge(P, k)
        #P = fillPrecisionMatrix_Predefined(P, i, p)

        count = 1
        t = increaseCount(count)
        if not quiet:
            print (t),
        P = P + np.diag(np.ones(p)*(t))
        while not is_pos_def(P):
            P = P - np.diag(np.ones(p)*(t))
            count += 1
            t = increaseCount(count)
            if not quiet:
                print (t),
            P = P + np.diag(np.ones(p)*(t))
        if not quiet:
            print ('final connections', len(np.where(P!=0)[0]))
        # for j in range(p):
        #     print len(np.where(P[j,:]!=0)[0])

        C = np.linalg.inv(P)
        if visualFlag:
            plt.imshow(P)
            plt.show()
            plt.imshow(C)
            plt.show()

        # x = np.random.multivariate_normal(np.zeros(p), C, size=s)

        x = []
        for t in range(1, int(s+1)):
            tmpx = np.random.multivariate_normal(np.random.normal(size=p)*t, C, size=1).reshape([p])
            x.append(tmpx)
        x = np.array(x)

        if i == 0:
            X = x
        else:
            X = np.append(X, x, 0)

        for j in range(int(s)):
            indexMapping[j+s*i] = i
        Ps.append(P)
        #print(np.array(Ps).shape)

    if visualFlag:
        plt.imshow(X)
        plt.show()

    return X, np.array(Ps)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    np.random.seed(1)

    num_cluster = 1
    p = 20
    X, Ps = dataGeneration(seed=1, n=1000, p=p, num_cluster=num_cluster, k=2, visualFlag=True, quiet=False)
    X = np.asmatrix(X)
    print(X.shape)
    print(Ps.shape)
    for i in range(p):
        for j in range(p):
            if(Ps[0, i, j] != 0):
                Ps[0, i, j] = 1
    #print(Ps[0])
   
    # plt.imshow(X)
    # plt.show()
    # from sklearn.cluster import KMeans
    # km = KMeans(n_clusters=num_cluster)
    # l = km.fit_predict(X)
    # print (l)

    # K = np.dot(X, X.T)
    # plt.imshow(K)
    # plt.show()
    #
    # C = np.dot(X.T, X)
    # plt.imshow(C)
    # plt.show()
    #
    # km = KMeans(n_clusters=num_cluster)
    # print (Ps)
    # l = km.fit_predict(Ps)
    # print (l)

