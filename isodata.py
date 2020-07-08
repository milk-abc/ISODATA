from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
iris=load_iris()
X=iris.data
y=iris.target
X_sepal=X[:,:2]
plt.scatter(X_sepal[:,0],X_sepal[:,1],c=y,cmap=plt.cm.gnuplot)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
theta_n=10#每一聚类中最少的样本数目
theta_s=0.12#一个聚类域中样本距离分布的标准差
theta_c=1.5#两个聚类中心之间的最小距离
L=2#每次迭代允许合并的最大聚类对数目
I=20#允许的最多迭代次数
k=3#初始聚类中心数目
split=0.4#分裂参数
#rj=split_k*thetaj#每个聚合的标准偏差向量(thetamax,0)
n=150
#Sodata(X,k,theta_n,theta_s,theta_c,L,I)
#numpy和list之间的问题
class Cluster:
    def __init__(self,nSamples,avgDist,center,std,data):
        self.nSamples=nSamples
        self.avgDist=avgDist
        self.center=center
        self.std=std
        self.data=data
class ClusterPair:
    def __init__(self,u:int,v:int,d:int):
        self.u=u
        self.v=v
        self.d=d
def findIdx(t,center):#找样本t属于哪个类
    ans=np.square(t-center[0]).sum()
    r=len(center)
    idx=0
    for i in range(1,r):
        tmp=np.square(t-center[i]).sum()
        if ans>tmp:
            idx=i
            ans=tmp
    return idx
def newcenter(N,Z):
    Z=np.array(Z)
    newZ=Z.sum(axis=0)/N
    return newZ
def indistance(X,newZ):
    newvalue=np.square(X-newZ)
    Dj=(np.sqrt(newvalue.sum(axis=1)).sum(axis=0))/len(X)
    return Dj
def stdvar(X,newZ):
    std=np.square(X-newZ).sum(axis=0)/len(X)
    D=np.sqrt(std)
    return D
def Sodata(dataset,k,Tn,Ts,Tc,L,I,split):
    split=0.4
    K=3#预期的聚类数目
    n=len(dataset)
    center=[]
    vec=[]
    for i in range(k):
        center.append(dataset[(i*50+np.random.randint(0,50))%150])
        c=Cluster(0,0,center[i],0,[])
        vec.append(c)  
    J=1
    isless=False
    while(True):
        for i in range(k):
            vec[i].nSamples=0
            vec[i].data=[]
        for d in dataset:
            
            idx=findIdx(d,center)
            vec[idx].data.append(d)
            vec[idx].nSamples+=1#来个列表把对应的分类的Data添加进去
        
        index=0#记录样本数目低于theta_N的index
        for i in range(k):
            if vec[i].nSamples<theta_n:
                isless=True
                index=i
                break
        if isless:
            k-=1
            center.pop(index)
            tmp=vec.pop(index)
            for d in tmp.data:
                idx=findIdx(d,center)
                vec[idx].data.append(d)
                vec[idx].nSamples+=1
            isless=False
        for i in range(k):
            vec[i].center=newcenter(vec[i].nSamples,vec[i].data)
            vec[i].avgDist=indistance(vec[i].data,vec[i].center)
            vec[i].std=stdvar(vec[i].data,vec[i].center)
        totalAvgDist=0
        for i in range(k):
            totalAvgDist+=vec[i].avgDist*vec[i].nSamples
        totalAvgDist/=n
        if J>=I:
            break
        if k<=K/2:
            maxsigma=[]
            for i in range(k):
                maxsigma.append(np.max(vec[i].std))
            print('maxsigma',maxsigma)
            for i in range(k):
                if maxsigma[i]>theta_s:
                    if (vec[i].avgDist>totalAvgDist and vec[i].nSamples>2*(theta_n+1)) or k<=K/2:
                        k=k+1
                        newc=Cluster
                        newc.center=vec[i].center-split*vec[i].std
                        vec.append(newc)
                        center.append(newc.center)
                        vec[i].center=vec[i].center+split*vec[i].std
                        center[i]=vec[i].center
                        break
        if (k>=2*K) or (J%2==0):
            info=[]
            count=0
            for i in range(k-1):
                for j in range(i+1,k):
                    distance=np.sqrt(np.square(vec[i].center-vec[j].center).sum())
                    if distance<theta_c:
                        info.append(ClusterPair(i,j,distance))
                        count+=1
            Dsort=sorted(info,key=lambda ClusterPair:ClusterPair.d)
            newD=Dsort[:L]
            cnt=0
            flag=[False for i in range(k)]
            dele=[False for i in range(k)]
            nTimes=0
            l=min(count,L)
            for i in range(l):
                u=newD[i].u
                v=newD[i].v
                if not flag[u] and not flag[v]:
                    nTimes+=1
                    if vec[u].nSamples<vec[v].nSamples:
                        flag[u]=True
                        dele[u]=True
                        for d in vec[u].data:
                            vec[v].data.append(d)
                        
                        vec[v].center=(vec[u].nSamples*vec[u].center+vec[v].nSamples*vec[v].center)/(vec[v].nSamples+vec[u].nSamples)
                        vec[v].nSamples+=vec[u].nSamples
                        center[v]=vec[v].center
                        newCenter.append(newclustercenter)
                    else:
                        flag[v]
                        dele[v]=True
                        for d in vec[v].data:
                            vec[u].data.append(d)
                        
                        vec[u].center=(vec[v].nSamples*vec[v].center+vec[u].nSamples*vec[u].center)/(vec[v].nSamples+vec[v].nSamples)
                        vec[u].nSamples+=vec[v].nSamples
                        center[u]=vec[u].center
                        newCenter.append(newclustercenter)
            for i in range(k):
                if(dele[i]):
                    vec.pop(i)
                    center.pop(i)
            k-=nTimes
            flag=None
            dele=None
        if J>=I:
            break
        J+=1
    print('最终类别数',len(vec),'第一类',len(vec[0].data),'第二类',len(vec[1].data),'第三类',len(vec[2].data))
    for i in range(k):
        print('第',i,'个聚类是','\n')
        print(vec[i].data)

Sodata(X,k,theta_n,theta_s,theta_c,L,I,split)   
