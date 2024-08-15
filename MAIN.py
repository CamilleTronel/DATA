
import pandas
import numpy
R=pandas.read_csv('C:\DATA_EXAMPLE\DATA_EXAMPLE.txt')
Time=[]
Value=[]
N=len(R)
flag=True
k=1
Q=list([])
M=list([])
while flag:
    if k+1==N:
        flag=False
    M.append([float(x) for x in R.loc[k-1][0].split(';')[1:-1]])
    if int(R.loc[k][0][0])!=int(R.loc[k-1][0][0]):
        Q.append(numpy.array(M))
        M=list([])
    k=k+1
M.append([float(x) for x in R.loc[N-1][0].split(';')[1:-1]])
Q.append(numpy.array(M))
### PRE
import numpy
import math

### NORM
import numpy
def invnorm(X,V):
    return X.mean()*numpy.array([(V[k]+1/X.mean()*X[k].var()) for k in range(0,X.shape[0])])
def norm(X):
    return 1/X.mean()*numpy.array([(X[k]-X[k].var()) for k in range(0,X.shape[0])])

def p1(m):
    return numpy.matmul(m,m.transpose())
def p2(m):
    return numpy.linalg.inv(m)
def p3(m):
    return numpy.diag(numpy.linalg.eig(m)[0])
def p3bis(m):
    return numpy.linalg.eig(m)[0]
def p00(m):
    return numpy.linalg.eig(m)[1]
def p11(m):
    return numpy.invert(numpy.linalg.eig(m)[1])

fonction =lambda x,theta: abs(numpy.exp(1j*theta*x))
class account():
    def rotate(theta,v,d):
        return numpy.diag(v)*numpy.array([fonction(1/len(numpy.diag(v))*(2*k*math.pi+d),theta) for k in range(len(numpy.diag(v)))])
    def cone(theta,v):
        return numpy.diag(v)*numpy.array([fonction(1/len(numpy.diag(v))*(2*k*math.pi),theta) for k in range(len(numpy.diag(v)))])
    def sort(v):
        return numpy.sort(v)


##
ld=list([])
ltheta=list([])
def mpool(v):
    N=v.size
    theta=numpy.random.uniform(-1,1)
    d=numpy.random.binomial(N,1/2)
    ld.append(d)
    ltheta.append(theta)
    return 1/N*sum([account.rotate(theta,norm(v),d) for k in range(0,N,1)])
    
def npool(v):
    N=v.size
    return 1/N*sum([account.rotate(numpy.random.uniform(-1,1),v,numpy.random.binomial(N,1/2)) for k in range(0,N,1)])
k=16

def p(m1):
    m2=p2(m1)
    v=p3(m2)
    return mpool(v)
def qp(m1):
    m2=p2(m1)
    return numpy.diag(numpy.linalg.eig(m2)[0])
##
N=16
LIST=list([])
def q(m0):
    N=int(math.sqrt(m0.size))
    m1=p1(m0)
    v=numpy.diag(p(m1.copy()))
    for k in range(0,N,1):
        m1=p1(v)
        LIST.append(qp(m1))
        v=p(m1.copy())
        v=numpy.diag(p(numpy.diag(v.copy())))
    return v
def invq(m0):
    R=numpy.array([])
    for k in range(N,0,-1):
        MATRICE=LIST[k]
        IMATRICE=numpy.inv(MATRICE)
        R=numpy.matmul(IMATRICE,R.copy(),MATRICE)
    return R
T=0
Q1=list([])
for M in Q:
    T+=1
    Q1.append(q(M))
Q2=list([])
for M in Q1:
    Q2.append(numpy.diag(M))

### BACKWARD
import numpy
import math
g = numpy.vectorize(lambda x :  1/math.pi*2*1/(1+pow(x,2)))
def dE(X,theta):
    return g(sum(numpy.multiply(numpy.transpose(theta),numpy.transpose(X))))*theta
    
k=10
N=len(Q2[0]) ##formule ??
M0=min(16,len(Q2))
Q3=numpy.array(Q2).reshape([M0,M0])
def theta_init():
    return numpy.random.uniform(-1,1,[N,M0])
def grad(X):
    theta=theta_init()
    for z in range(0,k,1):
        print(theta)
        theta=theta.copy()-dE(X,theta.copy())
    return theta
### FORWARD
import math
import numpy
f = numpy.vectorize(lambda x : 1/math.pi*2*math.atan(x))
def forward(X,theta):
    return f(sum(numpy.multiply(numpy.transpose(theta),numpy.transpose(X))))

Q0=math.floor(16/(T+1))
FONCTION_NORME = numpy.vectorize(lambda x : math.floor(x))
### TRAINING
def Z(V):
    X=V
    theta=grad(X)
    return numpy.array([FONCTION_NORME(forward(X,theta)) for i in range(1,Q0,1)])  
## EL TREE (VIZ)
import numpy
###
def gini(X,k,n):
    x=X[:][k]
    y=X[:][n]
    count=numpy.array([[0,0],[0,0]])
    for i in range(0,len(x),1):
        for j in range(0,len(y),1):
            if x[i]!=1:
                if y[i]!=1:
                    count[1][1]+=1
            if x[i]!=1:
                if y[j]!=0:
                    count[1][0]+=1
            if x[i]!=0:
                if y[j]!=1:
                    count[0][1]+=1
            if x[i]!=0:
                if y[j]!=0:
                    count[0][0]+=1
    return abs(count[0][0]/(count[1][0]+count[0][0]))*(1-abs(count[1][0]/(count[1][0]+count[0][0]))-abs(count[0][0]/(count[1][0]+count[0][0])))+abs(count[0][1]/(count[1][1]+count[0][1]))*(1-abs(count[1][1]/(count[1][1]+count[0][1]))-abs(count[0][1]/(count[1][1]+count[0][1])))
def partgini(X,r,n):
    x=X[:][r]
    y=X[:][n]
    count=numpy.array([[0,0],[0,0]])
    for i in range(0,len(x),1):
        for j in range(0,len(y),1):
            if x[i]!=1:
                if y[i]!=1:
                    count[1][1]+=1
            if x[i]!=1:
                if y[j]!=0:
                    count[1][0]+=1
            if x[i]!=0:
                if y[j]!=1:
                    count[0][1]+=1
            if x[i]!=0:
                if y[j]!=0:
                    count[0][0]+=1
        return (1-abs(count[1][0]/(count[1][0]+count[0][0]))),(1-abs(count[1][1]/(count[1][1]+count[0][1]))-abs(count[0][1]/(count[1][1]+count[0][1])))
def group(X,i):
    R=list([])
    for p in range(0,len(X[:]),1):
        if X[p][i]==1:
            R.append(X[:][p])
    return numpy.array(R)
def func(k,n,X,i):
    return gini(group(X,i),k,n)
##
def treelearn(X):
    s=[]
    i,j=numpy.random.binomial(T-1,1/2),numpy.random.binomial(T-1,1/2)
    s.append(i)
    s.append(j)
    g,d=partgini(group(X,i),i,j)
    for q in range(0,len(X),1): 
        if g==0:
            h=0
            G=0
            for k in range(len(X)):
                if not(k in s):
                    if gini(X,j,k)>G:
                        h=k
            index=s.pop(-1)
            s.append(h)
        if d==0:
            h=0
            G=0
            for k in range(len(X)):
                if not(k in s):
                    if gini(X,j,k)>G:
                        h=k
            index=s.pop(-2)
            s.append(h)
        g,d=partgini(group(X,h),h,index)
        return s
def learn_all():
    E=Z(Q3)
    S=treelearn(E)
    return S
def tree_forward(X,V):
    R=[]
    for s in range(0,len(X),1):
        for q in range(0,len(V),1):
            if X[s]>=V[q]:
                R.append(V[q])
    return R
def predict(X,V):
    RESULT=list([])
    for ligne in X:
        RESULT.append(tree_forward(X[0],V))
    return invq(RESULT)
