from math import sqrt
import pandas as pd
import numpy as np
import os
import time

customer_file = 'smallsize.csv'
w1 = 0.33
w2 = 0.33
w3 = 0.33
num_vechile = 2
num_nodes = 8
num_jobs = 6
M = 1000#当数据集为c201的时候改大点
df_customer = pd.read_csv(customer_file)
X = df_customer['X'].values
Y = df_customer['Y'].values
distMat = np.zeros((len(X),len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        distMat[i][j] = sqrt((X[i]-X[j])*(X[i]-X[j])+(Y[i]-Y[j])*(Y[i]-Y[j]))
d_max = distMat.max()
GIR = np.asarray(df_customer[['gir1','gir2','gir3','gir4']])
F = np.asarray(df_customer[['FdiL','FdiM','FdiMa','FdiJ','FdiV','FdiS','FdiD']])

s_time = np.asarray(df_customer[['L','M','Ma','J','V','S','D']])
e_time = np.asarray(df_customer[['startL','startM','startMa','startJ','startV','startS','startD']])
l_time = np.asarray(df_customer[['endL','endM','endMa','endJ','endV','endS','endD']])
arcs = [(x1, x2) for x1 in V for x2 in V if x1!=x2] #arcs
samelocSet = [[2],[3,4],[5,6],[7]]
Ya = 5
Yb = 5
a1= 1
a2 = 3

K = [k for k in range(num_vechile) ] #vehicles
V = [i for i in range(num_nodes) ]#all vetrice
N = [i for i in range(2,num_nodes) ]#patients
D = [d for d in range(0,7)]# planning days one week
DC = [1]
O = [0,1]
C = [0,1,2,3]
P = [i for i in range(len(samelocSet))]

G = [0,1,2,3,4]
H = [0,1,2,3]
from gurobipy import Model, GRB, quicksum, max_
start = time.time()
# Model
mdl = Model('LTRSP')
# decision variables
x = mdl.addVars(V,V, K, D,vtype=GRB.BINARY, name='x')
a = mdl.addVars(V, K, D, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='arrivaltime')
b = mdl.addVars(V, K, D, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='departtime')
z = mdl.addVars(V,D, vtype=GRB.BINARY)
w = mdl.addVars(K,D,vtype=GRB.CONTINUOUS,lb=0,ub=GRB.INFINITY)
gN = mdl.addVars(K,D,C,vtype=GRB.INTEGER)
Wt = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
W0 = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
W1 = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
W2 = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
W3 = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
q = mdl.addVars(N,K,vtype=GRB.BINARY)
qN = mdl.addVars(N,vtype=GRB.INTEGER)
u = mdl.addVars(N,G,D,vtype=GRB.BINARY)
v = mdl.addVars(N,H,D,vtype=GRB.BINARY)
wa = mdl.addVars(N,K,D,vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
wb = mdl.addVars(N,K,D,vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
pa = mdl.addVars(N,K,D,vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
pb = mdl.addVars(N,K,D,vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)


#objective
mdl.modelSense = GRB.MINIMIZE
mdl.setObjective(w1*(quicksum(x[i, j, k, d] * distMat[i, j] for i in V for j in V for k in K for d in D)) + w2*(Wt+W0+W1+W2+W3)
                 +w3*(quicksum(pa[i,k,d]for i in N for k in K for d in D)+quicksum(pb[i,k,d]for i in N for k in K for d in D)+ quicksum(qN[i] for i in N)))
# constraints
# flow constraints
mdl.addConstrs(quicksum(x[i, j, k, d] for k in K for i in V if i != j) == F[i][d] for j in N for d in D)
mdl.addConstrs(quicksum(x[0,j,k,d] for j in N) + quicksum(x[i,j,k,d] for i in DC for j in N) <=1 for k in K for d in D)
mdl.addConstrs(quicksum(x[i,0,k,d] for i in N) ==1 for k in K for d in D)
mdl.addConstrs(quicksum(x[i,j,k,d] for j in V if i != j) == quicksum(x[j,i,k,d] for j in V if i != j) for i in N for k in K for d in D)
#time window
mdl.addConstrs(b[i,k,d]==0 for i in O for k in K for d in D)
mdl.addConstrs(b[i,k,d]>=a[i,k,d] + s_time[i,d] for i in N for k in K for d in D )
mdl.addConstrs(b[i,k,d]>=e_time[i,d] + s_time[i,d] for i in N for k in K for d in D )
mdl.addConstrs(b[i,k,d]<=a[i,k,d] + s_time[i,d] + (1-z[i,d])*M for i in N for k in K for d in D)
mdl.addConstrs(b[i,k,d]<=e_time[i,d] + s_time[i,d] + z[i,d]*M for i in N for k in K for d in D)
#mdl.addConstrs((x[i, j, k,d] == 1) >> (a[i, k,d] == distMat[i, j]) for i in O for k in K for d in D)
mdl.addConstrs((x[i,j,k,d]==1)>> (a[j,k,d]==b[i,k,d]+distMat[i,j]) for i in V for j in V for k in K for d in D)

#workload balance
mdl.addConstrs(w[k,d]==quicksum(x[i,j,k,d]*s_time[i,d] for i in N for j in V if i!=j) for k in K for d in D)
mdl.addConstrs(gN[k,d,c]==quicksum(x[i,j,k,d]*GIR[i,c] for i in N for j in V if i!=j) for c in C for k in K for d in D)
mdl.addConstrs(Wt >= quicksum(w[k,d] for d in D)-quicksum(w[l,d] for d in D) for k in K for l in K if k !=l)
mdl.addConstrs(W0 >= quicksum(gN[k,d,0] for d in D)-quicksum(gN[l,d,0] for d in D) for k in K for l in K if k!=l)
mdl.addConstrs(W1 >= quicksum(gN[k,d,1] for d in D)-quicksum(gN[l,d,1] for d in D) for k in K for l in K if k!=l)
mdl.addConstrs(W2 >= quicksum(gN[k,d,2] for d in D)-quicksum(gN[l,d,2] for d in D) for k in K for l in K if k!=l)
mdl.addConstrs(W3 >= quicksum(gN[k,d,3] for d in D)-quicksum(gN[l,d,3] for d in D) for k in K for l in K if k!=l)

#continouity of care
mdl.addConstrs(quicksum(x[i,j,k,d] for j in V)<=q[i,k] for i in N for k in K for d in D)
mdl.addConstrs(q[i,k]<=quicksum(x[i,j,k,d] for j in V for d in D) for i in N for k in K)
mdl.addConstrs(qN[i]==quicksum(q[i,k] for k in K) for i in N)

#penalty of arrival time
mdl.addConstrs(quicksum(u[i,g,d] for g in G)==1 for i in N for d in D)
mdl.addConstrs(a[i,k,d]<=(e_time[i,d]-30)*u[i,0,d]+(e_time[i,d]-15)*u[i,1,d]+e_time[i,d]*u[i,2,d]+l_time[i,d]*u[i,3,d]+M*u[i,4,d] for i in N for k in K for d in D)
mdl.addConstrs(a[i,k,d]>=(e_time[i,d]-30)*u[i,1,d]+(e_time[i,d]-15)*u[i,2,d]+e_time[i,d]*u[i,3,d]+l_time[i,d]*u[i,4,d] for i in N for k in K for d in D)
mdl.addConstrs(wa[i,k,d]== Ya*u[i,0,d]+ a1*u[i,1,d] + a2*u[i,2,d] + 0*u[i,3,d]+Ya*u[i,4,d] for k in K for d in D)
mdl.addConstrs(pa[i,k,d]<=wa[i,k,d]+M*(1-quicksum(x[i,j,k,d] for j in V if i!=j)) for i in N for k in K for d in D)
mdl.addConstrs(pa[i,k,d]>=wa[i,k,d]-M*(1-quicksum(x[i,j,k,d] for j in V if i!=j)) for i in N for k in K for d in D)
mdl.addConstrs(pa[i,k,d]<=wa[i,k,d]+M*quicksum(x[i,j,k,d] for j in V if i!=j) for i in N for k in K for d in D)
#mdl.addConstr((quicksum(x[i,j,k,d] for j in V if j!=i)==1 )>>(pa[i,k,d]==wa[i,k,d]) for i in N for k in K for d in D)
#mdl.addConstr((quicksum(x[i,j,k,d] for j in V if j!=i)==0)>>(pa[i,k,d]==0) for i in N for k in K for d in D)

#penalty of service end time
mdl.addConstrs(quicksum(v[i,h,d] for h in H)==1 for i in N for d in D)
mdl.addConstrs(b[i,k,d]<=l_time[i,d]*v[i,0,d]+(l_time[i,d]+15)*v[i,1,d]+(l_time[i,d]+30)*v[i,2,d]+M*v[i,3,d] for i in N for k in K for d in D)
mdl.addConstrs(b[i,k,d]>=l_time[i,d]*v[i,1,d]+(l_time[i,d]+15)*v[i,2,d]+(l_time[i,d]+30)*v[i,3,d] for i in N for k in K for d in D)
mdl.addConstrs(wb[i,k,d]== 0*v[i,0,d]+ a1*v[i,1,d] + a2*v[i,2,d] + Yb*v[i,3,d] for k in K for d in D)
mdl.addConstrs(pb[i,k,d]<=wb[i,k,d]+M*(1-quicksum(x[i,j,k,d] for j in V if i!=j)) for i in N for k in K for d in D)
mdl.addConstrs(pb[i,k,d]>=wb[i,k,d]-M*(1-quicksum(x[i,j,k,d] for j in V if i!=j)) for i in N for k in K for d in D)
mdl.addConstrs(pb[i,k,d]<=wb[i,k,d]+M*quicksum(x[i,j,k,d] for j in V if i!=j) for i in N for k in K for d in D)
#mdl.addConstrs((quicksum(x[i,j,k,d] for j in V if j!=i)==1)>>(pb[i,k,d]==wb[i,k,d]) for i in N for k in K for d in D)
#mdl.addConstrs((quicksum(x[i,j,k,d] for j in V if j!=i)==0)>>(pb[i,k,d]==0) for i in N for k in K for d in D)


mdl.Params.MIPGap = 0.1
mdl.optimize()
end = time.time()
print(end-start)

route = []
for k in K:
    unsort_subroute = []
    sort_subroute = []
    for i in V:
        for j in V:
            if x[i, j, k].x > 0.99:
            # insert
            unsort_subroute.append((i, j))

    if len(sort_subroute) == 0:
        sort_subroute.append(unsort_subroute[0][0])
        sort_subroute.append(unsort_subroute[0][1])
    unsort_subroute.pop(0)




