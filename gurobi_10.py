import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
# read data from excel
# parameters ei, li, si, cij=tij,requ,qu
DataSet = 'rc201'
customer_file = r'D:\PycharmProjects\vrptw_test\solomondata\solomon10_csv\customers\%scustomers.csv'%(DataSet)
distanceMatrix_file =r'D:\PycharmProjects\vrptw_test\solomondata\solomon10_csv\distance_matrix\%scustomers.csv'%(DataSet)
save_obj = r'D:\PycharmProjects\vrptw_test\result_new'
save_sol = r'D:\PycharmProjects\vrptw_test\result_sol'
df_customers = pd.read_csv(customer_file)# allnode
requ = df_customers['level'].values#[0 1 2 1 2 1 1 1 1 2 1 1 1 2 1 2 2 1 1 1 1 1 1 1 2 1]
e_time = np.asarray(df_customers['readyTime'])
s_time = np.asarray(df_customers['treatTime'])
l_time = np.asarray(df_customers['dueTime'])

df_dist = pd.read_csv(distanceMatrix_file)
dist = np.asarray(df_dist)
d_max = dist.max()

qu25 = [3,1]

Ya = 5
Yd = 5
a1= 1
a2 = 3
M = 3500#当数据集为c201的时候改大点
# sets
num_vechile = 2
num_patients = 10#共26个节点 第0个是车站
K = [k for k in range(num_vechile) ] #vehicles
V = [i for i in range(num_patients+1) ]#all vetrice
N = [i for i in range(1,num_patients+1) ]#patients
arcs = [(x1, x2) for x1 in V for x2 in V if x1!=x2] #arcs
D = {(i,j):dist[i,j] for i,j in arcs}
Ua = [u for u in range(5)]
Vd =[v for v in range(4)]

start = time.time()
from gurobipy import Model, GRB, quicksum, max_

mdl = Model('VRPTW')
# decision variables
x = mdl.addVars(arcs, K, vtype=GRB.BINARY, name='x')
at = mdl.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='arrivaltime')<
dt = mdl.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
y = mdl.addVars(N, K, vtype=GRB.BINARY)
pa = mdl.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
pd = mdl.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
ua = mdl.addVars(V, Ua, vtype=GRB.BINARY)
vd = mdl.addVars(V, Vd, vtype=GRB.BINARY)
z = mdl.addVars(V, vtype=GRB.BINARY)
wa = mdl.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
wd = mdl.addVars(N, K, vtype=GRB.CONTINUOUS, ub=GRB.INFINITY)

# objective


#constraints

mdl.addConstrs(quicksum(x[i, j, k] for k in K for j in V if j != i) == 1 for i in N)
mdl.addConstrs(
    quicksum(x[i, j, k] for i in V if i != j) == quicksum(x[j, i, k] for i in V if i != j) for j in V for k in K)
mdl.addConstrs(quicksum(x[0, j, k] for j in N) == 1 for k in K)
# level matching
mdl.addConstrs(requ[i] * y[i, k] <= qu25[k] for k in K for i in N)
mdl.addConstrs(y[i, k] == quicksum(x[i, j, k] for j in V if j != i) for i in N for k in K)
# time window
# at = dt+travqlt
mdl.addConstrs(dt[i, k] + dist[i, j] - at[j, k] <= (1 - x[i, j, k]) * M for i in N for j in N for k in K if i != j)
mdl.addConstrs(at[j, k] <= dt[i, k] + dist[i, j] + (1 - x[i, j, k]) * M for i in N for j in N for k in K if i != j)
mdl.addConstrs(at[j, k] <= D[0, j] + M * (1 - x[0, j, k]) for j in N for k in K)
mdl.addConstrs(at[j, k] >= D[0, j] - M * (1 - x[0, j, k]) for j in N for k in K)
mdl.addConstrs((x[i, 0, k] == 1) >> (at[0, k] == dt[i, k] + dist[i, 0]) for i in N for k in K)

# dt = max(ei,ati)+stime
mdl.addConstrs(dt[0, k] == 0 for k in K)
mdl.addConstrs(dt[i, k] >= at[i, k] + s_time[i] for i in N for k in K)
mdl.addConstrs(dt[i, k] >= e_time[i] + s_time[i] for i in N for k in K)
mdl.addConstrs(dt[i, k] <= at[i, k] + s_time[i] + (1 - z[i]) * M for i in N for k in K)
mdl.addConstrs(dt[i, k] <= e_time[i] + s_time[i] + z[i] * M for i in N for k in K)

# 惩罚 departure
mdl.addConstrs(quicksum(vd[i, h] for h in Vd) == 1 for i in N)
mdl.addConstrs(vd[0, h] == 0 for h in Vd for k in K)
mdl.addConstrs(dt[i, k] <= l_time[i] * vd[i, 0] + (l_time[i] + 15) * vd[i, 1] +
               (l_time[i] + 30) * vd[i, 2] + M * vd[i, 3] for i in N for k in K)
mdl.addConstrs(dt[i, k] >=l_time[i] * vd[i, 1] + (l_time[i] + 15) * vd[i, 2] +
               (l_time[i] + 30) * vd[i, 3] + 0.001 for i in N for k in K)

# wd = y[ik]*pd[ik]
# mdl.addConstrs((y[i,k]==1)>>(pd[i,k]==0*vd[i,0]+a1*vd[i,1]+a2*vd[i,2]+Y*vd[i,3]) for i in N for k in K)
# mdl.addConstrs((y[i,k]==0)>>(pd[i,k]==0) for i in N for k in K)
mdl.addConstrs(pd[i, k] == 0 * vd[i, 0] + a1 * vd[i, 1] + a2 * vd[i, 2] + Yd * vd[i, 3] for i in N for k in K)
mdl.addConstrs(wd[i, k] <= pd[i, k] + M * (1 - y[i, k]) for i in N for k in K)
mdl.addConstrs(wd[i, k] >= pd[i, k] - M * (1 - y[i, k]) for i in N for k in K)
mdl.addConstrs(wd[i, k] <= M * y[i, k] for i in N for k in K)
mdl.addConstrs(wd[i, k] >= 0 for i in N for k in K)

# arrival time
mdl.addConstrs(quicksum(ua[i, g] for g in Ua) == 1 for i in N)
mdl.addConstrs(at[i, k] <= (e_time[i] - 30) * ua[i, 0] + (e_time[i] - 15) * ua[i, 1] + e_time[i] * ua[i, 2]
               + l_time[i] * ua[i, 3] + M * ua[i, 4] for i in N for k in K)
mdl.addConstrs(at[i, k] >= (e_time[i] - 30) * ua[i, 1] + (e_time[i] - 15) * ua[i, 2] + e_time[i] * ua[i, 3]
               + l_time[i] * ua[i, 4] + 0.001for i in N for k in K)

mdl.addConstrs(
    pa[i, k] == Ya * ua[i, 0] + a2 * ua[i, 1] + a1 * ua[i, 2] + 0 * ua[i, 3] + Ya * ua[i, 4] for i in N for k in K)
mdl.addConstrs(wa[i, k] <= pa[i, k] + M * (1 - y[i, k]) for i in N for k in K)
mdl.addConstrs(wa[i, k] >= pa[i, k] - M * (1 - y[i, k]) for i in N for k in K)
mdl.addConstrs(wa[i, k] <= M * y[i, k] for i in N for k in K)
mdl.addConstrs(wa[i, k] >= 0 for i in N for k in K)
# number of patients in each route
mdl.addConstrs(quicksum(y[i, k] for i in N) >= 2 for k in K)


cost_list =[]
penalty_list = []
obj_list = []
sol_list = []
w1 = 0
w2 = 1

sol_list = []

#开始优化
for e in range(0,101):
    w1 = 0 + e * 0.01
    w2 = 1 - e * 0.01
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(w1 * (quicksum(x[i, j, k] * dist[i, j] for i, j in arcs for k in K if i != j)) + w2 * (
            quicksum(wd[i, k] for i in N for k in K) + quicksum(wa[i, k] for i in N for k in K)))

    #mdl.Params.MIPGap = 0.1
    #mdl.Params.TimeLimit = 5000
    mdl.optimize()
    #A = [(x1, x2, x3) for x1, x2 in arcs for x3 in K]
    #active_arcs = [a for a in A if x[a].x > 0.99]
    #print(active_arcs)
    #print('weightobjvalue= %f' % mdl.ObjVal)

    cost = 0
    for i, j in arcs:
        for k in K:
            if x[i, j, k].x > 0.99:
                cost = x[i, j, k].x * dist[i, j] + cost

    penalty = 0
    for i in N:
        for k in K:
            if y[i, k].x > 0.99:
                print('wd[%d,%d] = %f' % (i, k, pd[i, k].x))
                print('at[%d,%d] = %f' % (i, k, at[i, k].x))
                print('dt[%d,%d] = %f' % (i, k, dt[i, k].x))
                print('wa[%d,%d] = %f' % (i, k, pa[i, k].x))
                penalty = wd[i, k].x + wa[i, k].x + penalty
    cost_list.append(cost)
    penalty_list.append(penalty)
    obj_list.append([cost,penalty])
    print([cost,penalty])


    route = []
    for k in K:
        unsort_subroute = []
        sort_subroute = []
        for i, j in arcs:
            if x[i, j, k].x > 0.99:
                # insert
                unsort_subroute.append((i, j))

        if len(sort_subroute) == 0:
            sort_subroute.append(unsort_subroute[0][0])
            sort_subroute.append(unsort_subroute[0][1])
        unsort_subroute.pop(0)

        while len(unsort_subroute) != 0:
            for unsort_node in unsort_subroute:
                if sort_subroute[0] != 0:
                    if unsort_node[1] == sort_subroute[0]:
                        sort_subroute.insert(0, unsort_node[0])
                        unsort_subroute.remove(unsort_node)
                        break
                if sort_subroute[-1] != 0:
                    if unsort_node[0] == sort_subroute[-1]:
                        sort_subroute.append(unsort_node[1])
                        unsort_subroute.remove(unsort_node)
                        break
        route.append(sort_subroute)
        print(route)
    #print(route)
    sol_list.append(route)

    mdl.reset(0)

end = time.time()
print(end-start)
print(cost_list)

print(penalty_list)
print(obj_list)
obj = np.array(obj_list)
obj_file = os.path.join(save_obj, '%sparetoobj_0.npy' % (DataSet))
np.save(obj_file,obj)

sol = np.array(sol_list)
print(sol)
sol_file = os.path.join(save_sol, '%sparetosol_0.npy' % (DataSet))
np.save(sol_file, sol)

plt.scatter(cost_list,penalty_list)
plt.show()


