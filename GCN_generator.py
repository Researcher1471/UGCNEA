import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pop_class import *
from function import *
from wasserstein import *

class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, adj):
        x=x.to(torch.float32)
        adj=adj.to(torch.float32)
        support = torch.mm(x, self.weight)  # 线性变换
        output = torch.mm(adj, support)  # 邻居信息聚合
        return output

class GCN_UP(nn.Module):
    def __init__(self, D, N):
        super(GCN_UP, self).__init__()
        self.conv1 = GCN(D, D)
        self.conv2 = GCN(D, D)
    
    def adj_Upsample(self, x):
        distances = torch.cdist(x, x)
        distances.fill_diagonal_(float('inf'))
        nearest_indices = torch.argmin(distances, dim=1)
        adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float32)
        for i, nearest_index in enumerate(nearest_indices):
            adj[i, nearest_index] = 1
        adj = adj.cuda()
        return adj
    def forward(self, x, adj):
        
        x = self.conv1(x, adj)
        
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, scale_factor=2, mode='linear')
        x = x.permute(0, 2, 1)
        x = x.squeeze(0)
        adj = self.adj_Upsample(x)
        
        x = self.conv2(x, adj)
        return x

class GCN_UP_Model(object):
    def __init__(self, n_var, popsize, fname, xl, xu):
        self.epoches=500
        self.lr=0.01
        self.n_var=n_var
        self.fname=fname
        self.xl=torch.from_numpy(xl).cuda()
        self.xu=torch.from_numpy(xu).cuda()
        self.model=GCN_UP(n_var, popsize).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.98)
    def ConDistloss(self, ref, Pop):
        distances = torch.cdist(ref, Pop, p=2)  # p=2 表示欧氏距离
        # 计算每个点的最近邻距离
        _, indices = torch.min(distances, dim=1)  # 获取最小距离的索引
        # 获取最小距离
        min_distances = distances[torch.arange(indices.shape[0]), indices]
        # 计算平均最近邻距离
        mt_dist = torch.mean(min_distances)
        return mt_dist
    def CrowdDegree(self, pop):
        distances = torch.cdist(pop, pop)
        distances_diag = distances.clone()
        distances_diag.fill_diagonal_(float('inf'))
        min_dis, indices = torch.min(distances_diag, dim=1)
        return torch.max(min_dis)
        
    def findominatImdex(self, pop,n_var):  
        N_particle = pop.shape[0]  
        NicheSize = 1
        DR=np.zeros((N_particle,N_particle))
        for i in range(N_particle):
            #根据距离选择
            point=pop[i,0:n_var]
            array=pop[:,0:n_var]
            distances = np.linalg.norm(array - point, axis=1)
            NicheIndex = np.argsort(distances)[:NicheSize]
            for j in NicheIndex:
                DR[i,j]=1
        return DR
    def train(self, train_data):
        self.model.train()
        print("运行train")
        DR=self.findominatImdex(train_data,self.n_var)
        DR=DR.astype(int)
        train_data=train_data
        for epoch in range(1,self.epoches+1):
            self.optimizer.zero_grad()
            X=torch.from_numpy(train_data).float().cuda()        
            edg=torch.from_numpy(DR).cuda()
            predict_result=self.model(X, edg)
            loss1=sinkhorn_wasserstein(predict_result, X)
            loss2=self.ConDistloss(X, predict_result)
            loss=loss1+loss2
            total_loss = loss.item()
            loss.backward() 
            self.optimizer.step()
            if epoch % 10 == 0:
                print("Epoch [{}/100], Loss: {:.4f}".format(epoch+1,total_loss))
        

    def predict_offspring(self, train_data):
        self.model.eval()
        print("运行test")
        num_nodes=train_data.shape[0]
        DR=self.findominatImdex(train_data,self.n_var)
        DR=DR.astype(int)
        train_data=train_data
        X=torch.from_numpy(train_data).float().cuda()
        edg=torch.from_numpy(DR).cuda()
        result=self.model(X, edg)
        result=torch.clamp(result, min=self.xl, max=self.xu)
        return result
 

# pi_value = np.pi
# xl=np.array([1,-1])     
# xu=np.array([3,1])         
# net=GCN_UP_Model(2, 200, 'MMF1', xl, xu)
# PS=sio.loadmat('./TruePS_PF/CEC2020/MMF1_Reference_PSPF_data.mat')['PS']
# PF=sio.loadmat('./TruePS_PF/CEC2020/MMF1_Reference_PSPF_data.mat')['PF']
# in_PS=PS[::4,:]
# in_PF=PF[::4,:]

# # parent_ps=sio.loadmat('./parent.mat')['ps']

# n_obj=2
# n_var=2
# fname='MMF1'
# Population=pop_class(in_PS,in_PF)
# net.train(in_PS)
# result=net.predict_offspring(in_PS)
# ps=result.cpu().detach().numpy()
# Offspring=ps
# Offspring_r=np.zeros((Offspring.shape[0],n_obj))
# for k in range (Offspring.shape[0]):
#     temp_x=Offspring[k,0:n_var]
#     temp_new_point=np.copy(temp_x)
#     t_reult=eval(fname)(temp_new_point)
#     t_reult=t_reult.reshape(1,n_obj)   
#     Offspring_r[k,0:n_obj]=t_reult
# pf=Offspring_r
# IGDx=IGD_calculation(ps,PS)
# IGDf=IGD_calculation(pf,PF)
# print("补全性能")
# print('IGDx',IGDx)
# print('IGDf',IGDf)
# sio.savemat('result.mat', {'ps':ps,'pf':pf})

# xl=np.array([1,-1])     
# xu=np.array([3,1])  
# net=VGAE_Model(2, 2, 'MMF1', xl, xu)
# PS=sio.loadmat('./TruePS_PF/CEC2020/MMF1_Reference_PSPF_data.mat')['PS']
# PF=sio.loadmat('./TruePS_PF/CEC2020/MMF1_Reference_PSPF_data.mat')['PF']
# n_var=2
# n_obj=2
# fname='MMF1'

# Population=pop_class(PS,PF)
# net.train(Population)
# result=net.predict_offspring(Population)
# ps=result[:,0:2]
# ps=ps.cpu().detach().numpy()
# Offspring=ps
# Offspring_r=np.zeros((Offspring.shape[0],n_obj))
# for k in range (Offspring.shape[0]):
#     temp_x=Offspring[k,0:n_var]
#     temp_new_point=np.copy(temp_x)
#     t_reult=eval(fname)(temp_new_point)
#     t_reult=t_reult.reshape(1,n_obj)   
#     Offspring_r[k,0:n_obj]=t_reult
# pf=Offspring_r
# IGDx=IGD_calculation(ps,PS)
# print("IGDx",IGDx)
# IGDf=IGD_calculation(pf,PF)
# print("IGDf",IGDf)
# sio.savemat('result.mat', {'ps':ps,'pf':pf})