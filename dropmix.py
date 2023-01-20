from calendar import different_locale
import difflib
from re import L
import re
from tkinter.messagebox import OK
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch
import torch.nn as nn
from model import GCN,Readout,Discriminator
from utils import compute_ppr


def similarity(x1, x2):
    
        x2 = x2.t()
        x = x1.mm(x2)
        x1_frobenius = x1.norm(p=2,dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(p=2,dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)
        final_value = x.mul(1 / x_frobenins)
        sort_queue, sort_index = torch.sort(final_value, dim=0, descending=False)
    
        return final_value




class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.gcn3=GCN(n_h,n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

   

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2,epoch):
      
        
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)
        
        h_2= self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)
     
        h_3= self.gcn1(seq2, adj, sparse)
        h_4= self.gcn2(seq2, diff, sparse)
        
        
        num_all=seq1.size()[1]  
        root_l=0.15
        root_h=0.95
        num_l=int(num_all*root_l)
        num_h=int(num_all*root_h)
        num=(num_h-num_l)*1
        
        
        sim1=similarity(h_3[0],h_1[0])
        sim1=torch.max(sim1,1)[0]
        
        sim2=similarity(h_4[0],h_2[0])
        sim2=torch.max(sim2,1)[0]
        
#        sim=sim1+sim2
#        index=torch.sort(sim)[1]
#        better_index=index[num_l:num_h]
        
        index1=torch.sort(sim1)[1]
        better_index1=index1[num_l:num_h]
        index2=torch.sort(sim2)[1]
        better_index2=index2[num_l:num_h]

        if epoch==50:
            s1=torch.sort(sim1)[0]
            inx=np.arange(num_all) 
            s1=s1.cpu().detach().numpy()
            an = np.polyfit(inx/(num_all/100), s1,13)
            plt.figure()
            yvals = np.polyval(an, inx/(num_all/100))
            plt.plot(inx/(num_all/100), yvals, label='legend')
            plt.scatter(inx/(num_all/100), s1, c='red', s=3, label='legend')
            a=list(range(0,100,5))
            a=np.array(a)
            plt.xticks(a)
            plt.savefig('node/iii.png')


    
        
        h_3_m=h_3[:,better_index1,:]
        h_4_m=h_4[:,better_index2,:]     
        h_3_s=h_3[:,better_index1,:]
        h_4_s=h_4[:,better_index2,:]
       
    
        idx = np.random.permutation(better_index1.size()[0]) 
        
        h_3_ss=h_3_s[:,idx,:]
        h_4_ss=h_4_s[:,idx,:]
    
#        number=int(h_1.size()[2]*0.1)
#        i=np.arange(h_1.size()[2])
#        i=i[:number]
        i=np.arange(0,h_1.size()[2],6)
            
        lam=0.35
        h_3_m[:,:,i]=h_3_ss[:,:,i]*lam+h_3_s[:,:,i]*(1-lam)
        h_4_m[:,:,i]=h_4_ss[:,:,i]*lam+h_4_s[:,:,i]*(1-lam)


   
        ret = self.disc(c_1, c_2, h_1, h_2, h_3_m, h_4_m,samp_bias1, samp_bias2)    
        

        del c_1, c_2, h_1, h_2,h_3,h_4
        torch.cuda.empty_cache()

        return ret,num
    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
 
        return (h_1 + h_2).detach(), c.detach()


    
