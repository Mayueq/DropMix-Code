import numpy as np
import scipy.sparse as sp
import torch,gc
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load
import random
from model import LogReg
from dropmix import Model
from torch.utils.data import DataLoader

acc_f=[]

def train(dataset, verbose=False):

    nb_epochs = 600
    patience=20
    lr = 0.001
    l2_coef = 0.002
    hid_units = 512
    sparse = False

    adj, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)


    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]
    sample_size = features.shape[0]
    batch_size = 1
    print(sample_size)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
   
    model = Model(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('cuda')
        model.cuda()
        labels = labels.cuda()
        
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
    ba, bd, bf = [], [], []
    for i in idx:
        ba.append(adj[i: i + sample_size, i: i + sample_size])
        bd.append(diff[i: i + sample_size, i: i + sample_size])
        bf.append(features[i: i + sample_size])
    
    ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
    bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
    bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
    
    if sparse:
        ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
        bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
    else:
        ba = torch.FloatTensor(ba)
        bd = torch.FloatTensor(bd)
        bf = torch.FloatTensor(bf)
  



    for epoch in range(nb_epochs):
       
        
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]
       

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()
        
          
        model.train()
        optimiser.zero_grad()
        logits,num = model(bf, shuf_fts, ba, bd, sparse, None, None, None,epoch)
        lbl_1 = torch.ones(batch_size, sample_size * 2)  
        lbl_2 = torch.zeros(batch_size, num*2)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        lbl = lbl.cuda()
        
        loss = b_xent(logits, lbl)

        loss.backward()
        optimiser.step()
        
        

        
        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
        
        del loss,logits,shuf_fts
        torch.cuda.empty_cache()

    if verbose:
        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('model.pkl'))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))
        print('lllllll')

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()

    embeds, _ = model.embed(features, adj, diff, sparse, None)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    print(train_lbls.shape)
    test_lbls = labels[idx_test]

    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log.cuda()
        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)
    print(accs.mean().item(), accs.std().item())
    acc_f.append(accs.mean())


