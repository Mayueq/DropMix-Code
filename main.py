import train
from train import *
from train import train


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        
    # 'cora', 'citeseer', 'pubmed'
    #dataset='citeseer'
    dataset = 'citeseer'
    for i in range(10):
        print('seed:',i)
        setup_seed(i)
        train(dataset)
        
print('\n final:',sum(acc_f)/len(acc_f))

 
