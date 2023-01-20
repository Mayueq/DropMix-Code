# DropMix
This is the Pytorch code for reproducing the results of the Paper: DropMix: Better Graph Contrastive Learning with Harder Negative Samples
DropMix is a simple and efficient method for Graph Contrastive Learning (GCL) based unsupervised learning settings to make better hard negative samples.
We show that with this method, GCL models can acheieve better performance on benchmark graph datasets such as Cora/Citeseer/Pubmed. 
This code is based on MVGRL model.

## Requirements 
This code is tested with Python 3.6 and requires following packages:

torch==1.10.1

numpy==1.19.2

networkx==2.5.1

pytz==2022.1

requests==2.27.1

scikit-learn==0.24.2

scipy==1.5.4

torch-cluster==1.5.9

torch-geometric==2.0.2

torch-scatter==2.0.9

torch-sparse==0.6.12

torchvision==0.11.2

# How to run 

For reproducing results of DropMix of Table2 in the paper, go to directory DropMix-Code and run the following commands:

`python main.py`

