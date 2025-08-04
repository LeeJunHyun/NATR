import torch_geometric
from torch_geometric import utils
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, CitationFull, WikipediaNetwork, Coauthor
import os.path as osp
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import pdb
import progressbar
import numpy as np
import math
import random
import torch_sparse

def load_dataset(data_name='cora_ml', device=torch.device('cpu'), val_ratio=0.05, test_ratio=0.1, fpSize=1024, radius=3, num_synthetic_nodes=32, num_synthetic_attributes=64, avg_synthetic_attributes=4, task='link'):
    data_name = data_name.lower()
    if data_name in ['cora_ml']:

        if data_name.endswith('_full'):
            data_name = data_name[:-5]

        saved_file_name = f'./dataset/CitationFull/{data_name}_{val_ratio}_{test_ratio}_{task}_ogb.pt'
        if osp.exists(saved_file_name):
            dataset = torch.load(saved_file_name)
            print(f'[Data Load] {saved_file_name}')
        else:
            
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset', 'CitationFull')

            if task=='link':
                transform = T.Compose([
                    T.ToSparseTensor(),
                    T.ToDevice(device),
                    T.RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True,
                                    add_negative_train_samples=False),
                ])
                dataset = CitationFull(path, name=data_name, transform=transform)[0]
            else:

                transform = T.Compose([
                    T.ToSparseTensor(),
                    T.ToDevice(device),
                ])
                dataset = CitationFull(path, name=data_name, transform=transform)[0]
                num_nodes_per_classes = []
                for i in range(dataset.y.max()+1):
                    num_nodes_per_classes.append((dataset.y==i).sum())
                num_nodes_per_classes = torch.Tensor(num_nodes_per_classes)
                ignored_classes = (num_nodes_per_classes<53).nonzero().squeeze()
                train_mask = torch.zeros(dataset.x.shape[0])
                val_mask = torch.zeros(dataset.x.shape[0])
                test_mask = torch.zeros(dataset.x.shape[0])
                for node_class in range(dataset.y.max()+1):
                    if node_class not in ignored_classes:
                        node_class_idx = (dataset.y==node_class).nonzero().squeeze()
                        perm = torch.randperm(len(node_class_idx))
                        train_idx = node_class_idx[perm[:20]]
                        val_idx = node_class_idx[perm[20:50]]
                        test_idx = node_class_idx[perm[50:]]
                        train_mask[train_idx] = 1
                        val_mask[val_idx] = 1
                        test_mask[test_idx] = 1
                dataset.train_mask = train_mask.cpu().nonzero().squeeze()
                dataset.val_mask = val_mask.cpu().nonzero().squeeze()
                dataset.test_mask = test_mask.cpu().nonzero().squeeze()
                dataset.num_classes = (dataset.y.max()+1 - len(ignored_classes)).cpu().item()
                if data_name =='cora':
                    idx_empty_class = (dataset.y == 67).nonzero().squeeze()
                    dataset.y[idx_empty_class] = 1
            
            torch.save(dataset,saved_file_name)
            print(f'[Data Save] {saved_file_name}')

    elif data_name in ['computers', 'photo']:
        saved_file_name = f'./dataset/Amazon/{data_name}_{val_ratio}_{test_ratio}_{task}_ogb.pt'
        if osp.exists(saved_file_name):
            
            dataset = torch.load(saved_file_name)
            print(f'[Data Load] {saved_file_name}')
        else:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset', 'Amazon')

            if task=='link':
                transform = T.Compose([
                    T.ToSparseTensor(),
                    T.ToDevice(device),
                    T.RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True,
                                    add_negative_train_samples=False),
                ])

                dataset = Amazon(path, name=data_name, transform=transform)[0]
            else:

                transform = T.Compose([
                    T.ToSparseTensor(),
                    T.ToDevice(device),
                ])
                
                
                dataset = Amazon(path, name=data_name, transform=transform)[0]
                num_nodes_per_classes = []
                for i in range(dataset.y.max()+1):
                    num_nodes_per_classes.append((dataset.y==i).sum())
                num_nodes_per_classes = torch.Tensor(num_nodes_per_classes)
                ignored_classes = (num_nodes_per_classes<53).nonzero().squeeze()
                train_mask = torch.zeros(dataset.x.shape[0])
                val_mask = torch.zeros(dataset.x.shape[0])
                test_mask = torch.zeros(dataset.x.shape[0])
                for node_class in range(dataset.y.max()+1):
                    if node_class not in ignored_classes:
                        node_class_idx = (dataset.y==node_class).nonzero().squeeze()
                        perm = torch.randperm(len(node_class_idx))
                        train_idx = node_class_idx[perm[:20]]
                        val_idx = node_class_idx[perm[20:50]]
                        test_idx = node_class_idx[perm[50:]]
                        train_mask[train_idx] = 1
                        val_mask[val_idx] = 1
                        test_mask[test_idx] = 1
                dataset.train_mask = train_mask.cpu().nonzero().squeeze()
                dataset.val_mask = val_mask.cpu().nonzero().squeeze()
                dataset.test_mask = test_mask.cpu().nonzero().squeeze()
                dataset.num_classes = (dataset.y.max()+1 - len(ignored_classes)).cpu().item()
            
                transform = T.Compose([
                    T.ToSparseTensor(),
                    T.ToDevice(device),
                ])

            torch.save(dataset,saved_file_name)
            print(f'[Data Save] {saved_file_name}')
    elif data_name in ['ogbl-ddi-subset']:
        saved_file_name = f'./dataset/ogbl_ddi/subset_morgan_{radius}_{fpSize}_0.1_0.1_link_ogb.pt'
        if osp.exists(saved_file_name):
            dataset = torch.load(saved_file_name)
            print(f'[Data Load] {saved_file_name}')
        
        else:
            import pandas as pd
            import gzip
            import shutil
            from rdkit import Chem
            from rdkit.Chem import AllChem

            dataset = PygLinkPropPredDataset(name = 'ogbl-ddi', transform=T.ToSparseTensor()) 
            split_edge = dataset.get_edge_split()
            train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
            graph = dataset[0]

            with gzip.open('./dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz', 'rb') as f_in:
                with open('./dataset/ogbl_ddi/mapping/nodeidx2drugid.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            nodeidx2drugid = pd.read_csv('./dataset/ogbl_ddi/mapping/nodeidx2drugid.csv')

            drug_bank_db = pd.read_csv('./dataset/drugbank/structure links.csv')

            drug_vocab = pd.read_csv('./dataset/drugbank/drugbank vocabulary.csv')

            inchikey_mol_dict = {}
            inchi_mol_dict = {}
            with Chem.SDMolSupplier('./dataset/drugbank/structures.sdf') as suppl:
                for mol in suppl:
                    try:
                        inchi = Chem.MolToInchi(mol)
                        inchi_mol_dict[inchi] = mol
                        inchikey = Chem.InchiToInchiKey(inchi)
                        inchikey_mol_dict[inchikey] = mol
                    except:
                        pass
            inchikey_all_list = list(inchikey_mol_dict.keys())
            inchi_all_list = list(inchi_mol_dict.keys())

            node_FP_dict={}

            for nodeidx in nodeidx2drugid['node idx']:
                drugid = nodeidx2drugid.loc[nodeidx, 'drug id']
                if (drug_bank_db['DrugBank ID']==drugid).sum()>0:

                    inchi = drug_bank_db.loc[drug_bank_db['DrugBank ID']==drugid,'InChI' ].item()
                    inchikey = drug_bank_db.loc[drug_bank_db['DrugBank ID']==drugid,'InChIKey' ].item()
                    if inchi in inchi_all_list:
                        mol = inchi_mol_dict[inchi]
                        FP = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fpSize).ToList()
                        if sum(FP) ==0:
                            pass
                        else:
                            node_FP_dict[nodeidx] = FP
                    elif inchikey in inchi_all_list:
                        mol = inchikey_mol_dict[inchikey]
                        FP = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fpSize).ToList()
                        if sum(FP) ==0:
                            pass
                        else:
                            node_FP_dict[nodeidx] = FP

           
            node_indices = list(node_FP_dict.keys())

            ####### original split #######
            x = torch.cat([torch.Tensor(node_FP_dict[nodeidx]).unsqueeze(0) for nodeidx in node_indices],dim=0).to(device)
            row, col, _ = graph.adj_t.coo()
            edge_index = torch.stack([col, row], dim=0).to(device)

            adj_t = graph.adj_t[node_indices,node_indices]
            dense_adj_train = utils.to_dense_adj(train_edge['edge'].T, max_num_nodes = dataset.data.num_nodes).squeeze()
            dense_adj_val = utils.to_dense_adj(valid_edge['edge'].T, max_num_nodes = dataset.data.num_nodes).squeeze()
            dense_adj_test = utils.to_dense_adj(test_edge['edge'].T, max_num_nodes = dataset.data.num_nodes).squeeze()

            dense_adj_val_neg = utils.to_dense_adj(valid_edge['edge_neg'].T, max_num_nodes = dataset.data.num_nodes).squeeze()
            dense_adj_test_neg = utils.to_dense_adj(test_edge['edge_neg'].T, max_num_nodes = dataset.data.num_nodes).squeeze()
            
            sparse_adj_train, edge_label_train = utils.dense_to_sparse(dense_adj_train[node_indices,:][:,node_indices])
            sparse_adj_val_pos, _ = utils.dense_to_sparse(dense_adj_val[node_indices,:][:,node_indices])
            sparse_adj_test_pos, _ = utils.dense_to_sparse(dense_adj_test[node_indices,:][:,node_indices])

            sparse_adj_val_neg, _ = utils.dense_to_sparse(dense_adj_val_neg[node_indices,:][:,node_indices])
            sparse_adj_test_neg, _ = utils.dense_to_sparse(dense_adj_test_neg[node_indices,:][:,node_indices])

            edge_label_val = torch.cat([torch.ones(sparse_adj_val_pos.shape[1]), torch.zeros(sparse_adj_val_neg.shape[1])],dim=0)
            edge_label_test = torch.cat([torch.ones(sparse_adj_test_pos.shape[1]), torch.zeros(sparse_adj_test_neg.shape[1])],dim=0)

            sparse_adj_val = torch.cat([sparse_adj_val_pos, sparse_adj_val_neg],dim=1)
            sparse_adj_test = torch.cat([sparse_adj_test_pos, sparse_adj_test_neg],dim=1)
            
            dataset = (torch_geometric.data.data.Data(x=x, edge_index =edge_index, edge_label=edge_label_train.to(device), edge_label_index=sparse_adj_train.to(device), adj_t=adj_t.to(device)),\
                        torch_geometric.data.data.Data(x=x, edge_index =edge_index, edge_label=edge_label_val.to(device), edge_label_index=sparse_adj_val.to(device)),\
                        torch_geometric.data.data.Data(x=x, edge_index =edge_index, edge_label=edge_label_test.to(device), edge_label_index=sparse_adj_test.to(device)))

            torch.save(dataset,saved_file_name)
            print(f'[Data Save] {saved_file_name}')

    elif data_name in ['ogbl-ddi-full']:

        saved_file_name = f'./dataset/ogbl_ddi/full_morgan_{radius}_{fpSize}_0.1_0.1_link_ogb.pt'
        if osp.exists(saved_file_name):
            
            dataset = torch.load(saved_file_name)
            print(f'[Data Load] {saved_file_name}')
        
        else:
            import pandas as pd
            import gzip
            import shutil
            from rdkit import Chem
            from rdkit.Chem import AllChem

            dataset = PygLinkPropPredDataset(name = 'ogbl-ddi', transform=T.ToSparseTensor()) 
            split_edge = dataset.get_edge_split()
            train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
            graph = dataset[0]

            with gzip.open('./dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz', 'rb') as f_in:
                with open('./dataset/ogbl_ddi/mapping/nodeidx2drugid.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            nodeidx2drugid = pd.read_csv('./dataset/ogbl_ddi/mapping/nodeidx2drugid.csv')

            drug_bank_db = pd.read_csv('./dataset/drugbank/structure links.csv')

            drug_vocab = pd.read_csv('./dataset/drugbank/drugbank vocabulary.csv')

            inchikey_mol_dict = {}
            inchi_mol_dict = {}
            with Chem.SDMolSupplier('./dataset/drugbank/structures.sdf') as suppl:
                for mol in suppl:
                    try:
                        inchi = Chem.MolToInchi(mol)
                        inchi_mol_dict[inchi] = mol
                        inchikey = Chem.InchiToInchiKey(inchi)
                        inchikey_mol_dict[inchikey] = mol
                    except:
                        pass
            inchikey_all_list = list(inchikey_mol_dict.keys())
            inchi_all_list = list(inchi_mol_dict.keys())

            node_FP_dict={}

            for nodeidx in nodeidx2drugid['node idx']:
                drugid = nodeidx2drugid.loc[nodeidx, 'drug id']
                if (drug_bank_db['DrugBank ID']==drugid).sum()>0:

                    inchi = drug_bank_db.loc[drug_bank_db['DrugBank ID']==drugid,'InChI' ].item()
                    inchikey = drug_bank_db.loc[drug_bank_db['DrugBank ID']==drugid,'InChIKey' ].item()
                    if inchi in inchi_all_list:
                        mol = inchi_mol_dict[inchi]
                        FP = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fpSize).ToList()
                        if sum(FP) ==0:
                            pass
                        else:
                            node_FP_dict[nodeidx] = FP
                    elif inchikey in inchi_all_list:
                        mol = inchikey_mol_dict[inchikey]
                        FP = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fpSize).ToList()
                        if sum(FP) ==0:
                            pass
                        else:
                            node_FP_dict[nodeidx] = FP

            node_indices = list(node_FP_dict.keys())
            

            x = torch.zeros(graph.adj_t.size(0),fpSize )
            for idx in node_indices: x[idx,:] = torch.Tensor(node_FP_dict[idx])
            x=x.to(device)
            row, col, _ = graph.adj_t.coo()
            edge_index = torch.stack([col, row], dim=0).to(device)
            
            sparse_adj_train = split_edge['train']['edge'].t()
            edge_label_train = torch.ones(sparse_adj_train.shape[1])

            sparse_adj_val = torch.cat([split_edge['valid']['edge'].t(),split_edge['valid']['edge_neg'].t()],dim=1)
            edge_label_val = torch.cat([torch.ones(split_edge['valid']['edge'].t().shape[1]), torch.zeros(split_edge['valid']['edge_neg'].t().shape[1])],dim=0)
            sparse_adj_test = torch.cat([split_edge['test']['edge'].t(),split_edge['test']['edge_neg'].t()],dim=1)
            edge_label_test = torch.cat([torch.ones(split_edge['test']['edge'].t().shape[1]), torch.zeros(split_edge['test']['edge_neg'].t().shape[1])],dim=0)
            

            dataset = (torch_geometric.data.data.Data(x=x, edge_index =edge_index, edge_label=edge_label_train.to(device), edge_label_index=sparse_adj_train.to(device), adj_t=graph.adj_t.to(device)),\
                        torch_geometric.data.data.Data(x=x, edge_index =edge_index, edge_label=edge_label_val.to(device), edge_label_index=sparse_adj_val.to(device)),\
                        torch_geometric.data.data.Data(x=x, edge_index =edge_index, edge_label=edge_label_test.to(device), edge_label_index=sparse_adj_test.to(device)))


            torch.save(dataset,saved_file_name)
            print(f'[Data Save] {saved_file_name}')

    else:
        print('Invalid data name')
        assert False

    
    if task=='link':
        if 'ogb' not in data_name:
            data = {}
            data['split_edge']={}
            data['split_edge']['train']={}
            data['split_edge']['valid']={}
            data['split_edge']['test']={}
            data['x']=dataset[0].x
            adj_t_row, adj_t_col = dataset[0].edge_label_index[0,:], dataset[0].edge_label_index[1,:]
            data['adj_t'] = torch_sparse.tensor.SparseTensor(row=adj_t_row, col=adj_t_col, sparse_sizes=[data['x'].shape[0],data['x'].shape[0]])

            data['split_edge']['train']['edge'] = dataset[0].edge_label_index.t()
            
            data['split_edge']['valid']['edge'] = dataset[1].edge_label_index[:,dataset[1].edge_label==1].t()
            data['split_edge']['valid']['edge_neg'] = dataset[1].edge_label_index[:,dataset[1].edge_label==0].t()
            
            data['split_edge']['valid']['edge_label'] = torch.cat([torch.ones(data['split_edge']['valid']['edge'].shape[0]), torch.zeros(data['split_edge']['valid']['edge_neg'].shape[0])],dim=0).to(device)

            data['split_edge']['test']['edge'] = dataset[2].edge_label_index[:,dataset[2].edge_label==1].t()
            data['split_edge']['test']['edge_neg'] = dataset[2].edge_label_index[:,dataset[2].edge_label==0].t()
            data['split_edge']['test']['edge_label'] = torch.cat([torch.ones(data['split_edge']['test']['edge'].shape[0]), torch.zeros(data['split_edge']['test']['edge_neg'].shape[0])],dim=0).to(device)


            data['split_edge']['train']['edge_label'] = torch.cat([torch.ones(data['split_edge']['train']['edge'].shape[0]), torch.zeros(data['split_edge']['valid']['edge_neg'].shape[0])],dim=0).to(device)
            data['split_edge']['eval_train'] = {'edge': data['split_edge']['train']['edge']}
        else:
            data = {}
            data['split_edge']={}
            data['split_edge']['train']={}
            data['split_edge']['valid']={}
            data['split_edge']['test']={}

            data['x']=dataset[0].x
            data['adj_t'] = dataset[0].adj_t

            data['split_edge']['train']['edge'] = dataset[0].edge_label_index.t()
            
            data['split_edge']['valid']['edge'] = dataset[1].edge_label_index[:,dataset[1].edge_label==1].t()
            data['split_edge']['valid']['edge_neg'] = dataset[1].edge_label_index[:,dataset[1].edge_label==0].t()
            
            data['split_edge']['valid']['edge_label'] = torch.cat([torch.ones(data['split_edge']['valid']['edge'].shape[0]), torch.zeros(data['split_edge']['valid']['edge_neg'].shape[0])],dim=0).to(device)

            data['split_edge']['test']['edge'] = dataset[2].edge_label_index[:,dataset[2].edge_label==1].t()
            data['split_edge']['test']['edge_neg'] = dataset[2].edge_label_index[:,dataset[2].edge_label==0].t()
            data['split_edge']['test']['edge_label'] = torch.cat([torch.ones(data['split_edge']['test']['edge'].shape[0]), torch.zeros(data['split_edge']['test']['edge_neg'].shape[0])],dim=0).to(device)


            data['split_edge']['train']['edge_label'] = torch.cat([torch.ones(data['split_edge']['train']['edge'].shape[0]), torch.zeros(data['split_edge']['valid']['edge_neg'].shape[0])],dim=0).to(device)
            data['split_edge']['eval_train'] = {'edge': data['split_edge']['train']['edge']}
    
    else:
        data = dataset
    return data
