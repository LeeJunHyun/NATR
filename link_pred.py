import argparse
import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric
from sklearn.metrics import roc_auc_score

from ogb.linkproppred import Evaluator
from ogb_data import load_dataset
from models import LinkPredictor,NATR, NATR_plug
from utils import check_discrete_attribute
from logger import LoggerSingleRun
import pdb

def train(model, predictor, node_feat, adj_t, split_edge, optimizer, batch_size, attributes, device, args):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        if args.use_node_feat and args.use_attribute:
            h = model(node_feat.weight, attributes, adj_t)
        elif args.use_attribute:
            h = model(attributes, adj_t)
        else:
            h = model(node_feat.weight, adj_t)

        edge = pos_train_edge[perm].t()
        edge_neg = negative_sampling(edge_index, num_nodes=adj_t.size(0),
                                num_neg_samples=perm.size(0), method='dense')

        
        if 'NATR' in args.model and args.aux_loss:
            loss = 0

            for i in range(h.shape[0]):
                pos_out = predictor(h[i][edge[0]], h[i][edge[1]])
                loss += -torch.log(pos_out + 1e-15).mean()
                neg_out = predictor(h[i][edge_neg[0]], h[i][edge_neg[1]])
                loss += -torch.log(1 - neg_out + 1e-15).mean()
                
        else:
            pos_out = predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            neg_out = predictor(h[edge_neg[0]], h[edge_neg[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            loss = pos_loss + neg_loss
        
        loss.backward()

        if args.use_node_feat: torch.nn.utils.clip_grad_norm_(node_feat.weight, 1.0)
        if 'NATR' in args.model: torch.nn.utils.clip_grad_norm_(model.attribute_embed.weight, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()


        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, node_feat, adj_t, split_edge, evaluator, batch_size, attributes, device, args):
    model.eval()
    predictor.eval()



    if args.use_attribute and args.use_node_feat:
        h = model(node_feat.weight, attributes, adj_t)
    elif args.use_attribute:
        h = model(attributes, adj_t)
    else:
        h = model(node_feat.weight, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=False):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size, shuffle=False):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size, shuffle=False):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size, shuffle=False):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size, shuffle=False):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    results['AUROC'] = (roc_auc_score(split_edge['train']['edge_label'].cpu().numpy(), torch.cat([pos_train_pred,neg_valid_pred],dim=0).cpu().numpy()),\
                        roc_auc_score(split_edge['valid']['edge_label'].cpu().numpy(), torch.cat([pos_valid_pred,neg_valid_pred],dim=0).cpu().numpy()),\
                        roc_auc_score(split_edge['test']['edge_label'].cpu().numpy(), torch.cat([pos_test_pred,neg_test_pred],dim=0).cpu().numpy()))

    for K in [5, 10, 20]:
        evaluator.K = K
        train_hits = evaluator.eval({'y_pred_pos': pos_train_pred,'y_pred_neg': neg_valid_pred})[f'hits@{K}']
        valid_hits = evaluator.eval({'y_pred_pos': pos_valid_pred,'y_pred_neg': neg_valid_pred})[f'hits@{K}']
        test_hits = evaluator.eval({'y_pred_pos': pos_test_pred,'y_pred_neg': neg_test_pred})[f'hits@{K}']
        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    

    return results


def main():

    parser = argparse.ArgumentParser(description='NATR-OGB-format')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=12345)

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8, help='for GAT')
    parser.add_argument('--use_attribute', action='store_true')
    parser.add_argument('--use_node_feat', action='store_true')

    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels_predictor', type=int, default=256)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--eval_batch_size', type=int, default=512 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--early_stop_count', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)

    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--fpSize', type=int, default=1024, help='num of attributes for Fingerprint (DDI)')
    parser.add_argument('--radius', type=int, default=3, help='radius for Fingerprint (DDI)')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    parser.add_argument('--model', type=str, default='GCN', choices=['NATR_plug', 'NATR'])
    parser.add_argument('--link_method', type=str, default='mul', choices=['cat','add','sub','mul', 'non-NN'])

    parser.add_argument('--NATR_GNN', type=str, default='GCN', choices=['MLP','GCN','SAGE', 'GAT', 'GATV2', 'SGC','ARMA','GraphUNet','PNA', 'GCNII','Graphormer'])
    parser.add_argument('--num_layers_encoder', type=int, default=2)
    parser.add_argument('--num_layers_decoder', type=int, default=2)
    parser.add_argument('--FFN_dim', type=int, default=512)
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.07)

    parser.add_argument('--num_synthetic_attributes', type=int, default=64)
    parser.add_argument('--num_synthetic_nodes', type=int, default=64)
    parser.add_argument('--avg_synthetic_attributes', type=int, default=5)
    args = parser.parse_args()
    print(args)

    if 'NATR' not in args.model: args.NATR_GNN = ''
    if args.model=='NATR':
        num_layers = max(args.num_layers, args.num_layers_decoder)
        args.num_layers = num_layers
        args.num_layers_decoder = num_layers

    project_name = args.data
    group_name = f'{args.model}'
    instance_name = f'{args.num_layers}_{args.hidden_channels}_{args.lr}_{args.dropout}'
    if args.use_wandb:
        import wandb
        if not args.exp_name =='': project_name += f'_{args.exp_name}'
        if 'NATR' in args.model: group_name += f'_{args.NATR_GNN}'
        if args.use_attribute:
            group_name += '_A'
        elif args.use_node_feat:
            group_name += '_N'
        if not args.run_name =='': group_name += f'_{args.run_name}'
        wandb.init(project=project_name, group=group_name, name=instance_name, config=args)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    data = load_dataset(data_name=args.data, device=device,val_ratio=args.val_ratio, test_ratio=args.test_ratio, \
                        fpSize=args.fpSize, radius=args.radius, num_synthetic_nodes=args.num_synthetic_nodes,\
                        num_synthetic_attributes=args.num_synthetic_attributes, avg_synthetic_attributes=args.avg_synthetic_attributes)
    adj_t = data['adj_t'].to_symmetric()
    split_edge = data['split_edge']


    # Remove non-use attribute & Add base (dummy) attribute
    add_dummy = False
    if data['x'].sum(dim=0).min()==0:
        data['x'] = data['x'][:,data['x'].sum(dim=0).nonzero().squeeze()]
    if data['x'].sum(dim=1).min()==0:
        data['x'] = torch.cat([data['x'], torch.ones(data['x'].shape[0],1).to(data['x'].device)],dim=1)
        add_dummy = True

    if not args.use_attribute and not args.use_node_feat:
        print('\n\n Invalid node feature \n\n')
        assert False
    elif args.use_attribute and 'NATR' in args.model:
        node_feat = torch.nn.Embedding(adj_t.size(0), args.hidden_channels).to(device)
        torch.nn.init.xavier_uniform_(node_feat.weight)

        attributes = data['x']
        in_channels =  attributes.shape[1]
        att_flag, att_stat = check_discrete_attribute(attributes)
        if not att_flag: 
            print('\n\nThe node feature is a dense matrix!\n\n')
            assert False
    elif args.use_attribute:
        attributes = data['x']
        att_flag, att_stat = check_discrete_attribute(attributes)
        if not att_flag: 
            
            print('\n\nThe node feature is a dense matrix!\n\n')
            assert False
        in_channels = attributes.shape[1]
        node_feat = False
    elif args.use_node_feat or 'NATR' in args.model:
        node_feat = torch.nn.Embedding(adj_t.size(0), args.hidden_channels).to(device)
        torch.nn.init.xavier_uniform_(node_feat.weight)
        in_channels = args.hidden_channels
        attributes = False
    else:
        print('\n\n Invalid node feature \n\n')
        assert False

    if 'JK' in args.model and 'NATR' in args.model:
        predictor = LinkPredictor(in_channels=args.hidden_channels*args.num_layers_decoder, hidden_channels=args.hidden_channels_predictor,\
                                    num_layers=args.num_layers_predictor, dropout=args.dropout, method=args.link_method).to(device)
    elif 'JK' in args.model:
        predictor = LinkPredictor(in_channels=args.hidden_channels*args.num_layers, hidden_channels=args.hidden_channels_predictor,\
                                    num_layers=args.num_layers_predictor, dropout=args.dropout, method=args.link_method).to(device)
    else:
        predictor = LinkPredictor(in_channels=args.hidden_channels, hidden_channels=args.hidden_channels_predictor,\
                                    num_layers=args.num_layers_predictor, dropout=args.dropout, method=args.link_method).to(device)
    if args.model=='NATR_plug':
        deg= None
        if args.NATR_GNN=='PNA':
            d = torch_geometric.utils.degree(data['split_edge']['train']['edge'][:,1],num_nodes=data['x'].shape[0], dtype=torch.long)
            deg = torch.zeros(d.max()+1, dtype=torch.long).to(device)
            deg += torch.bincount(d, minlength=deg.numel())
        model = NATR_plug(in_channels=in_channels, hidden_channels=args.hidden_channels, num_classes=args.hidden_channels, num_layers= args.num_layers,\
                        dropout=args.dropout, num_attributes=data['x'].shape[1], num_layers_encoder=args.num_layers_encoder,\
                        num_layers_decoder=args.num_layers_decoder, attribute_mask=data['x'],\
                        FFN_dim=args.FFN_dim, heads=args.heads, node_model=args.NATR_GNN, aux_loss=args.aux_loss, add_dummy=add_dummy,deg=deg\
                        ).to(device)
        if not args.from_scratch:
            model.node_model.load_state_dict(torch.load(osp.join('pretrained',\
                                            f'{args.data}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_GNN.pt')), strict=False)
            predictor.load_state_dict(torch.load(osp.join('pretrained',\
                                            f'{args.data}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_predictor.pt')))
            if args.use_node_feat:
                node_feat.load_state_dict(torch.load(osp.join('pretrained',\
                                                f'{args.data}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_node_feat.pt')))

    elif args.model=='NATR':
        deg= None
        if args.NATR_GNN=='PNA':
            d = torch_geometric.utils.degree(data['split_edge']['train']['edge'][:,1],num_nodes=data['x'].shape[0], dtype=torch.long)
            deg = torch.zeros(d.max()+1, dtype=torch.long).to(device)
            deg += torch.bincount(d, minlength=deg.numel())
        model = NATR(in_channels=in_channels, hidden_channels=args.hidden_channels, num_classes=args.hidden_channels, num_layers= args.num_layers,\
                        dropout=args.dropout, num_attributes=data['x'].shape[1], num_layers_encoder=args.num_layers_encoder,\
                        num_layers_decoder=args.num_layers_decoder, attribute_mask=data['x'],\
                        FFN_dim=args.FFN_dim, heads=args.heads, node_model=args.NATR_GNN, aux_loss=args.aux_loss, add_dummy=add_dummy,\
                        use_attribute=args.use_attribute,deg=deg).to(device)

        if args.NATR_GNN=='Graphormer':
            N = data['x'].shape[0]
            adj_ = data['adj_t'].to_dense().cpu()
            adj_ = adj_ + torch.eye(N)
            import os
            shortest_path_file_name = f'./shortest_{args.data}_{12345}_{args.val_ratio}.npy'
            if os.path.exists(shortest_path_file_name):
                shortest_path_result = np.load(shortest_path_file_name)
            else:
                import pyximport
                pyximport.install(setup_args={"include_dirs": np.get_include()})
                from . import algos
                adj_ = adj_.bool()
                shortest_path_result, path = algos.floyd_warshall(adj_.numpy())
                np.save(shortest_path_file_name, shortest_path_result)
            max_dist = np.amax(shortest_path_result)
            shortest_path_result = torch.from_numpy(shortest_path_result).to(device)
            model.shortest_path_result = shortest_path_result
            model.spatial_pos_encoder = torch.nn.Embedding(max_dist+1, args.heads, padding_idx=0).to(device)

            degree_ = torch.sum(adj_, dim=1).long().to(device)
            model.degree = degree_
            max_degree_ = torch.max(degree_).cpu().int().item()
            model.centrality_encoder = torch.nn.Embedding(max_degree_+1, args.hidden_channels, padding_idx=0).to(device)
    else:  
        print('\n\nInvalid model name\n\n')
        assert False


    print(predictor)
    print(model)

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'AUROC': LoggerSingleRun(),
        'Hits@5': LoggerSingleRun(),
        'Hits@10': LoggerSingleRun(),
        'Hits@20': LoggerSingleRun(),
    }


    if args.use_node_feat or 'NATR' in args.model:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(node_feat.parameters())\
                                        + list(predictor.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.use_attribute:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.use_wandb: wandb.watch(model)

    if args.test_only:
        model.load_state_dict(torch.load(osp.join('pretrained',\
                                        f'{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_GNN.pt')), strict=False)
        predictor.load_state_dict(torch.load(osp.join('pretrained',\
                                        f'{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_predictor.pt')))
        if not args.use_attribute:
            node_feat.load_state_dict(torch.load(osp.join('pretrained',\
                                            f'{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_node_feat.pt')))
        results = test(model=model, predictor=predictor, node_feat=node_feat, adj_t=adj_t, split_edge=split_edge,\
                            evaluator=evaluator, batch_size=args.batch_size, attributes=attributes, device=device, args=args)
        for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                print(f'[{key}] Train: {100 * train_hits:.2f}%, '
                    f'Valid: {100 * valid_hits:.2f}%, '
                    f'Test: {100 * test_hits:.2f}%')
        return None
        
    epoch_count = 0
    best_valid_hit = 0
    best_epoch = 0
    best_test_hit = 0
    for epoch in range(args.epochs):
        if epoch_count>args.early_stop_count: break
        epoch_count+=1
                    
        loss = train(model=model, predictor=predictor, node_feat=node_feat, adj_t=adj_t,split_edge=split_edge,\
                    optimizer=optimizer, batch_size=args.batch_size, attributes=attributes, device=device, args=args)
        if args.use_wandb: wandb.log({'loss':loss, 'epoch':epoch})

        if (epoch>0) and (epoch % args.eval_steps == 0):
            results = test(model=model, predictor=predictor, node_feat=node_feat, adj_t=adj_t, split_edge=split_edge,\
                            evaluator=evaluator, batch_size=args.eval_batch_size, attributes=attributes, device=device, args=args)
            print(f'[Epoch: {epoch:03d}/{args.epochs}] ', f'Loss: {loss:.4f} ({project_name},{group_name})')
            for key, result in results.items():
                if key =="top_nodes_Hits":
                    top_nodes_Hits = result
                    if args.use_wandb: wandb.log(result)
                else:

                    loggers[key].add_result(result)
                    train_hits, valid_hits, test_hits = result
                    print(f'[{key}] Train: {100 * train_hits:.2f}%, '
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%')
                    
                    if args.use_wandb: wandb.log({key+'_train':train_hits,key+'_valid':valid_hits,key+'_test':test_hits})

            valid_hit = results['Hits@20'][1]
            test_hit = results['Hits@20'][2]

            if valid_hit > best_valid_hit:
                epoch_count=0
                best_epoch = epoch
                best_valid_hit = valid_hit
                best_test_hit = test_hit
                best_top_nodes_Hits = top_nodes_Hits
                if not args.model=='NATR':
                    if epoch<args.epochs//2:
                        torch.save(model.state_dict(), osp.join('pretrained',\
                                    f'{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_GNN.pt'))
                        torch.save(predictor.state_dict(), osp.join('pretrained',\
                                    f'{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_predictor.pt'))
                        if args.use_node_feat:
                            torch.save(node_feat.state_dict(), osp.join('pretrained',\
                                        f'{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_node_feat.pt'))
                else:
                    torch.save(model.state_dict(), osp.join('pretrained',\
                                f'{args.data}_{args.model}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_GNN.pt'))
                    torch.save(predictor.state_dict(), osp.join('pretrained',\
                                f'{args.data}_{args.model}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_predictor.pt'))
                    if args.use_node_feat:
                        torch.save(node_feat.state_dict(), osp.join('pretrained',\
                                    f'{args.data}_{args.model}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_node_feat.pt'))

            print(f'Best Epoch: {best_epoch}, Valid Hits: {100*best_valid_hit:.2f}%, Test Hits: {100*best_test_hit:.2f}%')
            print('---')
    for key in loggers.keys():
        print(key)
        highest_valid, final_test = loggers[key].print_statistics()
        if args.use_wandb: wandb.log({key+'_final_valid':highest_valid,key+'_final_test':final_test})
    for key in best_top_nodes_Hits.keys():
        if args.use_wandb: wandb.log({f'final_{key}':best_top_nodes_Hits[key]})

if __name__ == "__main__":
    main()
