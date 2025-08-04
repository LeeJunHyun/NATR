import argparse
import random
import numpy as np
import os.path as osp


import torch
import torch.nn.functional as F


from evaluator_node import Evaluator

from ogb_data import load_dataset
from models import  NATR, NodePredictor, NATR_plug
from utils import check_discrete_attribute, ContrastiveLoss
from logger import LoggerSingleRun


def train(model, predictor, data, optimizer, node_feat, args):
    model.train()
    optimizer.zero_grad()
    if args.use_attribute:
        out = model(data.x, data.adj_t)
    elif args.use_node_feat or 'NATR' in args.model:
        out = model(node_feat.weight, data.adj_t)
    
    if 'NATR' in args.model and args.aux_loss:
        loss = 0
        for i in range(out.shape[0]):
            loss += F.nll_loss(predictor(out[i][data.train_mask]).log_softmax(dim=-1), data.y[data.train_mask])
    else:    
        out = predictor(out[data.train_mask]).log_softmax(dim=-1)
        loss = F.nll_loss(out.log_softmax(dim=-1), data.y[data.train_mask])
    loss.backward()

    if args.use_node_feat: torch.nn.utils.clip_grad_norm_(node_feat.weight, 1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model,  predictor,data, evaluator, node_feat, args):
    model.eval()

    if args.use_attribute:
        h = model(data.x, data.adj_t)
    elif args.use_node_feat or 'NATR' in args.model:
        h = model(node_feat.weight, data.adj_t)
    out = predictor(h).log_softmax(dim=-1)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask].view(-1,1),
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_mask].view(-1,1),
        'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask].view(-1,1),
        'y_pred': y_pred[data.test_mask],
    })['acc']

    
    return train_acc, valid_acc, test_acc


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
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--eval_steps', type=int, default=1)

    parser.add_argument('--data', type=str, default='photo')
    parser.add_argument('--fpSize', type=int, default=1024, help='num of attributes for Fingerprint (DDI)')
    parser.add_argument('--radius', type=int, default=3, help='radius for Fingerprint (DDI)')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    parser.add_argument('--model', type=str, default='GCN', choices=['NATR', 'NATR_plug'])

    parser.add_argument('--NATR_GNN', type=str, default='GCN', choices=['MLP','GCN','SAGE', 'GAT', 'GATV2', 'SGC','ARMA','GraphUNet'])
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

    if args.use_wandb:
        import wandb
        project_name = args.data
        group_name = f'{args.model}'
        instance_name = f'{args.num_layers}_{args.hidden_channels}_{args.lr}_{args.dropout}'
        if not args.exp_name =='': project_name += f'_{args.exp_name}'
        if 'NATR' in args.model: group_name += f'_{args.NATR_GNN}'
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
                        num_synthetic_attributes=args.num_synthetic_attributes, avg_synthetic_attributes=args.avg_synthetic_attributes, task='node')
    data['adj_t'] = data['adj_t'].to_symmetric()
    
    # Remove non-use attribute & Add base (dummy) attribute
    add_dummy = False
    if data['x'].sum(dim=0).min()==0:
        data['x'] = data['x'][:,data['x'].sum(dim=0).nonzero().squeeze()]
    if data['x'].sum(dim=1).min()==0:
        data['x'] = torch.cat([data['x'], torch.ones(data['x'].shape[0],1).to(data['x'].device)],dim=1)
        add_dummy = True

    if args.use_attribute and args.use_node_feat:
        node_feat = torch.nn.Embedding(data['adj_t'].size(0), args.hidden_channels).to(device)
        torch.nn.init.xavier_uniform_(node_feat.weight)

        in_channels = args.hidden_channels
        attributes = data['x']
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

    elif args.use_node_feat or args.model=='NATR':
        node_feat = torch.nn.Embedding(data['adj_t'].size(0), args.hidden_channels).to(device)
        torch.nn.init.xavier_uniform_(node_feat.weight)
        in_channels = args.hidden_channels
        attributes = False
    else:
        print('\n\n Invalid node feature \n\n')
        assert False

    
    predictor = NodePredictor(in_channels=args.hidden_channels, hidden_channels=args.hidden_channels_predictor, num_classes=data.num_classes, num_layers=args.num_layers_predictor, dropout=args.dropout).to(device)
    
    if args.model=='NATR_plug':
        model = NATR_plug(in_channels=in_channels, hidden_channels=args.hidden_channels, num_classes=args.hidden_channels, num_layers= args.num_layers,\
                        dropout=args.dropout, num_attributes=data['x'].shape[1], num_layers_encoder=args.num_layers_encoder,\
                        num_layers_decoder=args.num_layers_decoder, attribute_mask=data['x'],\
                        FFN_dim=args.FFN_dim, heads=args.heads, node_model=args.NATR_GNN, aux_loss=args.aux_loss, add_dummy=add_dummy,\
                        ).to(device)
        if args.aux_loss: model.contrastive_loss = ContrastiveLoss(num_attributes=data['x'].shape[1], temperature=args.temperature, device=device)
        if not args.from_scratch:
            model.node_model.load_state_dict(torch.load(osp.join('pretrained',\
                                            f'node_{args.data}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_GNN.pt')), strict=False)
            if not args.use_attribute:
                node_feat.load_state_dict(torch.load(osp.join('pretrained',\
                                                f'node_{args.data}_{args.NATR_GNN}_{args.hidden_channels}_{args.num_layers}_node_feat.pt')))
    elif args.model=='NATR':
        model = NATR(in_channels=in_channels, hidden_channels=args.hidden_channels, num_classes=args.hidden_channels, num_layers= args.num_layers,\
                        dropout=args.dropout, num_attributes=data['x'].shape[1], num_layers_encoder=args.num_layers_encoder,\
                        num_layers_decoder=args.num_layers_decoder, attribute_mask=data['x'],\
                        FFN_dim=args.FFN_dim, heads=args.heads, node_model=args.NATR_GNN, aux_loss=args.aux_loss, add_dummy=add_dummy,\
                        use_attribute=args.use_attribute).to(device)
        if args.aux_loss: model.contrastive_loss = ContrastiveLoss(num_attributes=data['x'].shape[1], temperature=args.temperature, device=device)
    else:  
        print('\n\nInvalid model name\n\n')
        assert False


    print(model)
    print(predictor)

    if args.use_wandb: wandb.watch(model)

    evaluator = Evaluator(name='ogbn-mag')
    logger = LoggerSingleRun()
    if args.use_attribute:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.use_node_feat or 'NATR' in args.model:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(node_feat.parameters()) + list(predictor.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best_valid=0
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, predictor, data, optimizer, node_feat, args)
        if args.use_wandb: wandb.log({'loss':loss, 'epoch':epoch})
        if epoch % args.eval_steps == 0:
            result = test(model, predictor, data, evaluator,node_feat, args)
            train_acc, valid_acc, test_acc = result
            logger.add_result((train_acc, valid_acc, test_acc))

            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
            if args.use_wandb: wandb.log({'train_acc':train_acc,'valid_acc':valid_acc,'test_acc':test_acc}) 

            if valid_acc > best_valid:
                torch.save(model.state_dict(), osp.join('pretrained',\
                            f'node_{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_GNN.pt'))
                torch.save(predictor.state_dict(), osp.join('pretrained',\
                            f'node_{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_predictor.pt'))
                if not args.use_attribute:
                    torch.save(node_feat.state_dict(), osp.join('pretrained',\
                                f'node_{args.data}_{args.model}_{args.hidden_channels}_{args.num_layers}_node_feat.pt'))


    highest_valid, final_test = logger.print_statistics()
    if args.use_wandb: wandb.log({'final_valid':highest_valid,'final_test':final_test})
if __name__ == "__main__":
    main()
