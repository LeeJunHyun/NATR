import torch

def check_discrete_attribute(x):
    # check if the node feature matrix is discrete(sparse) or continuous(dense)
    # print attribute statistic
    x = x > 1e-16
    num_attribute = x.shape[1]
    x_sum = x.sum(dim=1)
    
    att_stat = {}
    att_stat['max'] = x_sum.max().cpu().item()
    att_stat['avg']  = x_sum.float().mean().cpu().item()
    att_stat['median']  = x_sum.median().cpu().item()
    att_stat['min']  = x_sum.min().cpu().item()
    att_stat['ZeroFeat'] = (x_sum==0).sum().cpu().item()
    print(f"[Attribute Statistic] Max:{att_stat['max']}, Avg:{att_stat['avg']:.1f}, Median: {att_stat['median']}, Min: {att_stat['min']}, ZeroFeat: {att_stat['ZeroFeat']}")
    
    if att_stat['min']==num_attribute:
        return False, att_stat
    else:
        return True, att_stat

def check_edge_overlap(adj_t,split_edge):
    adj_t_row, adj_t_col, _ = adj_t.coo()

    valid_edge = split_edge['valid']['edge']

    test_edge = split_edge['test']['edge']

    valid_count = 0
    for edge in valid_edge:
        row, col = edge
        flag = torch.max(((adj_t_row==row)*(adj_t_col==col)).sum(),((adj_t_row==col)*(adj_t_col==row)).sum())
        if flag>0:
            valid_count+=1


    test_count = 0
    for edge in test_edge:
        row, col = edge
        flag = torch.max(((adj_t_row==row)*(adj_t_col==col)).sum(),((adj_t_row==col)*(adj_t_col==row)).sum())
        if flag>0:
            test_count+=1

    print(f'{valid_count/valid_edge.shape[0] *100}% of Valid edges is in adj_t\n{test_count/test_edge.shape[0] *100}% of Test edges is in adj_t')

