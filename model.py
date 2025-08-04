
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, SGConv, ARMAConv, JumpingKnowledge, PNAConv, GCN2Conv
import torch
import torch.nn.functional as F
import copy
import torch_geometric


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, method='cat'):
        super().__init__()
        self.method = method
        if not method=='non-NN':
            if method=='cat':
                in_channels=in_channels*2
            self.lins = torch.nn.ModuleList()
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, 1))

            self.dropout = dropout

    def forward(self, x_i, x_j):
        
        if self.method=='non-NN':
            return torch.sigmoid((F.normalize(x_i, dim=1)*F.normalize(x_j, dim=1)).sum(dim=-1))
        if self.method=='cat':
            x = torch.cat([x_i, x_j], dim=1)
        elif self.method=='mul':
            x = x_i *x_j
        elif self.method=='sub':
            x = x_i - x_j
        elif self.method=='add':
            x = x_i + x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class NodePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, num_classes))

        self.dropout = dropout

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class NATR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, num_attributes, num_layers_encoder,num_layers_decoder,\
                attribute_mask, FFN_dim=2048, heads=8, node_model='GCN', aux_loss=False, add_dummy=False, use_attribute=False, deg=None):
        super(NATR, self).__init__()
        self.heads= heads

        self.num_layers_decoder = num_layers_decoder
        self.num_layers_encoder = num_layers_encoder
        self.use_attribute = use_attribute
        self.node_embed = torch.nn.Linear(in_channels, hidden_channels)
        self.node_model = torch.nn.ModuleList()
        self.node_norm = torch.nn.ModuleList()
        self.attribute_norm = torch.nn.LayerNorm(hidden_channels)
        


        ################# Node Model ####################################
        self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        self.node_model_type = node_model
        if node_model=='MLP':
            for _ in range(num_layers_decoder):
                self.node_model.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='GCN':
            for _ in range(num_layers_decoder):
                self.node_model.append(GCNConv(hidden_channels, hidden_channels, cached=True))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='SAGE':
            for _ in range(num_layers_decoder):
                self.node_model.append(SAGEConv(hidden_channels, hidden_channels))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='GAT':
            for _ in range(num_layers_decoder - 1):
                self.node_model.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
            self.node_model.append(GATConv(hidden_channels, num_classes, heads=1, concat=False))
            self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='GATV2':
            for _ in range(num_layers_decoder - 1):
                self.node_model.append(GATv2Conv(hidden_channels, hidden_channels//heads, heads=heads, concat=True))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
            self.node_model.append(GATv2Conv(hidden_channels, num_classes, heads=1, concat=False))
            self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='SGC':
            self.node_model = SGConv(hidden_channels, hidden_channels, K=num_layers_decoder)
            self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='ARMA':
            self.node_model = ARMA(hidden_channels, hidden_channels, num_classes, num_layers_decoder, dropout, num_stacks=2,shared_weight=False)
            self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            for _ in range(num_layers_decoder):
                self.node_model.append(PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                            aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=None, towers=2, pre_layers=1, post_layers=1,
                            divide_input=False))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='GCNII':
            alpha=0.1
            theta=0.5
            shared_weights=True
            for layer in range(num_layers_decoder):
                self.node_model.append(GCN2Conv(hidden_channels, alpha, theta, layer+1, shared_weights, normalize=True))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
        elif node_model=='Graphormer':
            for _ in range(num_layers_decoder):
                self.node_model.append(GraphormerLayer(hidden_channels, heads))
                self.node_norm.append(torch.nn.LayerNorm(hidden_channels))
            
        ##############################################################


        ##############################################################
        if add_dummy:
            dummy_mask = torch.zeros(num_attributes,num_attributes)
            dummy_mask[-1,:] +=1
            dummy_mask[:,-1] +=1
            dummy_mask[-1,-1] = 0
            self.dummy_mask = dummy_mask.bool().to(attribute_mask.device)
        else:
            self.dummy_mask = None

        self.attribute = attribute_mask

        self.attribute_mask = attribute_mask.to_sparse()
        self.attribute_attention_mask = (1-attribute_mask).bool()

        self.attribute_embed = torch.nn.Embedding(num_attributes, hidden_channels)
        torch.nn.init.xavier_uniform_(self.attribute_embed.weight)

        ################# ENCODER ####################################
        if num_layers_encoder==0:
            self.encoder=MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout)

        else:
            encoder_layer = AttributeEncoderLayer(hidden_channels=hidden_channels, heads=heads, FFN_dim=FFN_dim, dropout=dropout)
            encoder_norm = torch.nn.LayerNorm(hidden_channels)
            self.encoder = AttributeEncoder(encoder_layer, num_layers_encoder, encoder_norm)

        ##############################################################

        ################# DECODER ####################################
        decoder_layer = AttributeDecoderLayer(hidden_channels=hidden_channels, heads=heads, FFN_dim=FFN_dim, dropout=dropout)

        self.num_layers_decoder = num_layers_decoder
        self.decoder = torch.nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers_decoder)])
        ##############################################################

        self.aux_loss = aux_loss
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.encoded_attribute_embed = None
        
    def forward(self, x, adj_t):
        if self.use_attribute:
            out = self.node_norm[0](self.node_embed(x))
        else:
            out = self.node_norm[0](x)
            
        node_pos = out

        if self.num_layers_encoder==0:
            attribute_embed = self.encoder(self.attribute_embed.weight)
        else:
            attribute_embed = self.encoder(src=self.attribute_embed.weight, attn_mask=self.dummy_mask)

        
        intermediate = []

        if type(self.node_model) in [SGConv, ARMA]:
            out = self.node_model(out, adj_t)
            out = F.relu(out)
            out = self.node_norm[1](out)
        elif self.node_model_type in ['Graphormer']:
            N = out.shape[0]
            out = out+self.centrality_encoder(self.degree)
            attn_bias = self.spatial_pos_encoder(self.shortest_path_result.view(-1)).permute(1,0).view(self.heads,N,N)

        for i in range(self.num_layers_decoder):

            # for SGConv, we only need to run MPNN once
            if type(self.node_model) not in [SGConv, ARMA]:
                if self.node_model_type=='MLP':
                    out = self.node_model[i](out)
                elif self.node_model_type=='GCNII':
                    out = self.node_model[i](out, node_pos, adj_t)
                elif self.node_model_type in ['Graphormer']:
                    out = self.node_model[i](out, attn_bias)
                else:
                    out = self.node_model[i](out, adj_t)
                out = F.relu(out)
                out = self.node_norm[i+1](out)

            out = self.decoder[i](query=out,attribute_embed=attribute_embed, node_pos=node_pos,\
                            attribute_pos=self.attribute_norm(self.attribute_embed.weight),attn_mask=self.attribute_attention_mask)

            if self.aux_loss:
                intermediate.append(self.norm(out))

        if self.aux_loss and self.training:
            self.encoded_attribute_embed = attribute_embed
            return torch.stack(intermediate)
        else:
            return self.norm(out)

class NATR_plug(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, num_attributes, num_layers_encoder,num_layers_decoder,\
                attribute_mask, FFN_dim=2048, heads=8, node_model='GCN', aux_loss=False, add_dummy=False,deg=None):
        super(NATR_plug, self).__init__()

        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        
        if node_model=='MLP':
            self.node_model = MLP(in_channels=in_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout)
        elif node_model=='GCN':
            self.node_model = GCN(in_channels=in_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout)
        elif node_model=='SAGE':
            self.node_model = SAGE(in_channels=in_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout)
        elif node_model=='GAT':
            self.node_model = GAT(in_channels=in_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout, heads=heads)
        elif node_model=='GATV2':
            self.node_model = GATV2(in_channels=in_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout, heads=heads)
        elif node_model=='SGC':
            self.node_model = SGC(in_channels=in_channels, num_classes=hidden_channels, K=num_layers)
        elif node_model=='ARMA':
            self.node_model = ARMA(in_channels=in_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout)
        
        self.node_norm = torch.nn.LayerNorm(hidden_channels)

        if add_dummy:
            dummy_mask = torch.zeros(num_attributes,num_attributes)
            dummy_mask[-1,:] +=1
            dummy_mask[:,-1] +=1
            dummy_mask[-1,-1] = 0
            self.dummy_mask = dummy_mask.bool().to(attribute_mask.device)
        else:
            self.dummy_mask = None
            
            
        self.attribute = attribute_mask
        self.attribute_mask = attribute_mask.to_sparse()
        self.attribute_attention_mask = (1-attribute_mask).bool()

        self.attribute_embed = torch.nn.Embedding(num_attributes, hidden_channels)
        torch.nn.init.xavier_uniform_(self.attribute_embed.weight)

        if num_layers_encoder==0:
            self.encoder=MLP(in_channels=hidden_channels, hidden_channels=hidden_channels,num_classes=hidden_channels, num_layers= num_layers, dropout=dropout)

        else:
            encoder_layer = AttributeEncoderLayer(hidden_channels=hidden_channels, heads=heads, FFN_dim=FFN_dim, dropout=dropout)
            encoder_norm = torch.nn.LayerNorm(hidden_channels)
            self.encoder = AttributeEncoder(encoder_layer, num_layers_encoder, encoder_norm)
            
        decoder_norm = torch.nn.LayerNorm(hidden_channels)
        decoder_layer = AttributeDecoderLayer(hidden_channels=hidden_channels, heads=heads, FFN_dim=FFN_dim, dropout=dropout)
        

        self.decoder = AttributeDecoder(decoder_layer, num_layers_decoder, decoder_norm, aux_loss)
        self.aux_loss = aux_loss

        
    def forward(self, x, adj_t):
        node_feat = self.node_model(x, adj_t)
        node_feat = self.node_norm(node_feat)

        if self.num_layers_encoder==0:
            attribute_embed = self.encoder(self.attribute_embed.weight)
        else:
            attribute_embed = self.encoder(src=self.attribute_embed.weight, attn_mask=self.dummy_mask)
        
        out = self.decoder(node_feat=node_feat, attribute_embed=attribute_embed,\
                            attribute_pos = self.attribute_embed.weight, attn_mask=self.attribute_attention_mask,\
                            )
        if self.aux_loss:
            if self.training:
                # for contrastive loss
                self.encoded_attribute_embed = attribute_embed
                node_feat = node_feat.unsqueeze(0)
                return torch.cat([node_feat,out],dim=0)
            else:
                return out[-1]

        return out


class AttributeEncoder(torch.nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src, attn_mask):
        output = src

        for layer in self.layers:
            output = layer(output, pos=src, attn_mask=attn_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class AttributeEncoderLayer(torch.nn.Module):

    def __init__(self, hidden_channels, heads, FFN_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(hidden_channels, heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(hidden_channels, FFN_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(FFN_dim, hidden_channels)

        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = torch.nn.ReLU()

    def forward(self, src, pos, attn_mask):
        q = k = src+pos
        src2, attn_weight = self.self_attn(q, k, value=src, attn_mask=attn_mask)
        self.attn_weight = attn_weight
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class AttributeDecoder(torch.nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, aux_loss=False,):
        super().__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.aux_loss = aux_loss
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, node_feat, attribute_embed, attribute_pos, attn_mask=None):
        output = node_feat
        intermediate = []

        for layer in self.layers:
            output = layer(query=output, attribute_embed=attribute_embed, node_pos=node_feat,\
                            attribute_pos=attribute_pos,attn_mask=attn_mask)
            if self.aux_loss:
                intermediate.append(self.norm(output))
         

        if self.norm is not None:
            output = self.norm(output)
            if self.aux_loss:
                intermediate.pop()
                intermediate.append(output)
                return torch.stack(intermediate)

        return output

class AttributeDecoderLayer(torch.nn.Module):

    def __init__(self, hidden_channels, heads, FFN_dim=2048, dropout=0.1):
        super().__init__()
        self.MHA = torch.nn.MultiheadAttention(hidden_channels, heads, dropout=dropout)
        
        self.linear1 = torch.nn.Linear(hidden_channels, FFN_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(FFN_dim, hidden_channels)

        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = torch.nn.ReLU()

    def forward(self, query, attribute_embed, node_pos, attribute_pos, attn_mask=None):
        q = query+node_pos
        k = attribute_embed + attribute_pos
        v = attribute_embed + attribute_pos
        src2 = self.MHA(q, k, value=v, attn_mask=attn_mask)[0]
        src = query + self.dropout1(src2)

        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, heads=8):
        super(Graphormer, self).__init__()
        self.heads = heads
        self.node_embed = torch.nn.Linear(in_channels, hidden_channels)

        encoder_layer = GraphormerLayer(hidden_channels, heads,FFN_dim=512, dropout=dropout)
        self.layers = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])

        self.dropout = dropout
        self.norm = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x, adj_t=None):
        N = x.shape[0]
        src = self.node_embed(x) 
        src = src+self.centrality_encoder(self.degree)

        attn_bias = self.spatial_pos_encoder(self.shortest_path_result.view(-1)).permute(1,0).view(self.heads,N,N)

        output = src

        for layer in self.layers:
            output = layer(output, attn_bias=attn_bias)

        if self.norm is not None:
            output = self.norm(output)

        return output


class GraphormerLayer(torch.nn.Module):
    def __init__(self, hidden_channels, heads, FFN_dim=512, dropout=0.5):
        super().__init__()
        self.self_attn = GraphormerMHA(embed_dim=hidden_channels,num_heads=heads,kdim=hidden_channels,vdim=hidden_channels,
                                    dropout=dropout,bias=True, self_attention=True)
        
        self.linear1 = torch.nn.Linear(hidden_channels, FFN_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(FFN_dim, hidden_channels)

        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = torch.nn.ReLU()

    def forward(self, src, attn_bias):
        
        src= self.norm1(src).unsqueeze(1)
        src2, _ = self.self_attn(src, src, src, attn_bias=attn_bias)

        src = src.squeeze() + self.dropout1(src2.squeeze())
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, num_classes))

        self.dropout = dropout

    def forward(self, x, adj_t=None):

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, num_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        # pdb.set_trace()
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GCN_JK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, mode='cat'):
        super(GCN_JK, self).__init__()

        self.node_embed = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        # self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, num_classes))
        self.jump = JumpingKnowledge(mode)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.node_embed.reset_parameters()

    def forward(self, x, adj_t):
        # pdb.set_trace()
        xs = []
        x = self.node_embed(x)
        xs += [x]
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]
        x = self.convs[-1](x, adj_t)
        xs += [x]
        x = self.jump(xs)

        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, num_classes))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, heads=8):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True))
        self.convs.append(GATConv(hidden_channels, num_classes, heads=1, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GAT_JK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, heads=8, mode='cat'):
        super(GAT_JK, self).__init__()

        self.node_embed = torch.nn.Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True))
        self.convs.append(GATConv(hidden_channels, num_classes, heads=1, concat=False))
        self.jump = JumpingKnowledge(mode)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.node_embed.reset_parameters()

    def forward(self, x, adj_t):
        xs = []
        x = self.node_embed(x)
        xs += [x]
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]
        x = self.convs[-1](x, adj_t)
        xs += [x]
        x = self.jump(xs)

        return x

class GATV2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, heads=8):
        super(GATV2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels//heads, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels//heads, heads=heads, concat=True))
        self.convs.append(GATv2Conv(hidden_channels, num_classes, heads=1, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SGC(torch.nn.Module):
    def __init__(self, in_channels, num_classes, K):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, num_classes, K=K)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)
        return x


class ARMA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, num_stacks=2,shared_weight=False):
        super(ARMA, self).__init__()

        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks, num_layers, shared_weight, dropout=dropout)
        self.conv2 = ARMAConv(hidden_channels, num_classes, num_stacks, 1, shared_weight, dropout=dropout)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_t)
        return x


class PNA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, deg):
        super(PNA, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.node_embed = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.convs.append(PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=None, towers=2, pre_layers=1, post_layers=1,
                           divide_input=False))
            self.batch_norms.append(torch_geometric.nn.BatchNorm(hidden_channels))
        self.convs.append(PNAConv(in_channels=hidden_channels, out_channels=num_classes,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=None, towers=2, pre_layers=1, post_layers=1,
                           divide_input=False))

        self.dropout = dropout

    def reset_parameters(self):
        for norm, conv in zip(self.batch_norms,self.convs):
            norm.reset_parameters()
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x = self.node_embed(x)

        for norm, conv in zip(self.batch_norms,self.convs[:-1]):
            x = conv(x, adj_t)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        
        return x



class GCNII(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers, dropout, alpha=0.1, theta=0.5, shared_weights=True):
        super(GCNII, self).__init__()
        self.node_embed = torch.nn.Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCN2Conv(hidden_channels, alpha, theta, 1, shared_weights, normalize=True))
        for layer in range(num_layers - 2):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer+2, shared_weights, normalize=True))
        self.convs.append(GCN2Conv(hidden_channels, alpha, theta, num_layers, shared_weights, normalize=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x = x_0 = self.node_embed(x).relu()
        for conv in self.convs[:-1]:
            x = conv(x, x_0, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, x_0, adj_t)
        return x


from typing import Optional

import torch



import math
from typing import Optional, Tuple

import torch
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn



class GraphormerMHA(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value