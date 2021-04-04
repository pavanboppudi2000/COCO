import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from torchsummary import summary
from torch_sparse import spmm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        # 两层分别：
        # 300，1024
        # 1024，2408
        # print('in_features:\n',in_features)
        # print('out_features:\n',out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('input的维度:',input.shape)                # torch.Size([20, 300])     torch.Size([20, 1024])
        # print('self.weight的维度',self.weight.shape)     # torch.Size([300, 1024])   torch.Size([1024, 2048])
        # print('adj的维度:',adj.shape)                    # torch.Size([20, 20])      torch.Size([20, 20])
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        # print('support的维度:',support.shape)            # torch.Size([20, 1024])    torch.Size([20, 2048])
        # print('output的维度:',output.shape)              # torch.Size([20, 1024])    torch.Size([20, 2048])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        # resnet101--model
        # num_classes--20
        # t--0.4
        # adj_file--'data/voc/voc_adj.pkl'
        # in_channel--300
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        # print('model.conv1:\n',model.conv1)
        # print('model.layer1:\n',model.layer1)
        # print('model.layer2:\n',model.layer2)
        # print('model.layer3:\n',model.layer3)
        # print('model.layer4:\n',model.layer4)
        # print('self.features:\n',self.features)
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.2)

        _adj = gen_A(num_classes, t, adj_file)
        # print('_adj的数据类型', type(_adj))      # _adj的数据类型 <class 'numpy.ndarray'>
        # print('_adj的维度', _adj.shape)         # _adj的维度 (20, 20)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        adj = gen_adj(self.A).detach()
        x = self.dropout(inp)
        x = self.gc1(x, adj)
        x = self.pairNorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = self.pairNorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc3(x, adj)
        

        x = x.transpose(0, 1)
        # print('feature的维度:',feature.shape)    # feature的维度: torch.Size([1, 2048])
        # print('x的维度:',x.shape)                # x的维度: torch.Size([2048, 20])
        x = torch.matmul(feature, x)
        # print('x的维度:',x.shape)                # x的维度: torch.Size([1, 20])
        # print(x.shape)
        # print(x.shape)
        return x
    
    def pairNorm(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
        x =  x / rownorm_individual
        return x


    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    # voc
    # num_classes=20
    # t=0.4
    # adj_file='data/voc/voc_adj.pkl'
    model = models.resnet101(pretrained=pretrained).to(device)
    # resnet101结构
    # print('resnet101 model:\n')
    # summary(model,(3,224,224))
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)


