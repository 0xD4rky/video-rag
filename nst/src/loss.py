# lets define our losses
from dataload import *


## 1. Content loss

class ContentLoss(nn.Module):

    def __init__(self,target):
        super().__init__()

        self.target = target.detach()
        # we use detach as we dont want to include the copy of 'target' tensor to be included in the 
        # computational graph for gradient calculation

    def forward(self, input):

        self.loss = F.mse_loss(input, self.target)
        return self.loss
    


## 2. Style Loss

def gram_matrix(input):

    a,b,c,d = input.size()

    """
    a = batch size
    b = number of feature maps (Nl)
    c*d = dimensions of a feature map (Ml {h*w} )
    """

    features = input.view(a*b,c*d)

    G = torch.matmul(features, features.t())

    return G.div(a*b*c*d)

