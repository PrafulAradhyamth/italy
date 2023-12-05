# this script is created by Lu Xiao
# this loss can be used to calculate the loss between the estimated global label and the real global label
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn as nn
import torch

class Global_Binary_loss(nn.Module):
    def __init__(self, weight=1, tanh_use=False):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        """
        super(Global_Binary_loss, self).__init__()
        # if ignore_label is not None:
        #     assert not square_dice, 'not implemented'
        #     ce_kwargs['reduction'] = 'none'
        self.weight = weight
        self.tanh_use = tanh_use
        # self.aggregate = aggregate
        # self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        # based on 3D Volum
        # get batch size
        b, c, H, W = net_output.shape

        #######################
        # motion area based
        #######################

        #######################
        # mean motion prob based
        #######################
        unique_ele = torch.unique(net_output[0, 1, :, :])
        print('unique elements in the output:', unique_ele)
        pred_mean_motion_prob = torch.tensor([torch.mean(net_output[i, 1, :, :]) for i in range(b)],
                                             dtype=torch.float, requires_grad=True)
        gt_mean_motion_prob = torch.tensor([torch.mean(target[i, 0, :, :]) for i in range(b)], dtype=torch.float)

        # calculate loss on batch
        global_loss = binary_cross_entropy_with_logits(pred_mean_motion_prob, torch.sigmoid(gt_mean_motion_prob))

        return global_loss

########test######
# gl = Global_Binary_loss()
# output = torch.rand((3,2,3,4,5), requires_grad=True)
#
# target = torch.rand((3,1,3,4,5), requires_grad=False)
#
# loss = gl(output, target)
# print("final", loss)
# # print(loss.backward())
# assert loss.backward() == None, "global loss cannot be backpropgated"
###########################