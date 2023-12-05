#combine pixel-wised label loss and global loss together
#local loss still uses deep supervision DC_CE_loss, global loss uses the defined binary_loss

from torch import nn
from losses.binary_loss import Global_Binary_loss

class LocalGlobalLoss(nn.Module):
	def __init__(self, local_loss, weight_local=1, weight_global=1):
		'''
		two weight factors can balance the weight of two losses, can be set by user
		'''
		super(LocalGlobalLoss, self).__init__()
		self.weight_global = weight_global
		self.weight_local = weight_local
		self.local_loss = local_loss
		self.global_loss = Global_Binary_loss()

	def forward(self, net_output, target):
		local_loss = self.local_loss(net_output, target)
		global_loss = self.global_loss(net_output, target)
		tot_loss = self.weight_local*local_loss + self.weight_global*global_loss

		return tot_loss