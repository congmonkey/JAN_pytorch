import torch
import numbers

def GaussianKernelDistance(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
  batch_size = int(source.size()[0])
  k1 = torch.sum(torch.pow(source, 2), 1).repeat(1,batch_size)
  k2 = torch.sum(torch.pow(target, 2), 1).resize(1, batch_size).repeat(batch_size, 1)
  k3 = torch.mm(source, torch.t(target))
  L2_distance = k1+k2-2*k3
  if not fix_sigma:
    bandwidth = torch.sum(L2_distance.data) / (batch_size**2-batch_size) * (kernel_mul**(float(-kernel_num)/2))
  else:
    bandwidth = fix_sigma * (kernel_mul**(-float(kernel_num)/2))
  bandwidth_list = [bandwidth * (kernel_mul**kernel_num) for i in xrange(kernel_num)]
  kernel_val = [torch.exp(L2_distance/-bandwidth_temp) for bandwidth_temp in bandwidth_list]
  return sum(kernel_val)


def MMDLoss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
  batch_size = int(source.size()[0])
  mmd_item1 = GaussianKernelDistance(source, source, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
  mmd_item2 = GaussianKernelDistance(target, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
  mmd_item3 = GaussianKernelDistance(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
  return torch.sum(mmd_item1 + mmd_item2 - 2 * mmd_item3) / float(batch_size**2)


def JMMDLoss(source_list, target_list, kernel_mul_list=None, kernel_num_list=None, fix_sigma_list=None):
  batch_size = int(source_list[0].size()[0])
  layer_num = len(source_list)
  if not kernel_mul_list:
    kernel_mul_list = [2.0] * layer_num
    kernel_num_list = [5] * layer_num
    fix_sigma_list = [None] * layer_num
  else:
    if isinstance(kernel_mul_list, numbers.Number):
      kernel_mul_list = [kernel_mul_list] * layer_num
    if isinstance(kernel_num_list, numbers.Number):
      kernel_num_list = [kernel_num_list] * layer_num
    if isinstance(fix_sigma_list, numbers.Number):
      fix_sigma_list = [fix_sigma_list] * layer_num
  jmmd_list = []
  for i in xrange(layer_num):
    source = source_list[i]
    target = target_list[i]
    kernel_mul = kernel_mul_list[i]
    kernel_num = int(kernel_num_list[i])
    fix_sigma = fix_sigma_list[i]
    jmmd_list.append(
        [GaussianKernelDistance(source, source, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma), 
         GaussianKernelDistance(target, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma), 
         GaussianKernelDistance(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)]
    )
  jmmd_loss = jmmd_list[0]
  for i in xrange(1, layer_num):
    jmmd_loss[0] = torch.mul(jmmd_loss[0], jmmd_list[i][0])
    jmmd_loss[1] = torch.mul(jmmd_loss[1], jmmd_list[i][1])
    jmmd_loss[2] = torch.mul(jmmd_loss[2], jmmd_list[i][2])
  jmmd_loss = jmmd_loss[0] + jmmd_loss[1] - 2 * jmmd_loss[2]
  return torch.sum(jmmd_loss) / float(batch_size**2)
