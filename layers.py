import torch
from torch import nn

def conv_dw_separable(input_filters, output_filters, depth_multiplier=1):
    return nn.Sequential(
        nn.Conv2d(input_filters, input_filters * depth_multiplier, kernel_size=3, padding=1, groups=input_filters, bias=False),
        nn.BatchNorm2d(input_filters * depth_multiplier),
        nn.ReLU(inplace=True),

        nn.Conv2d(input_filters * depth_multiplier, output_filters, kernel_size=1, bias=False),
        nn.BatchNorm2d(output_filters),
        nn.ReLU(inplace=True),
    )

def conv_2d(input_filters, output_filters, kernel_size, padding=0, stride=1):
    return nn.Sequential(
      nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, padding=padding, stride=stride),
      nn.BatchNorm2d(output_filters),
      nn.ReLU(inplace=True),
  )

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.Linear:
            m.bias.data.fill_(0.01)

class SSE(nn.Module):
  def __init__(self, in_channels, filter_count, depth_multiplier=1):
    super().__init__()
    
    squeeze_filters = filter_count
    expand_filters = 4 * filter_count

    self.squeeze = conv_2d(in_channels, squeeze_filters, kernel_size=1)
    self.expand1 = conv_2d(squeeze_filters, expand_filters, kernel_size=1)
    self.expand3_dw = conv_dw_separable(squeeze_filters, expand_filters, depth_multiplier=depth_multiplier)
    
    [module.apply(init_weights) for module in [self.squeeze, self.expand1, self.expand3_dw]]

  def forward(self, x):
    squeeze_out = self.squeeze(x)
    ex1_out = self.expand1(squeeze_out)          
    ex3_dw_out = self.expand3_dw(squeeze_out)
    
    # concatenate along filter dimension
    return torch.cat([ex3_dw_out, ex1_out], 1)
  
class Slim(nn.Module):
  def __init__(self, in_channels, filter_count, depth_multiplier=1):      
    super().__init__()
    
    expand_filters = 4 * filter_count
    dw_conv_filters = 3 * filter_count
    
    # 1x1 convolution to restructure the skip connection
    self.skip_projection = nn.Sequential(
      nn.Conv2d(in_channels, 2 * expand_filters, kernel_size=1, bias=False),
      nn.BatchNorm2d(2 * expand_filters),
      nn.ReLU(inplace=True),
    )
    
    self.sse1 = SSE(in_channels, filter_count, depth_multiplier=depth_multiplier)
    self.sse2 = SSE(2 * expand_filters,filter_count, depth_multiplier=depth_multiplier)
    self.expand3_dw = conv_dw_separable(2 * expand_filters, dw_conv_filters, depth_multiplier=depth_multiplier)

    [module.apply(init_weights) for module in [self.skip_projection, self.expand3_dw]]


  def forward(self, x):
    identity = x
    sse1_out = self.sse1(x)
    sse2_out = self.sse2(sse1_out + self.skip_projection(identity))
    return self.expand3_dw(sse2_out)

