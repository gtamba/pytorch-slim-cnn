import torch
from torch import nn
from torch import optim

# Helper / Shortcuts

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
        
# Network Definition

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


class SlimCNN(nn.Module):
  def __init__(self, filter_count_values=[16,32,48,64], initial_conv_filters=96, num_classes=40, depth_multiplier=1):
    super().__init__()
    
    #Store architecture hyper_params for model persistence / loading
    self.hyper_params = {"filter_count_values": filter_count_values, "initial_conv_filters": initial_conv_filters, "num_classes": num_classes, "depth_multiplier": depth_multiplier}

    self.conv1 = conv_2d(3, initial_conv_filters, 7, stride=2)
    self.max_pool1 = nn.MaxPool2d(3,2)
    
    self.slim1 = Slim(initial_conv_filters, filter_count_values[0])
    self.max_pool2 = nn.MaxPool2d(3,2)
    
    self.slim2 = Slim(filter_count_values[0] * 3, filter_count_values[1])
    self.max_pool3 = nn.MaxPool2d(3,2)

    self.slim3 = Slim(filter_count_values[1] * 3, filter_count_values[2])
    self.max_pool4 = nn.MaxPool2d(3,2)

    self.slim4 = Slim(filter_count_values[2] * 3, filter_count_values[3])
    self.max_pool5 = nn.MaxPool2d(3,2)
    
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(filter_count_values[3] * 3, num_classes)
    
    [module.apply(init_weights) for module in [self.conv1, self.fc]]
        
  def forward(self, x):
    out = self.max_pool1(self.conv1(x))
    out = self.max_pool2(self.slim1(out))
    out = self.max_pool3(self.slim2(out))
    out = self.max_pool4(self.slim3(out))
    out = self.global_pool(self.max_pool5(self.slim4(out)))
    
    return self.fc(torch.flatten(out, 1))

  def save(self, path, optimizer=None, scheduler=None):
    checkpoint_dictionary = {'hyper_params': self.hyper_params, 'model_state_dict': self.state_dict()}                                                
    if optimizer is not None:
      checkpoint_dictionary['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
      checkpoint_dictionary['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint_dictionary, path)

  @staticmethod
  def load_pretrained(path, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    hyper_params = checkpoint['hyper_params']
    model = SlimCNN(hyper_params['filter_count_values'], hyper_params['initial_conv_filters'], hyper_params['num_classes'], hyper_params['depth_multiplier'])
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
      scheduler.load_state_dict[checkpoint['scheduler_state_dict']]

    if any([optimizer, scheduler]):
      return model, [optimizer, scheduler]
    else:
      return model

