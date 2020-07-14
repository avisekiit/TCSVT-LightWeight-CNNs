import torch.nn as nn
import torch



class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class LIST(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(LIST, self).__init__()
    self.squezze_layer=nn.Conv2d(input_channel, int(input_channel/4), kernel_size=1)
    self.fire_layer=nn.Conv2d(int(input_channel/4), int(output_channel/2), kernel_size=1)
    self.depthwise=depthwise_separable_conv(int(input_channel/4), int(output_channel/2))

  def forward(self, x):
    stream=self.squezze_layer(x)
    out2=self.fire_layer(stream)
    out1=self.depthwise(stream)
    return torch.cat((out1,out2),1)
