from torch import nn

from .depth_model import DepthNet
from .pose_model import PoseNetResNet


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class UnDeepVO(nn.Module):
    def __init__(self, max_depth=100, min_depth=1, inverse_sigmoid=False):
        super(UnDeepVO, self).__init__()
        self.depth_net = DepthNet(max_depth=max_depth, min_depth=min_depth, inverse_sigmoid=inverse_sigmoid)
        self.pose_net = PoseNetResNet()
        self.depth_net.apply(init_weights)

    def depth(self, x):
        out = self.depth_net(x)
        return out

    def pose(self, x, reference_frame):
        (out_rotation, out_translation) = self.pose_net(x, reference_frame)
        return out_rotation, out_translation

    def forward(self, x, reference_frame):
        return self.depth(x), self.pose(x, reference_frame)
