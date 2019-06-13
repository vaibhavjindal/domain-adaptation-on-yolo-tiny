from torch import nn
import torch.nn.functional as F

from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx,x,l):
        ctx.l = l
        return x.view_as(x)

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.neg()*ctx.l,None


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

    def forward(self, p, img_size, var=None):
        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls
            io[..., :4] *= self.stride
            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
return io.view(bs, -1, 5 + self.nc), p


def create_grids(self, img_size=416, ng=(13, 13), device='cpu'):
    nx, ny = ng  # x and y grid size
    self.img_size = img_size
    self.stride = img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).float().view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor=1, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(16, 32, 3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2) 

        self.conv5 = nn.Conv2d(32, 64, 3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.pool6 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv7 = nn.Conv2d(64, 128, 3,stride=1,padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.pool8 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv9 = nn.Conv2d(128, 256, 3,stride=1,padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        self.pool10 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv11 = nn.Conv2d(256, 512, 3,stride=1,padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.debug_pad12 = nn.ZeroPad2d((0,1,0,1))
        self.pool12 = nn.MaxPool2d(kernel_size=2,stride=1)

        self.conv13 = nn.Conv2d(512, 1024, 3,stride=1,padding=1)
        self.bn13 = nn.BatchNorm2d(1024)

        self.conv14 = nn.Conv2d(1024, 256, 1,stride=1,padding=0)
        self.bn14 = nn.BatchNorm2d(256)

        self.conv15 = nn.Conv2d(256, 512, 3,stride=1,padding=1)
        self.bn15 = nn.BatchNorm2d(512)

        self.conv16 = nn.Conv2d(512, 33, 1,stride=1,padding=0)

        self.yolo17_1 = YOLOLayer([(81,82),(135,169),(344,319)],6,608)

        self.conv19 = nn.Conv2d(256, 128, 1,stride=1,padding=0)
        self.bn19 = nn.BatchNorm2d(128)

        self.upsample20 = Upsample(scale_factor=2,mode="nearest")

        self.conv22 = nn.Conv2d(256, 256, 3,stride=1,padding=1)
        self.bn22 = nn.BatchNorm2d(256)

        self.conv23 = nn.Conv2d(256, 33, 1,stride=1,padding=0)

        self.yolo24_2 = YOLOLayer([(10,14),(23,27),(37,58)],6,608)

    def forward(self, x):
        img_size = max(x.shape[-2:])
        output = []

        x = self.bn1(self.conv1(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.pool2(x)

        x = self.bn3(self.conv3(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.pool4(x)

        x = self.bn5(self.conv5(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.pool6(x)

        x = self.bn7(self.conv7(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.pool8(x)
        x2 = x

        x = self.bn9(self.conv9(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.pool10(x)

        x = self.bn11(self.conv11(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.debug_pad12(x)
        x = self.pool12(x)

        x = self.bn13(self.conv13(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.bn14(self.conv14(x)) 
        x = F.leaky_relu(x,0.1,True)
        x1 = x

        x = self.bn15(self.conv15(x)) 
        x = F.leaky_relu(x,0.1,True)

        x = self.conv16(x)

        out1 = self.yolo17_1(x,img_size)
        output.append(out1)

        x = x1#layer 18

        x = self.bn19(self.conv19(x))
        x = F.leaky_relu(x,0.1,True)

        x = self.upsample20(x)

        x = torch.cat([x,x2],1)#layer 21

        x = self.bn22(self.conv22(x))
        x = F.leaky_relu(x,0.1,True)

        x = self.conv23(x)

        out2 = self.yolo24_2(x,img_size)
        output.append(out2)

        if self.training:
            return output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p

















