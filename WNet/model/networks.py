import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from dcn_v2 import DCN
from .resnet import resnet18
from .dat import DAT


def init_method(net, init_type='normal'):
    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'resnet'):
            pass
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=0.02)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
    
def init_net(net, init_type='normal', initialize=True, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net)
    if initialize:
        init_method(net, init_type)
    else:
        pass
    return net
    
    
def define_model(model_type='WNet', resnet='resnet18', init_type='normal', initialize=True, gpu_ids=[]):

    if model_type == 'WNet':
        net = WNet(resnet=resnet)
    else:
        raise NotImplementedError
    print_network(net)

    return init_net(net, init_type, initialize, gpu_ids)

    
class DeformableConv(nn.Module):
    def __init__(self, num_filters=64):
        super(DeformableConv, self).__init__()
        self.offset = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(num_filters))
        self.dcpack_L2 = DCN(num_filters, num_filters, 3, stride=1, padding=1, dilation=1, deformable_groups=4,
                             extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fea):

        offset = self.offset(fea)
        fea_mm = self.relu(self.dcpack_L2([fea, offset], None))       
        fea = fea + fea_mm
        return fea


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
class ChannelAttention(nn.Module):
    def __init__(self, input_nc, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(input_nc, input_nc // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(input_nc // ratio, input_nc, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 
    

class CTFM(nn.Module):
    def __init__(self, input_nc, ratio=8):
        super(CTFM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(input_nc, input_nc//2, 1, bias=False)
        self.conv2 = nn.Conv2d(input_nc, input_nc//2, 1, bias=False)

        self.fc1 = nn.Conv2d(input_nc, input_nc//ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(input_nc//ratio, input_nc, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = x1 + x2
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        fea = avg_out + max_out
        fea = self.sigmoid(fea)

        x1 = x1 * fea
        x2 = x2 * fea
        out = torch.cat([x1, x2], dim=1)

        return out
 
    
class WNet(nn.Module):

    def __init__(self, resnet='resnet18'):
        super().__init__()

        self.resnet = resnet18()
        # if pretrained:
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
            
        self.backbone = DAT()  # [96, 192, 384, 768]
        path = './pretrained/dat_tiny_in1k_224.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.deconv_1 = DeformableConv(num_filters=64)
        self.deconv_2 = DeformableConv(num_filters=128)
        self.deconv_3 = DeformableConv(num_filters=256)
        self.deconv_4 = DeformableConv(num_filters=512)
        
        self.ctf_1 = CTFM(input_nc=64, ratio=8)
        self.ctf_2 = CTFM(input_nc=128, ratio=8)
        self.ctf_3 = CTFM(input_nc=256, ratio=8)
        self.ctf_4 = CTFM(input_nc=512, ratio=8)
        
        self.down4 = nn.Conv2d(768, 512, 1, 1, padding=0)
        self.down3 = nn.Conv2d(384, 256, 1, 1, padding=0)
        self.down2 = nn.Conv2d(192, 128, 1, 1, padding=0)
        self.down1 = nn.Conv2d(96, 64, 1, 1, padding=0)
        
        self.datdown4 = nn.Conv2d(1536, 512, 1, 1, padding=0)
        self.datdown3 = nn.Conv2d(768, 256, 1, 1, padding=0)
        self.datdown2 = nn.Conv2d(384, 128, 1, 1, padding=0)
        self.datdown1 = nn.Conv2d(192, 64, 1, 1, padding=0)
        
        self.dcndown4 = nn.Conv2d(1536, 512, 1, 1, padding=0)
        self.dcndown3 = nn.Conv2d(768, 256, 1, 1, padding=0)
        self.dcndown2 = nn.Conv2d(384, 128, 1, 1, padding=0)
        self.dcndown1 = nn.Conv2d(192, 64, 1, 1, padding=0)
        
        self.conv_4to3 = nn.Conv2d(1024, 256, 1, 1, padding=0)
        self.conv_3to2 = nn.Conv2d(768, 128, 1, 1, padding=0)
        self.conv_2to1 = nn.Conv2d(384, 64, 1, 1, padding=0)
        self.conv_1to0 = nn.Conv2d(192, 32, 1, 1, padding=0)

        self.classifier = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 2, 3, bn=False, relu=False)
            )

    def resnet_forward(self, x):
        fea0 = self.resnet.conv1(x)
        fea0 = self.resnet.bn1(fea0)
        fea0 = self.resnet.relu(fea0)
        fea0 = self.resnet.maxpool(fea0)
       
        fea1 = self.resnet.layer1(fea0)
        fea1 = self.deconv_1(fea1)
        
        fea2 = self.resnet.layer2(fea1)
        fea2 = self.deconv_2(fea2)
        
        fea3 = self.resnet.layer3(fea2)
        fea3 = self.deconv_3(fea3)

        fea4 = self.resnet.layer4(fea3)
        fea4 = self.deconv_4(fea4)

        return fea1, fea2, fea3, fea4

    def forward(self, t1_img, t2_img):
        
        t1_dat1, t1_dat2, t1_dat3, t1_dat4 = self.backbone(t1_img)
        t2_dat1, t2_dat2, t2_dat3, t2_dat4 = self.backbone(t2_img)
        
        t1_dat1 = self.down1(t1_dat1)
        t2_dat1 = self.down1(t2_dat1)
        t1_dat2 = self.down2(t1_dat2)
        t2_dat2 = self.down2(t2_dat2)
        t1_dat3 = self.down3(t1_dat3)
        t2_dat3 = self.down3(t2_dat3)
        t1_dat4 = self.down4(t1_dat4)
        t2_dat4 = self.down4(t2_dat4)
        
        t1_fea1, t1_fea2, t1_fea3, t1_fea4 = self.resnet_forward(t1_img)
        t2_fea1, t2_fea2, t2_fea3, t2_fea4 = self.resnet_forward(t2_img)
        
        diff_dat4 = torch.abs(t2_dat4 - t1_dat4)
        dat4 = torch.cat([t1_dat4, t2_dat4, diff_dat4], dim=1)
        dat4 = self.datdown4(dat4)
        diff_dcn4 = torch.abs(t2_fea4 - t1_fea4)
        fea4 = torch.cat([t1_fea4, t2_fea4, diff_dcn4], dim=1)
        fea4 = self.dcndown4(fea4)
        fus4 = self.ctf_4(dat4, fea4)
        fus4 = F.interpolate(fus4, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        fus4 = self.conv_4to3(fus4)
        
        diff_dat3 = torch.abs(t2_dat3 - t1_dat3)
        dat3 = torch.cat([t1_dat3, t2_dat3, diff_dat3], dim=1)
        dat3 = self.datdown3(dat3)
        diff_dcn3 = torch.abs(t2_fea3 - t1_fea3)
        fea3 = torch.cat([t1_fea3, t2_fea3, diff_dcn3], dim=1)
        fea3 = self.dcndown3(fea3)
        fus3 = self.ctf_3(dat3, fea3)
        fea3 = torch.cat([fus3, fus4], dim=1)
        fea3 = F.interpolate(fea3, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        fus3 = self.conv_3to2(fea3)

        diff_dat2 = torch.abs(t2_dat2 - t1_dat2)
        dat2 = torch.cat([t1_dat2, t2_dat2, diff_dat2], dim=1)
        dat2 = self.datdown2(dat2)
        diff_dcn2 = torch.abs(t2_fea2 - t1_fea2)
        fea2 = torch.cat([t1_fea2, t2_fea2, diff_dcn2], dim=1)
        fea2 = self.dcndown2(fea2)
        fus2 = self.ctf_2(dat2, fea2)
        fea2 = torch.cat([fus2, fus3], dim=1)
        fea2 = F.interpolate(fea2, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        fus2 = self.conv_2to1(fea2)

        diff_dat1 = torch.abs(t2_dat1 - t1_dat1)
        dat1 = torch.cat([t1_dat1, t2_dat1, diff_dat1], dim=1)
        dat1 = self.datdown1(dat1)
        diff_dcn1 = torch.abs(t2_fea1 - t1_fea1)
        fea1 = torch.cat([t1_fea1, t2_fea1, diff_dcn1], dim=1)
        fea1 = self.dcndown1(fea1)
        fus1 = self.ctf_1(dat1, fea1)
        fus = torch.cat([fus1, fus2], dim=1)
        fus = F.interpolate(fus, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        fus = self.conv_1to0(fus)

        pred = self.classifier(fus)
        return pred
 