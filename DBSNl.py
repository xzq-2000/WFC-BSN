import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt.data
from torch.autograd import Function
from torchvision import transforms
from PIL import Image
import os
class DepthWiseConv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,dilation):
 
        #这一行千万不要忘记
        super(DepthWiseConv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=2*dilation-1,
                                    dilation=dilation,
                                    padding='same',
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
 
        #逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        """ 通道注意力机制 同最大池化和平均池化两路分别提取信息，后共用一个多层感知机mlp,再将二者结合
        :param in_channel: 输入通道
        :param ratio: 通道降低倍率
        """
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道先降维后恢复到原来的维数
        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		# out = avg_out + max_out
        # return x*self.sigmoid(out)
        
        # 平均池化一支 (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,8,8) -> (2,512,1,1)
        avg = self.avg_pool(x)
        # 多层感知机mlp (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,1,1) -> (2,512/ratio,1,1)
        avg = self.fc1(avg)
        avg = self.relu1(avg)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        avg_out = self.fc2(avg)

        # 最大池化一支
        # (2,512,8,8) -> (2,512,1,1)
        max = self.max_pool(x)
        # 多层感知机
        # (2,512,1,1) -> (2,512/ratio,1,1)
        max = self.fc1(max)
        max = self.relu1(max)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        max_out = self.fc2(max)

        # (2,512,1,1) + (2,512,1,1) -> (2,512,1,1)
        out = avg_out + max_out
        return x*self.sigmoid(out)

class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, d, reduction=True):
        super(DF_Module, self).__init__()
        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        # 使用共享的卷积层
        self.a = nn.Parameter(torch.ones([]))
        self.b = nn.Parameter(torch.ones([]))
        self.c = nn.Parameter(torch.ones([]))
        self.d = nn.Parameter(torch.ones([]))
        
        self.conv0 = nn.Conv2d(dim_in//4, dim_out//4 , kernel_size=1, dilation=d, padding='same',groups=dim_in//4)
        self.conv1 = nn.Conv2d(dim_in//4, dim_out//4 , kernel_size=1, dilation=d, padding='same',groups=dim_in//4)
        self.conv2 = nn.Conv2d(dim_in//4, dim_out//4, kernel_size=1, dilation=d, padding='same',groups=dim_in//4)
        self.conv3 = nn.Conv2d(dim_in//4, dim_out//4 , kernel_size=1, dilation=d, padding='same',groups=dim_in//4)
        self.conv4 = nn.Conv2d(dim_out//4, dim_out , kernel_size=2*d-1, dilation=d, padding='same')
        '''
        self.conv0=DepthWiseConv(dim_in//4,dim_out//4,kernel_size=1,dilation=d)
        self.conv1=DepthWiseConv(dim_in//4,dim_out//4,kernel_size=1,dilation=d)
        self.conv2=DepthWiseConv(dim_in//4,dim_out//4,kernel_size=1,dilation=d)
        self.conv3=DepthWiseConv(dim_in//4,dim_out//4,kernel_size=1,dilation=d)
        self.conv4=DepthWiseConv(dim_in//4,dim_out,kernel_size=2*d-1,dilation=d)
        '''
    def forward(self, x, y):
        #print(x.shape)
        # 计算 x 和 y 的加法和差异
        add = torch.abs(x + y)
        dif = torch.abs(x - y)
        # 共享卷积操作
        #x = self.relu0(self.conv0(x))
        #y = self.relu1(self.conv1(y))
        #add = self.relu2(self.conv2(add))
        #dif = self.relu3(self.conv3(dif))
        #x = self.relu0(x)
        #y = self.relu1(y)
        #add = self.relu2(add)
        #dif = self.relu3(dif)
        # 合并特征图
        #print(self.relu0(self.conv0(x)).shape)
        #out = torch.cat((self.relu0(self.conv0(x)),  self.relu1(self.conv1(y)), self.relu2(self.conv2(add)), self.relu3(self.conv3(dif))), dim=1)
        #print(out.shape)
        #out=self.conv4(weight_x*x+weight_y*y+weight_add*add+weight_dif*dif)
        #out=self.conv4(x+y+add+dif)
        #print(weight_x)
        #out=torch.stack([x, y, add, dif])
        #print(out.shape)
        #out=weight_dif*torch.abs(x-y)+weight_add*torch.abs(x+y)+weight_x*x+weight_y*y
        out=torch.cat((x,y,add,dif),dim=1)
        return out
#生成4*in_size,1,2,2的小波卷积核
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    #print(11111,dec_hi,dec_lo)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    #print( dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),dec_filters.shape)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    #print(11111,rec_hi,rec_lo)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters
#wavelet_transform 函数实现了一个小波变换过程，它使用给定的滤波器对输入数据进行卷积，然后将结果重塑为包含四个频带的形式。
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

#实现了逆小波变换过程，它使用给定的滤波器对经过小波变换的数据进行转置卷积，然后将结果重塑为原始数据的形状。
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


def wavelet_transform_init(filters):
    class WaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = inverse_wavelet_transform(grad_output, filters)
            return grad, None

    return WaveletTransform().apply# 返回一个应用了 WaveletTransform 的函数。这个返回值可以像普通的 PyTorch 函数一样被调用，并且它会自动处理前向和反向传播。

#在前向传播时执行逆小波变换，在反向传播时应该再次执行逆小波变换
def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):

        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                x = inverse_wavelet_transform(input, filters)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad = wavelet_transform(grad_output, filters)
            return grad, None

    return InverseWaveletTransform().apply
'''
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, d, bias=True, wt_type='db1'):#'db1'
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels
        
        self.in_channels = in_channels
        self.dilation = d
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)#[batch*4,1,2,2]
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)
        self.base_conv = nn.Conv2d(in_channels, in_channels, 2*d-1, padding='same', stride=1, dilation=d,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.Conv2d(in_channels * 4, in_channels * 4, 2*d-1, padding='same', stride=1, dilation=d,
                       groups=in_channels * 4, bias=False)
        self.dfm=DF_Module(out_channels,out_channels,d)
    def forward(self, x):
        curr_x_ll = x#8,32,40,40
        #小波分解
        curr_x = self.wt_function(curr_x_ll)#8,32,4,20,20
        shape_x = curr_x.shape
        #将小波分解的四张图片cat到一起
        curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])#8,128,20,20
        #使用扩张卷积
        curr_x_tag = self.wavelet_convs(curr_x_tag)#8,128,20,20
        #恢复四张图片
        curr_x_tag = curr_x_tag.reshape(shape_x)#8,32,4,20,20
        #反小波卷积
        x1=self.iwt_function(curr_x_tag)#8,32,40,40
        #原图直接卷积
        #x2 = self.base_scale(self.base_conv(x))#8,32,40,40
        x2=self.base_scale(x)
        #多尺度融合
        #y=x1+x2
        y = self.dfm(x1,x2)
        return y
'''
def toimagef1(b,i):
    b = b.squeeze(0)# 因为b1是单通道，所以不需要循环
    b1=b[0,0,:,:]
    b2=b[0,1,:,:]
    b3=b[0,2,:,:]
    b4=b[0,3,:,:]
    tensors1 = [ b1, b1,b1]
    tensors2 = [ b2, b2,b2]
    tensors3 = [ b3, b3,b3]
    tensors4 = [ b4, b4,b4]
    # 使用torch.stack沿着新的维度堆叠这些张量
    out1 = torch.stack(tensors1, dim=0)
    out2 = torch.stack(tensors2, dim=0)
    out3 = torch.stack(tensors3, dim=0)
    out4 = torch.stack(tensors4, dim=0)

    out1= out1.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    out2= out2.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    out3= out3.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    out4= out4.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    # 将张量转换为PIL图像
    image_pil1 = transforms.ToPILImage()(out1)
    image_pil2 = transforms.ToPILImage()(out2)
    image_pil3 = transforms.ToPILImage()(out3)
    image_pil4 = transforms.ToPILImage()(out4)
    # 保存图像
    output_dir = 'lunwen/10'
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    image_pil1.save(f'lunwen1/f1/output_image{i}_1.jpg')
    image_pil2.save(f'lunwen1/f1/output_image{i}_2.jpg')
    image_pil3.save(f'lunwen1/f1/output_image{i}_3.jpg')
    image_pil4.save(f'lunwen1/f1/output_image{i}_4.jpg')
def toimagef2(b,i):
    b = b.squeeze(0)# 因为b1是单通道，所以不需要循环
    b1=b[0,0,:,:]
    b2=b[0,1,:,:]
    b3=b[0,2,:,:]
    b4=b[0,3,:,:]
    tensors1 = [ b1, b1,b1]
    tensors2 = [ b2, b2,b2]
    tensors3 = [ b3, b3,b3]
    tensors4 = [ b4, b4,b4]
    # 使用torch.stack沿着新的维度堆叠这些张量
    out1 = torch.stack(tensors1, dim=0)
    out2 = torch.stack(tensors2, dim=0)
    out3 = torch.stack(tensors3, dim=0)
    out4 = torch.stack(tensors4, dim=0)

    out1= out1.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    out2= out2.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    out3= out3.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    out4= out4.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    # 将张量转换为PIL图像
    image_pil1 = transforms.ToPILImage()(out1)
    image_pil2 = transforms.ToPILImage()(out2)
    image_pil3 = transforms.ToPILImage()(out3)
    image_pil4 = transforms.ToPILImage()(out4)
    # 保存图像
    output_dir = 'lunwen/10'
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    image_pil1.save(f'lunwen1/f2/output_image{i}_1.jpg')
    image_pil2.save(f'lunwen1/f2/output_image{i}_2.jpg')
    image_pil3.save(f'lunwen1/f2/output_image{i}_3.jpg')
    image_pil4.save(f'lunwen1/f2/output_image{i}_4.jpg')
def toimage(b1,i):
    b1 = b1.squeeze(0)[0]# 因为b1是单通道，所以不需要循环
    tensors1 = [ b1, b1,b1]
    b1=torch.stack(tensors1, dim=0)
    b1 = b1.mul(255).byte()  # 将像素值从[0.0, 1.0]缩放回[0, 255]
    # 将张量转换为PIL图像
    image_pil = transforms.ToPILImage()(b1)
    # 保存图像
    image_pil.save(f'lunwen/x1/output_image{i}.jpg')
global tt
tt=0
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, d, bias=True, wt_type='db1'):#'db1'
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.dilation = d
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)#[batch*4,1,2,2]
        self.iwt_filter1=self.iwt_filter[:self.iwt_filter.shape[0]//4,:,:,:]
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_filter1 = nn.Parameter(self.iwt_filter1, requires_grad=False)
        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)
        self.iwt_function1 = inverse_wavelet_transform_init(self.iwt_filter1)
        self.base_conv = nn.Conv2d(in_channels, in_channels//4, 2*d-1, padding='same', stride=1, dilation=d,
                                   groups=in_channels//4, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels//4, 1, 1])

        self.wavelet_convs = nn.Conv2d(in_channels * 4, in_channels, 2*d-1, padding='same', stride=1, dilation=d,
                       groups=in_channels, bias=False)
        self.dfm=DF_Module(out_channels,out_channels,d)
    def forward(self, x):
        curr_x_ll = x#8,32,40,40
        #小波分解
        curr_x = self.wt_function(curr_x_ll)#8,32,4,20,20
        global tt
        tt=tt+1
        #toimagef1(curr_x,tt)
        shape_x = curr_x.shape
        shape_x1=[shape_x[0],shape_x[1]//4,shape_x[2],shape_x[3],shape_x[4]]
        #将小波分解的四张图片cat到一起
        curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])#8,128,20,20

        #使用扩张卷积
        curr_x_tag = self.wavelet_convs(curr_x_tag)#8,128,20,20
        #恢复四张图片
        curr_x_tag = curr_x_tag.reshape(shape_x1)#8,32,4,20,20
        #toimagef2(curr_x_tag,tt)
        #反小波卷积
        x1=self.iwt_function1(curr_x_tag)#8,32,40,40
        #原图直接卷积
        x2 = self.base_scale(self.base_conv(x))#8,32,40,40
        
        #toimage(x1,tt)
        #x2=self.base_scale(x)
        #多尺度融合
        #y=x1+x2
        y = self.dfm(x1,x2)
        return y

#对输入数据进行缩放
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9,mask=True):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module,mask)
        self.branch2 = DC_branchl(3, base_ch, num_module,mask)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)
        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module,mask=True):
        super().__init__()
        ly = []
        self.mask=mask
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        # ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=2*stride-1, padding=stride-1)]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch,2*stride-1) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        if self.mask==True:
            self.body[0].set_mask(True)
        else:
            self.body[0].set_mask(False)
        return self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, in_ch,k):
        super().__init__()

        ly = []
        #ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        #ly += [ WTConv2d(in_ch, in_ch, kernel_size=k, dilation=stride,wt_levels=1) ]
        ly += [ WTConv2d(in_ch, in_ch, d=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=k, dilation=stride,padding='same') ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        #global tt
        #tt=tt+1
        #toimage(x,tt)
        #toimage(x+self.body(x),tt)
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def set_mask(self,enable):
        _, _, kH, kW = self.weight.size()
        if enable==True:
            self.mask.fill_(1)
            self.mask[:, :, kH//2, kH//2] = 0
        else:
            self.mask.fill_(1)
    def forward(self, x):
        self.weight.data =self.weight.data* self.mask
        return super().forward(x)
