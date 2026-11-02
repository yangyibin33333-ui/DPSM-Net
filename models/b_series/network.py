import torch
from torch import nn
from torch.nn import functional as F
from .reference_blocks import ConvBNAct, DetailPreservingHalfStem, SubpixelDetailReconstruction, RIFTStructuralDirectionalModelingBlock

class DoubleConv(nn.Sequential):
    def __init__(self, ci, co):
        super().__init__(ConvBNAct(ci,co),ConvBNAct(co,co))

class SlimCNN(nn.Module):
    def __init__(self, version):
        super().__init__()
        if version not in ('B0','B1','B2','B2-Conv'): raise ValueError(version)
        self.version=version
        # Register every shared tensor before adding variant-specific modules.
        self.stage1=DoubleConv(40,40)
        self.down1=ConvBNAct(40,80,stride=2 if version=='B0' else 1)
        self.stage2=DoubleConv(80,80)
        self.down2=ConvBNAct(80,160,stride=2)
        self.stage3=DoubleConv(160,160)
        self.down3=ConvBNAct(160,320,stride=2)
        self.stage4=DoubleConv(320,320)
        self.decode3=DoubleConv(480,160)
        self.decode2=DoubleConv(240,80)
        self.decode1=DoubleConv(120,40)
        self.mask=nn.Conv2d(40,1,1)
        if version=='B0': self.stem=ConvBNAct(3,40)
        else:
            self.stem=DetailPreservingHalfStem(3,40,16)
            self.reconstruct=SubpixelDetailReconstruction(40,16)
        if version=='B2': self.rsdm=RIFTStructuralDirectionalModelingBlock(40)
        if version=='B2-Conv': self.rsdm=PlainConvResidual(40,23)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or x.shape[-2]%8 or x.shape[-1]%8:
            raise ValueError('Expected RGB input with H,W divisible by 8')
        if self.version=='B0':
            early=self.stage1(self.stem(x));f1=early
        else:
            detail,half=self.stem(x)
            early=self.stage1(half)
            if self.version in ('B2','B2-Conv'): early=self.rsdm(early)
            f1=self.reconstruct(detail,early)
        f2=self.stage2(self.down1(early))
        f3=self.stage3(self.down2(f2))
        f4=self.stage4(self.down3(f3))
        up=lambda a,b:F.interpolate(a,size=b.shape[-2:],mode='bilinear',align_corners=False)
        d3=self.decode3(torch.cat([up(f4,f3),f3],1))
        d2=self.decode2(torch.cat([up(d3,f2),f2],1))
        d1=self.decode1(torch.cat([up(d2,f1),f1],1))
        return self.mask(d1)

class PlainConvResidual(nn.Module):
    """Budget control: only 1x1, ordinary 3x3, 1x1; no directional gate."""
    def __init__(self,channels=40,hidden=23):
        super().__init__()
        if channels!=40 or hidden!=23:raise ValueError('This frozen control was verified for C40/hidden23 only')
        self.pre_norm=nn.GroupNorm(8,channels)
        self.reduce=nn.Conv2d(channels,hidden,1,bias=False)
        self.spatial=nn.Conv2d(hidden,hidden,3,padding=1,bias=False)
        self.mix_norm=nn.GroupNorm(1,hidden)
        self.act=nn.GELU()
        self.expand=nn.Conv2d(hidden,channels,1,bias=False)
    def forward(self,x):
        z=self.spatial(self.reduce(self.pre_norm(x)))
        return x+self.expand(self.act(self.mix_norm(z)))

def build_model(name):return SlimCNN(name)
