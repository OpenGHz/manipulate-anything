from m2t2.backbones.pointnet2_base import PointNet2Base
from m2t2.backbones.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


class PointNet2MSGCls(PointNet2Base):
    def __init__(
        self, num_points, downsample, radius,
        radius_mult, use_rgb=True, norm='BN'
    ):
        super(PointNet2MSGCls, self).__init__()

        self.use_rgb = use_rgb
        c_in = 3 if use_rgb else 0
        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 32, 32, 64]],
                norm=norm
            )
        )
        c_out_0 = 64 + 64
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128], [c_out_0, 64, 64, 128]],
                norm=norm
            )
        )
        c_out_1 = 128 + 128
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_1, 128, 128, 256], [c_out_1, 128, 128, 256]],
                norm=norm
            )
        )
        c_out_2 = 256 + 256
        self.SA_modules.append(
            PointnetSAModule(mlp=[c_out_2, 256, 256, 512], norm=norm)
        )

        self.out_channels = {
            'res0': c_in, 'res1': 128, 'res2': 256, 'res3': 512, 'res4': 512
        }
