from m2t2.backbones.pointnet2_base import PointNet2Base
from m2t2.backbones.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG


class PointNet2MSG(PointNet2Base):
    def __init__(
        self, num_points, downsample, radius,
        radius_mult, use_rgb=True, norm='BN'
    ):
        super(PointNet2MSG, self).__init__()

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
        radius = radius * radius_mult

        num_points = num_points // downsample
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[radius, radius * radius_mult],
                nsamples=[16, 32],
                mlps=[[c_out_2, 256, 256, 512], [c_out_2, 256, 256, 512]],
                norm=norm
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + c_in, 128, 128])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + c_out_0, 256, 256])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + c_out_1, 512, 512])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512])
        )

        self.out_channels = {
            'res0': 128, 'res1': 256, 'res2': 512, 'res3': 512, 'res4': 1024
        }
