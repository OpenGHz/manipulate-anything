from m2t2.backbones.pointnet2_msg import PointNet2MSG
from m2t2.backbones.pointnet2_msg_cls import PointNet2MSGCls


model_registry = {
    'pointnet2_msg': PointNet2MSG,
    'pointnet2_msg_cls': PointNet2MSGCls
}


def build_backbone(cfg):
    model = model_registry[cfg.type]
    return model(
        cfg.num_points, cfg.downsample, cfg.radius,
        cfg.radius_mult, cfg.use_rgb
    )
