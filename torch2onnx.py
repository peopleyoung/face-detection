import numpy as np
import torch

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from utils.config import cfg_mnet, cfg_re50
from loguru import logger


#--------------------------------------#
#   一定注意backbone和model_path的对应。
#   在更换facenet_model后，
#   一定要注意重新编码人脸。
#--------------------------------------#
class Retinaface(object):
    _defaults = {
        #----------------------------------------------------------------------#
        #   retinaface训练完的权值路径
        #----------------------------------------------------------------------#
        "retinaface_model_path": 'model_data/Retinaface_mobilenet0.25.pth',
        #----------------------------------------------------------------------#
        #   retinaface所使用的主干网络，有mobilenet和resnet50
        #----------------------------------------------------------------------#
        "retinaface_backbone": "mobilenet",
        #----------------------------------------------------------------------#
        #   retinaface中只有得分大于置信度的预测框会被保留下来
        #----------------------------------------------------------------------#
        "confidence": 0.5,
        #----------------------------------------------------------------------#
        #   retinaface中非极大抑制所用到的nms_iou大小
        #----------------------------------------------------------------------#
        "nms_iou": 0.3,
        #----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        #----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        #----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #----------------------------------------------------------------------#
        "letterbox_image": True,

        #----------------------------------------------------------------------#
        #   facenet训练完的权值路径
        #----------------------------------------------------------------------#
        "facenet_model_path": 'model_data/facenet_mobilenet.pth',
        #----------------------------------------------------------------------#
        #   facenet所使用的主干网络， mobilenet和inception_resnetv1
        #----------------------------------------------------------------------#
        "facenet_backbone": "mobilenet",
        #----------------------------------------------------------------------#
        #   facenet所使用到的输入图片大小
        #----------------------------------------------------------------------#
        "facenet_input_shape": [160, 160, 3],
        #----------------------------------------------------------------------#
        #   facenet所使用的人脸距离门限
        #----------------------------------------------------------------------#
        "facenet_threhold": 0.9,

        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinaface
    #---------------------------------------------------#
    def __init__(self, encoding=0, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   不同主干网络的config信息
        #---------------------------------------------------#
        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net = RetinaFace(cfg=self.cfg, phase='eval',
                              pre_train=False).eval()
        self.facenet = Facenet(backbone=self.facenet_backbone,
                               mode="predict").eval()
        device = torch.device('cuda' if self.cuda else 'cpu')

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path,
                                map_location=device)
        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path, map_location=device)
        self.facenet.load_state_dict(state_dict, strict=False)
        print('Finished!')
        return self.net, self.facenet


def main(retinaface_model, facenet_model):
    retinaface_model.eval().cuda()
    #
    x = torch.ones(1, 3, 640, 640).cuda()
    onnx_path = 'retinaface_model.onnx'
    logger.info('\t>>write onnx: {}'.format(onnx_path))
    torch.onnx.export(retinaface_model,
                      (x),
                      onnx_path,
                      verbose=False,
                      export_params=True,
                      opset_version=11,
                      input_names=["input"], output_names=["output_loc", "output_conf", "output_landms"])

    logger.info("Converted onnx model engine file finished.")

    facenet_model.eval().cuda()
    #
    x = torch.ones(1, 3, 160, 160).cuda()
    onnx_path = 'facenet_model.onnx'
    logger.info('\t>>write onnx: {}'.format(onnx_path))
    torch.onnx.export(facenet_model,
                      (x),
                      onnx_path,
                      verbose=False,
                      export_params=True,
                      opset_version=11,
                      input_names=["input"], output_names=["output"])

    logger.info("Converted onnx model engine file finished.")


if __name__ == '__main__':
    retinaface = Retinaface()
    retinaface_model, facenet_model = retinaface.generate()
    main(retinaface_model, facenet_model)
