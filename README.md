# YOLOX目标检测

## 项目相关代码

项目git地址:https://gitlab.zmjkf.cn/industrial-software/ze06-video-analysis/products/research/ds-keypoints-pose.git


## 项目相关环境

项目docker地址：[Harbor (zmjkf.cn)](https://harbor.zmjkf.cn/aivision/deepstream6.0.1:video-splicing-v1.0)

harbor账号:AIvision,密码:Zmj123456#

拉取命令：docker pull harbor.zmjkf.cn/aivision/deepstream6.0.1:video-splicing-v1.0

## 项目部署流程

```
#拷贝代码,并把模型拷进当前根目录
git clone https://gitlab.zmjkf.cn/industrial-software/ze06-video-analysis/products/research/ds-keypoints-pose.git
#启动一个容器，并做好配置文件夹映射，对应容器里文件夹为/ds-keypoints-pose/workdir，这里将其映射到本地/home/zzmj/ds-keypoints-pose/
docker run -it --name ds-keypoints-pose --gpus all --network host -v /home/zzmj/ds-keypoints-pose:/ds-keypoints-pose harbor.zmjkf.cn/aivision/deepstream6.0.1:video-splicing-v1.0 /bin/bash
#安装其他依赖环境
cd /ds-keypoints-pose
pip install -r requirements.txt
#把yolox模型转化成onnx,-b代表batchsize，这里为1
cd /ds-keypoints-pose/trt
python torch2onnx.py -f exps/example/mot/yolox_s_mix_det.py -c bytetrack_s_mot17.pth.tar -b 1
#把onnx模型转换为engine
/usr/src/tensorrt/bin/trtexec --onnx=yolox_model.onnx --saveEngine=model_yolox.engine --fp16 --explicitBatch --workspace=32
mv model_trt_6.engine ../configs/
# 把hrnet模型转化成onnx,-b代表batchsize，这里为1
python hrnet2onnx.py
/usr/src/tensorrt/bin/trtexec --onnx=hrnet_model.onnx --saveEngine=model_hrnet.engine --fp16 --explicitBatch --workspace=32
mv model_hrnet.engine ../configs/
#生成deepstream需要的动态链接库,nvdsparsebbox_yolox.cpp中第31行static const int NUM_CLASSES为需要的类别数量
cd ../nvdsinfer_custom_impl_yolox/
make
cd ../
python3 run.py
```

