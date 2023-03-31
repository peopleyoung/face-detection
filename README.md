# YOLOX目标检测

## 项目相关代码

## 项目相关环境

## 项目部署流程

```
#拷贝代码,并把模型拷进当前根目录

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

