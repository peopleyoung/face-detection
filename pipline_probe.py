import torch
import datetime
import json
import numpy as np
import ctypes
import time
from PIL import Image, ImageDraw, ImageFont
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import (Alignment_1, compare_faces, letterbox_image,
                         preprocess_input)
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              retinaface_correct_boxes)
from trtmodel_infer import TRTWrapper
from app.utils.fps import FPSMonitor
from app.utils.is_aarch_64 import is_aarch64
from app.utils.bus_call import bus_call
import pyds
from gi.repository import GObject, Gst, GstRtspServer
from app.pipeline import Pipeline
import cv2
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')


class PipelineProbe(Pipeline):

    def __init__(self, args, urls):
        super().__init__(args, urls)
        if self.args.tiler:
            self.fps_streams = FPSMonitor(0)
        else:
            self.fps_streams = []
            for i in range(self.num_sources):
                self.fps_streams.append(FPSMonitor(i))

        self.roi_shape = args.roi_wh
        self.roi_start_wh = args.roi_start_wh
        self.confidence = args.confidence
        self.nms_iou = args.nms_iou
        self.facenet_threhold = args.facenet_threhold

        # load model
        self.retinaface = TRTWrapper(self.args.retinaface_model_path)
        self.facenet = TRTWrapper(self.args.facenet_model_path)
        #########
        self.retinaface_backbone = self.args.retinaface_backbone
        self.facenet_backbone = self.args.facenet_backbone
        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        self.anchors = Anchors(self.cfg, image_size=(
            self.args.retinaface_model_inputsize[1], self.args.retinaface_model_inputsize[2])).get_anchors().cuda()
        self.scale = [
            self.roi_shape[0],
            self.roi_shape[1],
            self.roi_shape[0],
            self.roi_shape[1]
        ]
        self.scale_for_landmarks = [
            self.roi_shape[0],
            self.roi_shape[1],
            self.roi_shape[0],
            self.roi_shape[1],
            self.roi_shape[0],
            self.roi_shape[1],
            self.roi_shape[0],
            self.roi_shape[1],
            self.roi_shape[0],
            self.roi_shape[1]
        ]

        if self.args.tiler:
            self.fps_streams = FPSMonitor(0)
        else:
            self.fps_streams = []
            for i in range(self.num_sources):
                self.fps_streams.append(FPSMonitor(i))

        self.known_face_encodings = np.load(
            "model_data/{backbone}_face_encoding.npy".format(
                backbone=self.facenet_backbone))
        self.known_face_names = np.load(
            "model_data/{backbone}_names.npy".format(
                backbone=self.facenet_backbone))

        self._add_probes()

    def change_url(self):
        msg = self.consumer.poll(timeout_ms=2)
        if not msg:
            return True
        for key, value in msg.items():
            res = value[-1]
        msg = eval(str(res.value, encoding="utf-8"))

        self.args.index = msg["num"]
        for i in range(self.args.camera_num):
            channel_id = self.cameras[(self.args.index + i) %
                                      len(self.cameras)]["ChannelId"]
            url = self.nvr.get_channel_stream(int(channel_id))
            self.stop_release_source(i)
            self.add_source(i, url)
        return True

    def delete_sources(self):
        dic = {0: (self.change_num + 1) % 2, 1: (self.change_num) % 2}
        self.change_num += 1
        # print(self.change_num)
        for i in range(0, 2):
            self.stop_release_source(i)
        # time.sleep(2)
        for i in range(0, 2):
            self.add_source(i, self.video_uri[dic[i]])

        return True


    def osd_sink_pad_buffer_probe(self, _, info, index):
        self.fps_streams[index].get_fps()
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            n_frame = pyds.get_nvds_buf_surface(
                hash(gst_buffer), frame_meta.batch_id)
            image = cv2.resize(n_frame[self.roi_start_wh[1]:self.roi_start_wh[1] + self.roi_shape[1],
                                       self.roi_start_wh[0]:self.roi_start_wh[0] + self.roi_shape[0], 0:3],
                               (self.args.retinaface_model_inputsize[1], self.args.retinaface_model_inputsize[2]))
            image = np.array(image, dtype=np.float32)
            model_input = torch.from_numpy(preprocess_input(
                image).transpose(2, 0, 1)).unsqueeze(0).cuda()
            output = self.retinaface(dict(input=model_input))

            loc, conf, landms = output['output_loc'], output['output_conf'], output['output_landms']

            boxes = decode(loc.data.squeeze(
                0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors,
                                  self.cfg['variance'])
            # -----------------------------------------------------------#
            #   对人脸检测结果进行堆叠
            # -----------------------------------------------------------#
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(
                boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms) > 0:
                boxes_conf_landms[:,
                                  :4] = boxes_conf_landms[:, :4] * self.scale
                boxes_conf_landms[:, 5:] = boxes_conf_landms[:,
                                                             5:] * self.scale_for_landmarks
                # -----------------------------------------------#
                #   Facenet编码部分-开始
                # -----------------------------------------------#
                face_encodings = []
                for boxes_conf_landm in boxes_conf_landms:
                    # ----------------------#
                    #   图像截取，人脸矫正
                    # ----------------------#
                    boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
                    crop_img = np.array(n_frame[self.roi_start_wh[1]:self.roi_start_wh[1] + self.roi_shape[1],
                                                self.roi_start_wh[0]:self.roi_start_wh[0] + self.roi_shape[0], 0:3])[
                                                    int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                                                    int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                    landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                        [int(boxes_conf_landm[0]),
                         int(boxes_conf_landm[1])])
                    crop_img, _ = Alignment_1(crop_img, landmark)

                    # ----------------------#
                    #   人脸编码
                    # ----------------------#
                    crop_img = cv2.resize(
                        crop_img, (self.args.facenet_model_inputsize[2], self.args.facenet_model_inputsize[1])) / 255
                    crop_img = torch.from_numpy(
                        crop_img.transpose(2, 0, 1)).unsqueeze(0).cuda()
                    face_encoding = self.facenet(dict(input=crop_img))
                    face_encoding = face_encoding['output'][0].cpu().numpy()
                    face_encodings.append(face_encoding)
                # -----------------------------------------------#
                #   人脸特征比对-开始
                # -----------------------------------------------#
                face_names = []
                for face_encoding in face_encodings:
                    # -----------------------------------------------------#
                    #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
                    # -----------------------------------------------------#
                    matches, face_distances = compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=self.facenet_threhold)
                    name = "Unknown"
                    # -----------------------------------------------------#
                    #   取出这个最近人脸的评分
                    #   取出当前输入进来的人脸，最接近的已知人脸的序号
                    # -----------------------------------------------------#
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    face_names.append(name)
                # -----------------------------------------------#
                #   人脸特征比对-结束
                # -----------------------------------------------#
                for i, b in enumerate(boxes_conf_landms):
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    # ---------------------------------------------------#
                    #   b[0]-b[3]为人脸框的坐标，b[4]为得分
                    # ---------------------------------------------------#
                    cv2.rectangle(n_frame, (b[0] + self.roi_start_wh[0], b[1] + self.roi_start_wh[1]),
                                  (b[2] + self.roi_start_wh[0], b[3] +
                                   self.roi_start_wh[1]), (0, 0, 255), 2)
                    cx = b[0] + self.roi_start_wh[0]
                    cy = b[1] - 12 + + self.roi_start_wh[1]
                    cv2.putText(n_frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX,
                                0.5, (255, 255, 255))

                    name = face_names[i]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(n_frame, name,
                                (b[2] + self.roi_start_wh[0], b[3] + self.roi_start_wh[1]), font, 0.5, (255, 255, 255), 1)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def _add_probes(self):
        if not self.args.tiler:
            for i in range(self.num_sources):
                rtsp_sinkpad = self._get_static_pad(self.rtsp_bin_list[i])
                rtsp_sinkpad.add_probe(Gst.PadProbeType.BUFFER,
                                       self.osd_sink_pad_buffer_probe, i)
