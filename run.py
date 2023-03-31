import importlib
import yaml
from pipline_probe import PipelineProbe
import argparse
import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)  #（代表仅使用第0，1号GPU）
# from nvr_post import Easynvr
# from kafka import KafkaProducer, KafkaConsumer
# from kafka import TopicPartition

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='configs/config.yaml')
    
    parser.add_argument("-sync",
                        type=int,
                        default=0,
                        help="1 for video input sync")
    parser.add_argument("--tiler", action='store_true')

    configs, remaining = parser.parse_known_args()

    with open(configs.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    opt = vars(configs)
    parser.set_defaults(**opt)
    args = parser.parse_args()

    # nvr=Easynvr(args.nvr_url,args.nvr_user,args.nvr_pwd)
    # 从组中获取所有原始视频流
    # cameras=nvr.get_group_list(1)

    # if args.index==-1:
    #     topic="position"
    #     consumer = KafkaConsumer(topic,auto_offset_reset='latest',bootstrap_servers=args.kafka)
    #     partitions=consumer.partitions_for_topic(topic)

    #     latest_time=0
    #     msg=None
    #     for i in partitions:
    #         partition = TopicPartition(topic, i)
    #         offset=consumer.position(partition)

    #         consumer.seek(partition, offset-1)
    #         for msg in consumer:
    #             if msg.timestamp>latest_time:
    #                 latest_time=msg.timestamp
    #                 msg=msg.value
    #             break
    #     msg=eval(str(msg,encoding = "utf-8"))
    #     args.index=msg["num"]
    #     consumer.close()
    # urls=[]
    # for i in range(args.camera_num):
    #     channel_id=cameras[(args.index+i)%len(cameras)]["ChannelId"]
    #     url=nvr.get_channel_stream(int(channel_id))
    #     urls.append(url)
    # print(urls)
    # urls = ["rtsp://admin:zmj123456@192.168.116.13"]
    urls = ["rtsp://172.16.66.245:5548/live/stream_20"]  # ,
    pipeline = PipelineProbe(args, urls)
    pipeline.run()
