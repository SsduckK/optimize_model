import sys

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
import os.path as op
from glob import glob
import time

from .detect_module import MainDetector as MD
from ros_imgtime_msg.msg import Imagetime
from detecting_result_msg.msg import Result


class Detector(Node):
    def __init__(self, ckpt_list):
        super().__init__("observer")
        self.detector = MD(ckpt_list)
        self.image_received_time = 0
        self.before_model = 0
        self.after_model = 0
        self.image_subscriber = self.create_subscription(Imagetime, "sending_image", self.subscribe_image, 10)
        self.result_publisher = self.create_publisher(Result, "sending_result", 10)

    def subscribe_image(self, image_time):
        image = image_time.image
        self.image_received_time = image_time.timestamp
        msg_img = CvBridge().imgmsg_to_cv2(image, "bgr8")
        compressed_image = self.compress_image(msg_img, 80)
        self.before_model = time.time()
        detected_result = self.detector(msg_img)
        self.after_model = time.time()
        bboxes, classes, scores = self.get_bboxes_result(detected_result.pred_instances)
        self.publish_result(bboxes, classes, scores, self.image_received_time, self.before_model, self.after_model)

    def compress_image(self, image, compress_ratio):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_ratio]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        compress_image = cv2.imdecode(encimg, 1)
        return compress_image

    def get_bboxes_result(self, instances):
        bboxes = instances["bboxes"].cpu().numpy()
        classes = instances["labels"].cpu().numpy()
        scores = instances["scores"].cpu().numpy()
        bboxes_bytes = bboxes.tobytes()     # float32
        classes_bytes = classes.tobytes()     # int64
        scores_bytes = scores.tobytes()     # float32
        return bboxes_bytes, classes_bytes, scores_bytes

    def publish_result(self, bboxes, classes, scores, received_time, before_model_time, after_model_time):
        detection_result_timestamp = Result()
        detection_result_timestamp.timestamp = [received_time[0], before_model_time, after_model_time]
        detection_result_timestamp.bboxes = list(bboxes)
        detection_result_timestamp.classes = list(classes)
        detection_result_timestamp.scores = list(scores)
        self.result_publisher.publish(detection_result_timestamp)

def system_init():
    sub_package_path = os.path.dirname(os.path.abspath(__file__))
    if sub_package_path not in sys.path:
        sys.path.append(sub_package_path)

    package_path = os.path.dirname(os.path.abspath(__file__))
    if package_path not in sys.path:
        sys.path.append(package_path)


def main(args=None):
    system_init()
    rclpy.init(args=args)
    ckpt_list = [t for t in glob(op.join("/mnt/intHDD/mmdet_ckpt/yolov7", "*")) if "_base_" not in t]
    node = Detector(ckpt_list)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SigInt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()