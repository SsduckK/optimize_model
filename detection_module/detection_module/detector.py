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
from mmdet.apis import init_detector, inference_detector

from .detect_module import MainDetector as MD
from rod_msg.msg import Rodmsg
from detecting_result_msg.msg import Result
from rclpy.time import Time
import utils

class Detector(Node):
    def __init__(self, ckpt_list, reserve_frames=5):
        super().__init__("server")
        self.detectors = self.load_detectors(ckpt_list)
        self.model_selection = 0
        self.compression = 0
        self.frame_meta = np.zeros((reserve_frames, 4))     # client to server time, detection time, model selection, compression
        self.meta_index = 0
        self.metric_by_model = {}   # {'{model_name}_{compression}': [TP, FP, FN]}
        self.image_subscriber = self.create_subscription(Image, "sending_image", self.subscribe_image, 10)
        self.result_publisher = self.create_publisher(Result, "sending_result", 10)

    def load_detectors(self, ckpt_list):
        items = {glob(op.join(config, "*.py"))[0]: glob(op.join(config, "*.pth"))[0] for config in ckpt_list}
        models = [init_detector(config, pth) for config, pth in items.items()]
        return models

    def subscribe_image(self, imgmsg):
        publ_time = Time.from_msg(imgmsg.header.stamp).nanoseconds
        subs_time = self.get_clock().now().nanoseconds
        image = cv2.imdecode(np.array(imgmsg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.compression, image_name = imgmsg.encoding.split("/")
        detector = self.get_detector()
        od_result = self.get_result(detector, image)
        detc_time = self.get_clock().now().nanoseconds

        self.frame_meta[self.meta_index] = np.array([subs_time-publ_time, detc_time-subs_time,
                                                     self.model_selection, self.compression])
        self.meta_index = (self.meta_index + 1) % self.frame_meta.shape[0]

        evaluater = self.evaluate_detection(od_result, image_name, image)
        self.model_selection, self.compression = np.random.randint(0, 3), np.random.randint(50, 100)
        # self.model_selection, self.compression = self.dqn.run(self.frame_meta, )
        # self.dqn.train()
        self.publish_result()


        cv2.imshow("image", image)
        cv2.waitKey(0)
        bboxes, classes, scores = self.get_bboxes_result(detected_result.pred_instances)
        self.publish_result(bboxes, classes, scores, self.image_received_time, self.before_model, self.after_model)

    def get_detector(self, detector_index=0):
        return self.detectors[detector_index]

    def get_result(self, detector, image):
        result = inference_detector(detector, image)
        pred = result.pred_instances
        scores = pred.scores
        bboxes = pred.bboxes
        labels = pred.labels
        bboxes = bboxes[scores > 0.3].cpu().numpy()
        labels = labels[scores > 0.3].cpu().numpy()
        scores = scores[scores > 0.3].cpu().numpy()
        return {"bboxes": bboxes, "labels": labels, "scores": scores}

    def evaluate_detection(self, result, image_name, image=None):
        label_data = utils.load_label(image_name)
        iou, iou_coord = utils.compute_iou_general(label_data, result["bboxes"])
        utils.get_confusionmatrix(iou, label_data["category"], result)

    def bboxes_result_tobytes(self, instances):
        bboxes = instances["bboxes"].cpu().numpy()
        classes = instances["labels"].cpu().numpy()
        scores = instances["scores"].cpu().numpy()
        bboxes_bytes = bboxes.tobytes()     # float32
        classes_bytes = classes.tobytes()     # int64
        scores_bytes = scores.tobytes()     # float32
        return bboxes_bytes, classes_bytes, scores_bytes

    def publish_result(self, bboxes, classes, scores, received_time, before_model_time, after_model_time):
        detection_result_timestamp = Result()
        detection_result_timestamp.timestamp = [received_time, before_model_time, after_model_time]
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