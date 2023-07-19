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
from .label_reader import LabelReader
from .evaluator import Evaluator


class Detector(Node):
    def __init__(self, gt_path, ckpt_list, reserve_frames=5):
        super().__init__("server")
        self.detectors = self.load_detectors(ckpt_list)
        self.read_labels = LabelReader(gt_path)
        self.evaluate = Evaluator()
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
        det_input = self.det_input_process()    # func
        dqn_input = self.dqn_input.get()    # class
        dqn_result = self.dqn.run()           # class
        det_result = self.detector.run(det_input)    # class
        dqn_train_input = self.dqn_input.update(det_result)  # class
        self.dqn.train(dqn_train_input)

        publ_time = Time.from_msg(imgmsg.header.stamp).nanoseconds
        subs_time = self.get_clock().now().nanoseconds
        image = cv2.imdecode(np.array(imgmsg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.compression, image_name = imgmsg.encoding.split("/")
        detector = self.get_detector()
        od_output = self.get_result(detector, image)
        detc_time = self.get_clock().now().nanoseconds

        self.frame_meta[self.meta_index] = np.array([subs_time-publ_time, detc_time-subs_time,
                                                     self.model_selection, self.compression])
        self.meta_index = (self.meta_index + 1) % self.frame_meta.shape[0]
        gt_label = self.read_labels(image_name)
        eval_res = self.evaluate(od_output, gt_label, "model_name")

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
        labels = np.expand_dims(labels[scores > 0.3].cpu().numpy(), axis=-1)
        scores = np.expand_dims(scores[scores > 0.3].cpu().numpy(), axis=-1)
        return {"bboxes": bboxes, "category": labels, "scores": scores}

    def bboxes_result_tobytes(self, instances):
        bboxes = instances["bboxes"].cpu().numpy()
        classes = instances["category"].cpu().numpy()
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
    package_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(package_path)


def main(args=None):
    system_init()
    rclpy.init(args=args)
    gt_path = "/mnt/intHDD/kitti/training/label_2"
    ckpt_list = [t for t in glob(op.join("/mnt/intHDD/mmdet_ckpt/yolov7", "*")) if "_base_" not in t]
    node = Detector(gt_path, ckpt_list)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SigInt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()