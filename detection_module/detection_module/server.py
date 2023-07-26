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
import torch
import time

from .detect_module import Detectors
from rod_msg.msg import Rodmsg
from detecting_result_msg.msg import Result
from rclpy.time import Time
from .label_reader import LabelReader
from .evaluator import Evaluator
from .DQN import DQN
from .DQN_preparer import DqnInput
from .utils import draw_bboxes


class Server(Node):
    def __init__(self, gt_path, ckpt_list, reserve_frames=5):
        super().__init__("server")
        self.detector = Detectors(ckpt_list)
        self.dqn = DQN()
        self.dqn_input = DqnInput()
        self.read_labels = LabelReader(gt_path)
        self.evaluate = Evaluator()
        self.model_selection = 0
        self.limit_time = 0.05
        self.compression = 0
        self.metric_by_model = {}   # {'{model_name}_{compression}': [TP, FP, FN]}
        self.image_subscriber = self.create_subscription(Image, "sending_image", self.subscribe_image, 10)
        self.result_publisher = self.create_publisher(Result, "sending_result", 10)

    def subscribe_image(self, imgmsg):
        publ_time = Time.from_msg(imgmsg.header.stamp).nanoseconds
        subs_time = self.get_clock().now().nanoseconds
        image, image_name = self.det_input_process(imgmsg)    # func
        dqn_input = self.dqn_input.get(recent_frame=3)      # class -> dqn_input["model"], dqn_input["compression"]
        det_result, detection_time = self.detector.run(self.detector.models[self.model_selection], image) # class
        gt_label = self.read_labels(image_name)
        eval_res = self.evaluate(det_result, gt_label, self.model_selection)    # [recall, precision]
        reward = self.calculate_reward(eval_res, detection_time, self.limit_time)
        self.dqn_input.update(self.model_selection, self.compression, subs_time-publ_time, detection_time)
        # draw_bboxes(image, det_result["bboxes"])
        # dqn_result = self.dqn.run()           # class
        # dqn_train_input = self.dqn_input.update(det_result)  # class
        # self.dqn.train(dqn_train_input)

        # self.model_selection, self.compression = self.dqn.run(self.frame_meta, )
        # self.dqn.train()
        # self.publish_result()

        # cv2.imshow("image", image)
        # cv2.waitKey(0)

    def det_input_process(self, image_msg):
        image = cv2.imdecode(np.array(image_msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.compression, image_name = image_msg.encoding.split("/")
        return image, image_name

    def calculate_reward(self, evaluation_result, detection_time, limit_time):
        recall, precision = evaluation_result
        F1_score = 2 * recall * precision / (recall + precision)
        reward = F1_score + 1 - (detection_time - limit_time)/limit_time
        return reward

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
    node = Server(gt_path, ckpt_list)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SigInt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()