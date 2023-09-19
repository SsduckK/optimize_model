import sys

import pandas as pd
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

from .detect_module import Detectors
from rod_msg.msg import Rodmsg
from rclpy.time import Time
from .label_reader import LabelReader
from .evaluator import Evaluator
from .DQN import DQN
from .DQN_memory import ReplayMemory
from .utils import draw_bboxes


class Server(Node):
    def __init__(self, gt_path, ckpt_list, reserve_frames=5):
        super().__init__("server")
        self.detector = Detectors(ckpt_list)
        self.memory = ReplayMemory()
        self.dqn = DQN(self.memory)
        self.read_labels = LabelReader(gt_path)
        self.evaluate = Evaluator()
        self.frame_index = 0
        self.limit_time = 0.05
        self.metric_by_model = {}   # {'{model_name}_{compression}': [TP, FP, FN]}
        self.image_subscriber = self.create_subscription(Image, "sending_image", self.subscribe_image, 10)
        self.result_publisher = self.create_publisher(Rodmsg, "sending_result", 10)

    def subscribe_image(self, imgmsg):
        publ_time = Time.from_msg(imgmsg.header.stamp).nanoseconds
        subs_time = self.get_clock().now().nanoseconds

        image, image_name, model_selection, compression = self.det_input_process(imgmsg)    # class -> dqn_input["model"], dqn_input["compression"]
        det_result, detection_time = self.detector.run(self.detector.models[model_selection], image) # class

        gt_label = self.read_labels(image_name)
        draw_bboxes(image, det_result, gt_label, self.frame_index, 0)
        eval_res = self.evaluate(det_result, gt_label, model_selection, compression)    # [recall, precision]
        reward = self.calculate_reward(eval_res, detection_time, self.limit_time)
        cur_state = {"model": model_selection, "compression": compression,
                     "C2S_time": subs_time-publ_time, "det_time": detection_time, "reward": reward}
        self.memory.append_data(self.frame_index, cur_state)
        # action, reward
        next_model_selection, next_compression = self.dqn.select_action(self.memory.latest_state())
        self.dqn.optimize_model(self.memory)
        self.dqn.update_model()
        self.frame_index += 1
        bboxes, classes, scores = self.bboxes_result_tobytes(det_result)
        self.publish_result(bboxes, classes, scores, next_compression, next_model_selection)

    def det_input_process(self, image_msg):
        image = cv2.imdecode(np.array(image_msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        model_selection, compression, image_name = image_msg.encoding.split("-")
        image_name = image_name.split("/")[-1]
        model_selection = int(model_selection)
        compression = (int(compression) / 30) - 1
        return image, image_name, model_selection, compression

    def calculate_reward(self, evaluation_result, detection_time, limit_time):
        recall, precision = evaluation_result
        F1_score = 2 * recall * precision / (recall + precision)
        reward = F1_score + 1 - detection_time/limit_time
        return reward   # 0 ~ 1

    def bboxes_result_tobytes(self, instances):
        bboxes = instances["bboxes"]
        classes = instances["category"]
        scores = instances["scores"]
        bboxes_bytes = bboxes.tobytes()     # float32
        classes_bytes = classes.tobytes()     # int64
        scores_bytes = scores.tobytes()     # float32
        return bboxes_bytes, classes_bytes, scores_bytes

    def publish_result(self, bboxes, classes, scores, compression, model_num):
        detection_result_timestamp = Rodmsg()
        detection_result_timestamp.bboxes = list(bboxes)
        detection_result_timestamp.classes = list(classes)
        detection_result_timestamp.scores = list(scores)
        detection_result_timestamp.compression = int(compression)
        detection_result_timestamp.model_number = int(model_num)
        self.result_publisher.publish(detection_result_timestamp)

def system_init():
    package_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(package_path)


def main(args=None):
    system_init()
    rclpy.init(args=args)
    gt_path = "/mnt/intHDD/cityscapes/annotations_json/"
    ckpt_list = [t for t in glob(op.join("/mnt/intHDD/mmdet_ckpt/yolox", "*")) if "_base_" not in t]
    node = Server(gt_path, ckpt_list)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SigInt)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()