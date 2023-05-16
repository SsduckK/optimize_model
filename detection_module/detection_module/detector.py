import sys

# sys.path.append("/home/gorilla/.pyenv/versions/model_opt/lib/python3.8/site-packages")
import rclpy
from rclpy.node import Node
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


class Detector(Node):
    def __init__(self, ckpt_list):
        super().__init__("observer")
        self.detector = MD(ckpt_list)
        self.camera_sending_time = 0
        self.before_model = 0
        self.after_model = 0
        self.image_subscriber = self.create_subscription(Imagetime, "sending_image", self.subscribe_image, 10)

    def subscribe_image(self, image_time):
        image = image_time.image
        self.camera_sending_time = image_time.timestamp
        msg_img = CvBridge().imgmsg_to_cv2(image, "bgr8")
        self.before_model = time.time()
        detected_result = self.detector(msg_img)
        self.after_model = time.time()
        bboxes, classes, scores = self.get_bboxes_result(detected_result.pred_instances)
        cv2.imshow("sub_image", msg_img)
        cv2.waitKey(2)

    def get_bboxes_result(self, instances):
        bboxes = instances["bboxes"].cpu().numpy()
        labels = instances["labels"].cpu().numpy()
        scores = instances["scores"].cpu().numpy()
        bboxes_bytes = bboxes.tobytes()     # float32
        labels_bytes = labels.tobytes()     # int64
        scores_bytes = scores.tobytes()     # float32
        return bboxes_bytes, labels_bytes, scores_bytes

    def publsih_result(self, bboxes, classes, scores):
        pass


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
    ckpt_list = glob(op.join("/mnt/intHDD/mmdet_ckpt/test_yolo", "*"))
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