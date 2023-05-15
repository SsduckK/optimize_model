import sys

sys.path.append("/home/gorilla/.pyenv/versions/model_opt/lib/python3.8/site-packages")
# sys.path.append("/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/mmdetection")
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


class Detector(Node):
    def __init__(self, ckpt_list):
        super().__init__("observer")
        self.detector = MD(ckpt_list)
        self.sending_time = 0
        self.before_model = 0
        self.after_model = 0
        self.image_subscriber = self.create_subscription(Image, "sending_image", self.subscribe_image, 10)
        self.timer_subscriber = self.create_subscription(Float64, "sending_moment", self.subscribe_send_time, 10)

    def subscribe_image(self, image):
        msg_img = CvBridge().imgmsg_to_cv2(image, "bgr8")
        self.before_model = time.time()
        detected_result = self.detector(msg_img)
        self.after_model = time.time()
        cv2.imshow("sub_image", msg_img)
        cv2.waitKey(2)

    def subscribe_sen_time(self, timer):
        sub_timer = Float64()
        sub_timer.data = timer.data
        print(sub_timer)


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