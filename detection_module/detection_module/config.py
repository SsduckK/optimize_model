SCORE_THRESHOLD = 0.3
BATCH_SIZE = 128
MODEL_NUM = 3
COMPRESSION_NUM = 3
TARGET_TIME = 0.03
MEAN_RANGE = 100
EPISODE_UNIT = 300
NUM_EPISODES = 50

RESULT_PATH = "/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/detection_module/data"
VALIDATING_DATA = ("/home/gorilla/lee_ws/ros/src/optimize_model/"
                   "detection_module/detection_module/data/validation_data/validate_data1.json")
PARAMETER = {
    "model": {"mean": 0.9658869395711501, "std": 0.7872064525353218},
    "compression": {"mean": 0.9756335282651072, "std": 0.8238589120862889},
    "c2s_time": {"mean": 65772099.15789474, "std": 24930302.24839799},
    "det_time": {"mean": 0.019964976617467284, "std": 0.012859863498887488}
    }
