from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer

import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file("./configs/mask_rcnn_R_50_FPN_1x.yaml")
        self.cfg.MODEL.WEIGHTS = "./output/model_final.pth"

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
        instance_mode = ColorMode.IMAGE_BW)

        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)