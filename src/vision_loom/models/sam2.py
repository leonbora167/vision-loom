from transformers import Sam2Processor, Sam2Model
import torch
from PIL import Image
import numpy as np 
# import matplotlib.pyplot as plt
import cv2
from vision_loom.models.grounding_dino import GroundingDINOTiny
import logging
from vision_loom.utils.io import mask_to_polygon, mask_to_contour_format
import os
from pathlib import Path 
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class SAM2:
    def __init__(self, model_id="facebook/sam2.1-hiera-tiny", device_map="auto", cache_dir = "./cache"):
        self.model_id = model_id
        self.device_map = device_map 
        self.cache_dir = cache_dir

        model = Sam2Model.from_pretrained(self.model_id,
                                  device_map = self.device_map,
                                  cache_dir=self.cache_dir)
        self.model = model.to(model.device)
        logging.info("SAM2 Model Loaded")
        
        self.processor = Sam2Processor.from_pretrained(self.model_id,
                                          use_fast=True,
                                          cache_dir = self.cache_dir)
        logging.info("SAM2 Processor Loaded")

    def detect(self, image, text_labels, image_path, save_results=False, results_folder="detector_results"):
        detector_model = GroundingDINOTiny()
        detector_results = detector_model.detect(image, text_labels, image_path)
        if detector_results:
            bboxes = detector_results[0]["boxes"].cpu().numpy()
            conf_scores = detector_results[0]["scores"].cpu().numpy()
            text_labels = detector_results[0]["text_labels"]
            labels = detector_results[0]["labels"] #Should return integer labels in the future as per the warnings currently 
            for index, box in tqdm(enumerate(bboxes)):
                x1, y1, x2, y2 = map(int, box)
                confidence_score = conf_scores[index]
                text_class = text_labels[index]
                input_boxes = [[[x1, y1, x2, y2]]]
                inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0] #masks of one object of one class currently
                poly_lines = mask_to_contour_format(masks, image)
                if save_results:
                    self.save_results(poly_lines, text_class, confidence_score, results_folder, image_path)
        else:
            logging.info("No object found from prompt labels")
            return None 
        
    def save_results(self, poly_lines, text_class, confidence_score, results_folder, image_path):
            os.makedirs(results_folder, exist_ok=True)
            file_name = Path(image_path).name.split(".")[0]
            label_path = os.path.join(results_folder, file_name+".txt")
            results = str(text_class) + "\t" + poly_lines + "\t" + str(confidence_score)

            with open(label_path, "a") as f:
                f.write(results + "\n")