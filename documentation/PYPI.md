# Vision Loom

Leverage Vision Language Models to label images and create training data for other computer vision tasks. 

Vision Loom is an automated labeling framework designed to bridge the gap between high-reasoning Foundation Models and real-time edge deployment. Leverage the "zero-shot" capabilities of Vision-Language Models (VLMs) to generate high-fidelity training data without the manual overhead.

| Model | Task | Reference
| :--- | :---: | ---: |
Grounding Dino Tiny | Object Detection | detector_test.py
SAM-2 | Object Segmentation | segmentation_test.py

## Save detection results
  
* Install the library. Install the latest version for the updated features and comments.

```python
pip install vision-loom
```

```python 
from vision_loom.models.grounding_dino import GroundingDINO 
from vision_loom.utils.io import load_images 
```

* Enter a single image or path to a folder containing images
```python
images = "vision_loom/test_notebooks/test_images"
dataloader = load_images(images)
```

* Once the data loader is created, type in the items you want to detect and create the bounding boxes for. If your set of images has multiple objects over multiple images, I recommend giving the prompt for all the objects together to save compute.
```python
prompt = [["car", "person", "book", "bike"]]
``` 

* Run the pipeline.
```python
for img_path, image in tqdm(dataloader):
    results = model.detect(image, prompt, img_path, save_results=True)
``` 
![Detector Pipeline V1](https://raw.githubusercontent.com/leonbora167/vision-loom/documentation/Detector_Pipeline_V1.jpg)

## 📦 Releases
See [CHANGELOG](./CHANGELOG.md) for version history.

## Roadmap 

- [ ] Add Image Classification Support
- [ ] Options for export to classical SOTA formats like YOLO, COCO etc. 