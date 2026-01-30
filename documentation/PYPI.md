# Vision Loom

Library to create labeling data for computer vision applications, leveraging VLM's. 

## Save detection results  
* pip install vision-loom

```python code
from vision_loom.models.grounding_dino import GroundingDINO 
from vision_loom.utils.io import load_images 
```

* Enter a single image or path to a folder containing images
```python
images = "vision_loom/test_notebooks/test_images"
```
