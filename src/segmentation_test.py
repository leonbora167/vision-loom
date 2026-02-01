from vision_loom.models.sam2 import SAM2
from vision_loom.utils.io import load_images
from PIL import Image 
from tqdm import tqdm

model = SAM2() 

#images = "vision_loom\\test_notebooks\\car.jpeg" #Single File
images = "vision_loom/test_notebooks/test_images" # Folder

dataloader = load_images(images)
prompt = [["car", "person", "book"]]

for img_path, image in tqdm(dataloader):
    results = model.detect(image, prompt, img_path, save_results=True)
