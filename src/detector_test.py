from vision_loom.models.grounding_dino import GroundingDINO 
from vision_loom.utils.io import load_images 
from tqdm import tqdm 

model = GroundingDINO() 

#images = "vision_loom\\test_notebooks\\car.jpeg" #Single File
images = "vision_loom/test_notebooks/test_images" # Folder

dataloader = load_images(images)
prompt = [["car", "person", "book", "bike"]]

for img_path, image in tqdm(dataloader):
    results = model.detect(image, prompt, img_path)
    