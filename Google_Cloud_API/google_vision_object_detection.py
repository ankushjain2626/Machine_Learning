import io, os
from numpy import random
from google.cloud import vision
from pillow_utility import draw_borders
import pandas as pd
from PIL import Image

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"GoogleCloudPlatformKey.json"
client = vision.ImageAnnotatorClient()

file_name = 'sample_input.jpg'
image_path = os.path.join(r'.\Images', file_name)

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision.types.Image (content=content)
response = client.object_localization(image=image)
localized_object_annotations = response.localized_object_annotations

pillow_image = Image.open(image_path)
df = pd.DataFrame(columns=['name', 'score'])
for obj in localized_object_annotations:
    df = df.append(
        dict(
            name=obj.name,
            score=obj.score
        ),
        ignore_index=True)
    
    r, g, b = random.randint(1, 150), random.randint(1, 150), random.randint(150, 255)

    draw_borders(pillow_image, 
                pillow_image.size, 
                obj.bounding_poly, 
                (r, g, b),
                obj.name, 
                obj.score)

print(df)
pillow_image.show()