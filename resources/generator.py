import os
from PIL import Image

class ImageGenerator:
    def __init__(self, directory):
        """
        This class generates paths of images from a specified directory.
        
        Methods:
        - image_generator(): Generates images (paths) from the directory one by one.
        """
        self.directory = directory
        self.processed_paths = set()

    def image_generator(self):
        for root, _, files in os.walk(self.directory):
            for file in files:
                
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    file_path = os.path.join(root, file)
                    
                    if file_path in self.processed_paths:
                        continue
                     
                    try:
                        image = Image.open(file_path)
                        yield image
                        self.processed_paths.add(file_path) 
                        
                    except Exception as e:
                        print(f"Error opening image {file_path}: {e}")
