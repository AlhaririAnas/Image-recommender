import os
from PIL import Image


class ImageGenerator:
    def __init__(self, directory):
        """
        This class generates paths of images from a specified directory.

        Methods:
        - image_generator(starting_path=None): Generates images (paths) from the directory one by one, starting from the specified path.
        """
        self.directory = directory

    def image_generator(self, starting_path=None):
        start_yielding = starting_path is None
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.lower().endswith(("png", "jpg", "jpeg")):
                    file_path = os.path.join(root, file)

                    if start_yielding:
                        try:
                            image = Image.open(file_path)
                            yield image
                        except Exception as e:
                            print(f"Error opening image {file_path}: {e}")

                    if starting_path and file_path == starting_path:
                        start_yielding = True
