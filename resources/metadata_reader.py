from resources.generator import ImageGenerator
from PIL import Image
import sqlite3

# Path to the directory containing the images
directory_path = "D:\\BigData-Data\\data\\image_data"
img_gen = ImageGenerator(directory_path).image_generator()

def create_database():
    """
    Creates a SQLite database and a table if it does not exist.
    """
    conn = sqlite3.connect('image_metadata.db')  # Connect to the SQLite database
    c = conn.cursor()
    # SQL command to create a table if it does not exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            height INTEGER,
            width INTEGER,
            format TEXT,
            mode TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_metadata(image):
    """
    Retrieves metadata of the given image.
    """
    start_index = image.filename.index('data')
    
    metadata_dict = {
        "filename": image.filename[start_index:].strip(),
        "height": image.height,
        "width": image.width,
        "format": image.format,
        "mode": image.mode
    }
    return metadata_dict
