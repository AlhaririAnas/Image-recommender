from resources.generator import ImageGenerator
from PIL import Image
import pandas as pd
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

def save_metadata_in_database(metadata):
    """
    Saves the image metadata in the SQLite database.
    """
    conn = sqlite3.connect('image_metadata.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO metadata (filename, height, width, format, mode) 
        VALUES (:filename, :height, :width, :format, :mode)
    ''', metadata)
    conn.commit()
    conn.close()
    print(f"Metadata for {metadata['filename']} saved to database.")

def fetch_metadata(limit=5):
    """
    Fetches a limited number of metadata entries from the SQLite database and returns it as a Pandas DataFrame.
    """
    conn = sqlite3.connect('image_metadata.db')
    query = f"SELECT * FROM metadata LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    create_database()  # Ensure that the database and table exist
    
    for _ in range(10):  # We can use while True instead of a for-loop later
        try:
            image = next(img_gen)
            metadata = get_metadata(image)
            save_metadata_in_database(metadata)
        except StopIteration:
            print("No more images to process.")
            break
    
    # Fetch the first 5 metadata entries and print them as a DataFrame
    df = fetch_metadata(limit=5)
    print(df)
