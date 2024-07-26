from resources.generator import ImageGenerator
import pandas as pd
import sqlite3


def create_database():
    """
    Creates a SQLite database and a table if it does not exist.
    """
    conn = sqlite3.connect("image_metadata.db")  # Connect to the SQLite database
    c = conn.cursor()
    # SQL command to create a table if it does not exist
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            height INTEGER,
            width INTEGER,
            format TEXT,
            mode TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def get_metadata(image):
    """
    Retrieves metadata of the given image.
    """
    start_index = image.filename.index("data")

    metadata_dict = {
        "filename": image.filename[start_index:].strip(),
        "height": image.height,
        "width": image.width,
        "format": image.format,
        "mode": image.mode,
    }
    return metadata_dict


def save_metadata_in_database(metadata):
    """
    Saves the image metadata in the SQLite database.
    """
    conn = sqlite3.connect("image_metadata.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO metadata (filename, height, width, format, mode) 
        VALUES (:filename, :height, :width, :format, :mode)
    """,
        metadata,
    )
    conn.commit()
    conn.close()
    # print(f"Metadata for {metadata['filename']} saved to database.")


def fetch_metadata(limit=5):
    """
    Fetches a limited number of metadata entries from the SQLite database and returns it as a Pandas DataFrame.
    """
    conn = sqlite3.connect("image_metadata.db")
    query = f"SELECT * FROM metadata LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_last_entry():
    """
    Retrieves the last entry ID from the 'metadata' table in the SQLite database.
    Returns the last entry ID as an integer.
    """
    conn = sqlite3.connect("image_metadata.db")
    query = "SELECT id FROM metadata ORDER BY id DESC LIMIT 1"
    df = pd.read_sql_query(query, conn)
    try:
        id = int(df.id.iloc[0])
    except Exception:
        id = 0
    conn.close()
    return id


if __name__ == "__main__":
    # Path to the directory containing the images
    directory_path = "D:/data/image_data"
    img_gen = ImageGenerator(directory_path).image_generator()

    create_database()  # Ensure that the database and table exist

    for _ in range(10):  # We can use while True instead of a for-loop later
        try:
            image = next(img_gen)
            print(image.filename)
            metadata = get_metadata(image)
            save_metadata_in_database(metadata)
        except StopIteration:
            print("No more images to process.")
            break

    # Fetch the first 5 metadata entries and print them as a DataFrame
    df = fetch_metadata(limit=10)
    print(df)
