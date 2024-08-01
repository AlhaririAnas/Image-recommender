from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
import sqlite3
import os
from resources.similarity import get_most_similar
import urllib.parse

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_images(image_paths, args, similarity_measures):
    conn = sqlite3.connect("image_metadata.db")
    color_based = []
    embedding_based = []
    yolo_based = []

    color_most_similar, embedding_most_similar, yolo_most_similar = get_most_similar(image_paths, args, similarity_measures)

    for id in color_most_similar:
        query = f"SELECT filename FROM metadata WHERE id = {id}"
        color_based.append(os.path.join(args.path, conn.execute(query).fetchall()[0][0]))

    for id in embedding_most_similar:
        query = f"SELECT filename FROM metadata WHERE id = {id}"
        embedding_based.append(os.path.join(args.path, conn.execute(query).fetchall()[0][0]))

    for id in yolo_most_similar:
        query = f"SELECT filename FROM metadata WHERE id = {id}"
        yolo_based.append(os.path.join(args.path, conn.execute(query).fetchall()[0][0]))

    conn.close()
    return color_based, embedding_based, yolo_based

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        files = request.files.getlist("file")
        filenames = []
        full_paths = []
        for file in files:
            if file.filename == "":
                return redirect(request.url)
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                filenames.append(filename)  # Store the filename for URL generation
                full_paths.append(file_path)  # Store the full path for processing
        similarity_measures = request.form.getlist("similarity_measure")
        encoded_filenames = urllib.parse.quote(",".join(filenames))
        encoded_full_paths = urllib.parse.quote(",".join(full_paths))
        encoded_measures = urllib.parse.quote(",".join(similarity_measures))
        return redirect(url_for("uploaded_files", filenames=encoded_filenames, full_paths=encoded_full_paths, similarity_measures=encoded_measures))
    return """
    <!doctype html>
    <title>Upload Images</title>
    <h1>Upload Images (up to 5)</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file multiple>
      <label><input type="checkbox" name="similarity_measure" value="color"> Color</label>
      <label><input type="checkbox" name="similarity_measure" value="embedding"> Embedding</label>
      <label><input type="checkbox" name="similarity_measure" value="yolo"> YOLO</label>
      <input type=submit value=Upload>
    </form>
    """

@app.route("/uploads/<path:filenames>/<path:full_paths>/<similarity_measures>")
def uploaded_files(filenames, full_paths, similarity_measures):
    filenames = urllib.parse.unquote(filenames).split(",")
    full_paths = urllib.parse.unquote(full_paths).split(",")
    similarity_measures = urllib.parse.unquote(similarity_measures).split(",")
    args = app.config.get("ARGS", {"path": UPLOAD_FOLDER})  # Use UPLOAD_FOLDER as default path
    
    color_based, embedding_based, yolo_based = process_images(full_paths, args, similarity_measures)
    
    results = {
        "uploaded_images": filenames,
        "color_based": color_based,
        "embedding_based": embedding_based,
        "yolo_based": yolo_based
    }

    print("Results:", results)  # Debugging statement to print results

    return render_template("display_images.html", results=results)

@app.route("/uploads/<path:filename>")
def uploaded_file_static(filename):
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        print(f"Error serving file: {filename}. Error: {e}")
        abort(404)

# Custom Jinja2 filter to get the basename
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

if __name__ == "__main__":
    app.run(debug=True)
