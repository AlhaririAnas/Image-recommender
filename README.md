# Image Recommender

## Overview

This repository houses a project that finds the top five images similar to any given photo from a large dataset of almost 500,000 images. The software utilizes Python to recommend images based on various similarity metrics such as color schemes, embeddings, and YOLO object detection.

## Getting Started

### Prerequisites

Ensure Python is installed on your system and you have the necessary permissions to execute scripts.

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/AlhaririAnas/Image-recommender.git
```

Navigate to the project directory:

```bash
cd Image-recommender
```

Install dependencies:

```bash
pip install .
```

## Setting up your dataset

To set up a dataset with all similarities and metadata, use:
```bash
python -m resources.main -m -s -p "../Parent_folder_of_dataset"
```

Use this commands to specify:

|  Command 	|  Explanation 	|   Tip	|
|---	|---	|---	|
|  -m 	|  set up the image_metadata.db 	|   	|
|  -s	|  set up a pickle file with all similarity information 	|   	|
|  -p	|   path to your '/data'	directory with your images|  Default: 'D:/'	|
|  -d	|  device to use: cpu or cuda 	|  If not specified, cuda will be used if available 	|
|  --pkl_file 	|  path to a pickle file, in which the similarity information are stored 	|  Default: 'similarities.pkl' 	|
|  --checkpoint	|  Number of images after which the --pkl_file will be updated | Default: 100  	|

If the program crashes, you can simply restart it and it will continue where it left off.

If neither the `-m` nor the `-s` flag are specified, it will launch the web interface...

## Launching the Web Interface

To start the web-based interface, use the following command:
```bash
python -m resources.main -p "../Parent_folder_of_dataset"
```

After running the command, the system will open a browser window automatically. If it doesn’t, manually click the IP address shown in the terminal output using Ctrl + click.
This process may take some time, especially if you are performing it for the first time. This is because the pickle file is loaded and the clusters of the color histograms and the embeddings are calculated. To change the clustering methods, please refer to `resources.main.create_and_save_clustering_model`.

## Recommending Images
To recommend images similar to one or more input images:
![Web interface](https://github.com/AlhaririAnas/Image-recommender/blob/readme/Web%20Interface.png)

Click on "Choose files" and select the input images.
Choose the type of similarity comparison by checking the appropriate boxes:

* Color
* Embedding
* YOLO

Select a distance measure from the dropdown menu (Euclidean, Manhattan, Cosine).

Click "Upload" to start the image similarity search. The process may take a few seconds, and the results will be displayed based on the selected metrics.

## Contact

For any inquiries, please contact:

* Jonah Gräfe: jonah.graefe@study.hs-duesseldorf.de
* Anas Alhariri: anas.alhariri@study.hs-duesseldorf.de
