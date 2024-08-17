# Image Recommender

## Overview

This repository houses a project that finds the top five images similar to any given photo from a large dataset of almost 500,000 images. The software utilizes Python to recommend images based on various similarity metrics such as color schemes, embeddings, and YOLO object detection.

## Getting Started

### Prerequisites

Ensure Python is installed on your system and you have the necessary permissions to execute scripts.

### Installation

Clone the repository to your local machine:

```bash
git clone [URL to Image Recommender repository]
```

Navigate to the project directory:

```bash
cd "path_to/Image-recommender"
```

### Launching the Web Interface

To start the web-based interface, use the following command:
```bash
python -m resources.main -p "../Parent_folder_of_dataset"
```

After running the command, the system will open a browser window automatically. If it doesn’t, manually click the IP address shown in the terminal output using Ctrl + click.

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
