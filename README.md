
---

# Media Campaign Analysis Toolkit

This toolkit is designed to assist in the analysis and comparison of media files distributed across various campaigns. It includes utilities for extracting media files, generating embeddings, and computing similarities between campaigns on a day-wise basis or through average clip model embeddings.

## Features

- **Media File Extraction**: Unzips archive files containing media and stores all JPEG and PNG filenames.
- **Embedding Generation**: Creates embeddings for image files listed in a specific text file for further analysis.
- **Campaign Similarity Analysis**: Calculates the similarity between campaigns based on the average clip model embeddings of shared images.
- **Day-wise Similarity Matrix**: Computes a matrix representing day-wise similarity between two campaigns.

## Installation

Clone this repository to your local machine using:

```sh
git clone <repository-url>
```

Ensure you have Python 3.6 or later installed. Install all required dependencies by running:

```sh
pip install -r requirements.txt
```

## Usage

1. **Extract Media Files**

   To extract media files from zipped archives and generate a list of all JPEG and PNG files, run:

   ```sh
   python extract.py <path-to-zip-files>
   ```

   This will create a `filenames.txt` file containing the paths to all extracted JPEG and PNG files.

2. **Generate Embeddings**

   Before calculating similarities, you need to generate embeddings for your images:

   ```sh
   python run.py
   ```

   Make sure `filenameslist.txt` contains the paths to the images you wish to process.

3. **Campaign Similarity Analysis**

   To calculate the similarity between the average clip model embeddings of images from two campaigns, run:

   ```sh
   python campaign-smi <campaign-1-directory> <campaign-2-directory>
   ```

4. **Day-wise Similarity Matrix**

   For computing a day-wise similarity matrix between two campaigns, use:

   ```sh
   python hk-day.py <campaign-1-directory> <campaign-2-directory>
   ```
