
---

# Media Campaign Analysis Toolkit

This toolkit is designed to assist in the analysis and comparison of media files distributed across various campaigns. It includes utilities for extracting media files, generating embeddings, and computing similarities between campaigns on a day-wise basis or through average clip/blip/convnext model embeddings.

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

Ensure you have Python 3.8 or later installed.

## Usage

1. **Extract Media Files**

   To extract media files from zipped archives and generate a list of all JPEG and PNG files, run:

   ```sh
   python extract.py
   ```

   This will create a `filenames.txt` file containing the paths to all extracted JPEG and PNG files.

2. **Generate Embeddings**

   Before calculating similarities, you need to generate embeddings for your images:

   ```sh
   python run_clip.py
   ```

   or

   ```sh
   python run_blip2.py
   ```

   or

   ```sh
   python run_convnext.py
   ```

   Make sure `filenameslist.txt` contains the paths to the images you wish to process.

3. **KMeans on campaigns**

    Creating Kmeans clusters for a given campaign

    ```sh
        python taiwan_kmeans.py
    ```


4. **Campaign Similarity Analysis**

   To calculate the similarity between the average model embeddings of images from two campaigns, run:

   ```sh
   python campaign-sim.py
   ```

4. **Day-wise Similarity Matrix**

   For computing a day-wise similarity matrix between two campaigns, use:

   ```sh
   python hk-day.py
   ```
