#!/usr/bin/env python
# coding: utf-8

import os
import io
import sys
import pickle
import traceback

import cv2
import numpy as np
import pandas as pd

import glob
import filetype

from multiprocessing import Pool
#from multiprocess import Pool

from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, VGG19, InceptionV3, EfficientNetB0, EfficientNetB1, DenseNet121, ConvNeXtXLarge, ConvNeXtSmall, ConvNeXtLarge


def load_image(image_path):
  """Read and preprocess and image."""
  # If image paths don't exist, then just skip and move on.
  if not os.path.exists(image_path):
    return False

  # Read the image.
  image = cv2.imread(image_path)

  # If the image is corrupted or empty, then continue on without doing anything.
  if image is None:
    print("Unusable Image: " + str(image_path))
    return False

  # Resize the image.
  return cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB)


def build_feature_extractor(name = "resnet50", input_shape = None):
  """Constructs a feature extraction model from a pretrained base."""
  # Select the correct feature extractor.
  name = name.lower()
  if name == "resnet50":
    _model_base = ResNet50
  elif name == "vgg19":
    _model_base = VGG19
  elif name == "inceptionv3":
    _model_base = InceptionV3
  elif name == "effnet0":
    _model_base = EfficientNetB0
  elif name == "effnet1":
    _model_base = EfficientNetB1
  elif name == "densenet":
    _model_base = DenseNet121
  elif name == "convnext":
    _model_base = ConvNeXtXLarge #ConvNeXtLarge #ConvNeXtSmall #ConvNeXtXLarge
  else:
    raise ValueError(f"Received invalid feature extractor base {name}.")

  # Create the input tensor for the model.
  input_tensor = Input(shape = (256, 256, 3)) if input_shape is None else Input(shape = input_shape)

  # Load the feature extractor.
  _pretrained_transfer_model = _model_base(include_top = False, weights = 'imagenet', input_tensor = input_tensor)
  # Make the layers non-trainable.
  for layer in _pretrained_transfer_model.layers:
    layer.trainable = False

  # Perform global average pooling to reduce output layer dimensions.
  x = GlobalAveragePooling2D()(_pretrained_transfer_model.output)

  # Build the model.
  model = Model(input_tensor, x)

  # Compile the model.
  model.compile(
      loss = categorical_crossentropy,
      optimizer = Adam()
  )

  # Return the complete feature extractor base.
  return model



def extract_features(image_paths, feature_extractor, batch_size=100):
    """Conducts the actual feature extraction on images."""

    arrs = []
    for i in range(0, len(image_paths), batch_size):
        print("Batch: ", i)
        image_batch = [load_image(img_path) for img_path in image_paths[i:i+batch_size]]
        try:
            arrs.append(feature_extractor.predict(np.array(image_batch)))
        except Exception as e:
            print("ERROR on Batch:", i)
            traceback.print_exc()
            arrs.append([None] * len(image_batch))

    # Return the compiled feature vectors.
    return np.concatenate(arrs)



def generate_embeddings(user_path, feature_extractor, feature_model, base_save_path, sample_size=None):

    user = user_path.rpartition("/")[-1]
    print("User:", user)

    # Create the base path.
    base_image_path = user_path #+ "/images/"

    # Get all of the different paths.
    image_paths = []
    for file in os.listdir(base_image_path):
        if file.endswith(".mp4"):
            continue
        image_paths.append(os.path.join(base_image_path,file))

    # Check each image to make sure it really is an image
    image_paths = [img_path for img_path in image_paths if filetype.guess(img_path) is not None]

    if len(image_paths) < 2:
        return None

    print("[%s] Images: %d" % (user, len(image_paths)))
    np.random.shuffle(image_paths)

    if sample_size is None:
        sample_size = len(image_paths)

    paths_csv_file = os.path.join(base_save_path, f'{user}_{feature_model}_paths.csv')
    this_path_df = pd.DataFrame(image_paths[:sample_size], columns=["path"])
    this_path_df.to_csv(paths_csv_file)

    # Conduct the feature extraction.
    feature_vectors = extract_features(image_paths[:sample_size], feature_extractor, batch_size=100)

    # Save the features to a pickle file.
    with open(os.path.join(base_save_path, f'{user}_{feature_model}_embeddings.pickle'), 'wb') as file:
      pickle.dump(feature_vectors, file)

    return None



# Create the path to save the features to.
#feature_model_name = 'convnext'
feature_model_name = 'convnext'
base_save_path = 'user.imgs.embeddings.%s/' % feature_model_name
sample_size = None

def func(user_path):
    local_feature_extractor = build_feature_extractor(name = feature_model_name)

    generate_embeddings(
        user_path,
        local_feature_extractor,
        feature_model_name,
        base_save_path,
        sample_size
    )

if __name__ == "__main__":


    target_path = sys.argv[1]
    print("Path:", target_path)

    users = [user_path for user_path in glob.glob(target_path)]
    users = [u for u in users if "cuda" not in u]
    print(users[:10])

    collected_users = [user_path for user_path in glob.glob(base_save_path + "/*.pickle")]
    collected_users = [u.rpartition("/")[-1].partition("_")[0] for u in collected_users]
    print(collected_users[:10])

    remaining_users = [u for u in users if u.partition("_")[0] not in collected_users]
    print("Remaining:", len(remaining_users))
    print("Already:", len(collected_users))

    np.random.shuffle(remaining_users)

    with Pool(1) as p:
        p.map(func, remaining_users)