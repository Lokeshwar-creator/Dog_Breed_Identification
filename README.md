# Dog Breed Identification

A Python computer-vision project that identifies a dog breed from an image. The pipeline trains (or reuses) a ResNet50V2-based classifier, evaluates it on a held-out test split, produces diagnostic plots, and predicts the breed of one chosen image.

The dataset layout follows the Kaggle **Dog Breed Identification** dataset: image filenames are IDs from `labels.csv` with a `.jpg` extension.

## What the project does

1. Reads the image IDs and breed labels from `dataset/labels.csv`.
2. Selects 60 breed classes from the available labels.
3. Loads and resizes the matching training images to 224 x 224 pixels.
4. Preprocesses the image arrays with ResNet50V2 preprocessing.
5. Encodes breed names as integer labels and makes a stratified 80/20 train-test split.
6. Applies augmentation to the training images.
7. Loads an existing trained model, or trains a new ResNet50V2 transfer-learning model.
8. Evaluates the model and shows accuracy, a classification report, class support counts, and confusion matrices.
9. Predicts the breed and confidence for the configured input image, then displays the annotated result.

## Project structure

```text
Dog Breed Identification/
|-- backend/
|   |-- main.py                  # End-to-end training, evaluation, and prediction script
|   `-- index.html               # Unwired HTML upload-page template
|-- dataset/
|   |-- labels.csv               # Image ID-to-breed labels
|   |-- train/                   # Labeled dog images used by main.py
|   |-- test/                    # Dataset test images (not used by main.py)
|   |-- sample_submission.csv    # Original dataset submission template
|   `-- dog_breed_model.h5       # Model loaded/saved by main.py
|-- images/                      # Images available for single-image prediction
|-- dog_breed_model.h5           # Additional model file; not used by main.py by default
`-- README.md
```

## Requirements

- Python 3.10 or a compatible Python version supported by your TensorFlow installation
- The dataset files in the structure shown above
- Enough RAM to hold the selected image set in memory; the script loads all selected images into a NumPy array before training
- Internet access on the first training run if TensorFlow must download ImageNet weights for ResNet50V2

Install the Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
```

If PowerShell prevents virtual-environment activation, run the commands in Command Prompt or adjust your local PowerShell execution policy.

## Dataset setup

Place the Dog Breed Identification dataset under `dataset/`:

```text
dataset/
|-- labels.csv
|-- train/
|   `-- <image-id>.jpg
|-- test/
|   `-- <image-id>.jpg
`-- sample_submission.csv
```

`main.py` uses only `labels.csv` and `dataset/train/`. Its test metrics come from an 80/20 split of the labeled training data; it does not use `dataset/test/` because those images have no labels.

## Configure the script

Open `backend/main.py` and verify these values near the top of the file:

```python
labels_path = r"...\dataset\labels.csv"
train_file = r"...\dataset\train"
model_path = r"...\dataset\dog_breed_model.h5"
pred_img_path = r"...\images\2.jpg"
```

The current script uses absolute Windows paths. If you move or clone this project, update those paths before running it. Set `pred_img_path` to any supported dog image you want to classify.

The primary training settings are also defined at the top of the script:

| Setting | Current value | Purpose |
| --- | ---: | --- |
| `num_breeds` | 60 | Number of selected classes |
| `im_size` | 224 | Width and height of every model input |
| `batch_size` | 64 | Images per training/evaluation batch |
| `epochs` | 20 | Training passes over the augmented data |
| `learning_rate` | 0.001 | RMSprop learning rate |

## Run the full workflow

From the project root, run:

```powershell
python backend/main.py
```

### First run: model training

If `dataset/dog_breed_model.h5` does not exist, the script:

1. Chooses the 60 classes by sorting breed-frequency labels in descending order, then taking alternating entries from the first 121 positions. This is the exact current selection logic, not simply the 60 most frequent breeds.
2. Builds a ResNet50V2 backbone with ImageNet weights and freezes its pretrained layers.
3. Adds batch normalization, global average pooling, two 50% dropout layers, a 1,024-unit ReLU layer, and a softmax classifier with one output per selected breed.
4. Trains with RMSprop and sparse categorical cross-entropy.
5. Saves the trained model to `dataset/dog_breed_model.h5`.

Training augmentation includes rotation, horizontal shifting, vertical shifting, shear, zoom, and horizontal flips.

### Later runs: loading and evaluation

When `dataset/dog_breed_model.h5` already exists, the script loads it instead of training again. It still reloads and preprocesses the dataset, re-creates the deterministic split (`random_state=42`), runs evaluation, generates plots, and predicts the configured single image.

To train from scratch, rename or move `dataset/dog_breed_model.h5`, then run the command again. Keep a backup if you want to preserve the current trained model.

## Outputs

During every run, the console reports:

- number of unique labels and selected breed classes;
- any training images that could not be read;
- test accuracy and a full per-class classification report;
- per-class test-set sample counts;
- a compact precision/recall/F1 table; and
- the predicted breed and confidence for `pred_img_path`.

Matplotlib opens these figures:

- confusion matrix of prediction counts;
- normalized confusion matrix (per-class recall);
- bar chart of per-class test support;
- training/validation accuracy and loss curves, only when a new model was trained; and
- the selected prediction image with a green border and prediction label.

## Web template status

`backend/index.html` contains an upload form intended for a `/predict` endpoint, but this repository currently has no Flask, FastAPI, Django, or other server implementation. Opening the HTML alone will not make predictions, and `main.py` is the supported execution path. A backend route must be implemented before the form can process uploads.

## Windows console note

The script uses ASCII status labels such as `[OK]` and `[INFO]` so it runs in legacy Windows consoles that use the cp1252 encoding. This avoids `UnicodeEncodeError` caused by unsupported emoji characters.

## Limitations and next steps

- The model is trained on only the selected 60 classes, so it cannot identify breeds outside that set.
- The pipeline holds all selected images in memory; a streaming `tf.data` pipeline would scale better.
- The test split is rebuilt at runtime rather than saved as an artifact.
- The model file is HDF5 (`.h5`); TensorFlow’s newer `.keras` format is preferable for future work.
- The current green box outlines the entire image; it is a display annotation, not object detection or dog localization.
