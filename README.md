# Video-Based Face Recognition System

A computer vision project that implements a complete video-based face recognition pipeline, including face detection, face tracking, face alignment, supervised face identification, unsupervised face re-identification, open-set evaluation, and open-set recognition challenge models.

The system can be trained from facial video data and then used to identify known subjects or re-identify faces by cluster assignment in new video streams. It supports both image-sequence datasets and live webcam input through OpenCV.

## Features

- **Face detection with MTCNN**: Detects the largest visible face in each frame.
- **Fast face tracking**: Uses OpenCV template matching to track the face between detections and re-initializes with MTCNN when tracking confidence drops.
- **Face alignment**: Crops and normalizes detected faces to `224 x 224` pixels before feature extraction.
- **Deep face embeddings**: Uses an ONNX ResNet-50 FaceNet-style model through OpenCV DNN to extract normalized 128-dimensional face embeddings.
- **Closed-set face identification**: Implements k-nearest neighbor classification from scratch using Euclidean distances in embedding space.
- **Open-set unknown rejection**: Predicts `unknown` when the nearest-class distance is too large or the posterior probability is too low.
- **Dual-embedding recognition**: Stores and compares both color and grayscale embeddings to improve robustness against illumination and color variation.
- **Unsupervised face clustering**: Implements k-means clustering from scratch for unlabeled face re-identification.
- **Cluster diagnostics**: Tracks the k-means objective value over iterations and evaluates sensitivity to random initialization.
- **DIR curve evaluation**: Evaluates open-set face identification using false alarm rates, similarity thresholds, and detection and identification rate curves.
- **Open-set recognition challenge models**: Includes Single Pseudo Label (SPL) and Multi Pseudo Label (MPL) training strategies using scikit-learn pipelines.

## Project Workflow

The project is organized around two main runtime workflows.

### 1. Training

The training module captures frames from a video sequence or webcam, detects and tracks faces, aligns the face crop, and stores embeddings for later testing.

In **identification mode**, the system stores labeled face embeddings in a gallery for k-NN classification. In **clustering mode**, the system stores unlabeled embeddings, fits k-means, selects the best clustering result across multiple random initializations, and saves the cluster model.

### 2. Testing

The testing module loads the trained recognition or clustering model, processes incoming video frames, and overlays the result on the video stream. In identification mode, it displays the predicted identity, posterior probability, and class distance. In clustering mode, it displays the nearest cluster and the distance distribution to all cluster centers.

## Methodology

### Face Detection, Tracking, and Alignment

The `FaceDetector` class combines MTCNN detection with OpenCV template matching. The first frame or a failed tracking state is handled by MTCNN. Subsequent frames are tracked by searching a local region around the previous bounding box using `cv2.matchTemplate` with normalized correlation. The bounding box is clipped to image boundaries, and the resulting face crop is resized to `224 x 224` pixels.

### Face Embedding Extraction

The `FaceNet` class loads `resnet50_128.onnx` through OpenCV DNN. Each aligned face is converted from BGR to RGB, mean-normalized, forwarded through the network, and L2-normalized to produce a 128-dimensional embedding.

### Face Identification

The `FaceRecognizer` class stores labeled gallery samples and performs brute-force k-nearest neighbor classification. For each query face, the system extracts both color and grayscale embeddings, computes Euclidean distances against the corresponding gallery embeddings, averages both distance vectors, and predicts the majority label among the nearest neighbors.

The recognizer also estimates:

- posterior probability as `k_i / k`, where `k_i` is the number of nearest neighbors belonging to the predicted class;
- distance to the predicted class as the minimum distance among nearest neighbors from that class;
- an `unknown` decision when the distance exceeds `max_distance` or the posterior probability is below `min_prob`.

### Face Clustering and Re-Identification

The `FaceClustering` class stores unlabeled embeddings and performs k-means clustering without using scikit-learn. Cluster centers are initialized randomly from stored embeddings. The algorithm iterates between assignment and center update steps until assignments stabilize, centers converge, or the maximum iteration count is reached.

During inference, a query face is assigned to the cluster with the smallest Euclidean distance to its center. The full distance vector to all clusters is also returned for visualization.

### Open-Set Evaluation

The open-set evaluation module fits a nearest-neighbor classifier on training embeddings, predicts labels and similarities for test embeddings, and sweeps a range of false alarm rates. For each false alarm rate, it selects a similarity threshold using unknown test samples and computes the corresponding rank-1 identification rate for known samples. The DIR curve script plots this trade-off and highlights operating points for low false alarm and high identification-rate requirements.

### Challenge: SPL and MPL Open-Set Recognition

The challenge implementation provides two open-set recognition strategies:

- **SPL — Single Pseudo Label**: Treats all known-unknown samples as one additional unknown class.
- **MPL — Multi Pseudo Label**: Assigns individual pseudo-labels to known-unknown samples and learns a larger multi-class model.

Both strategies use a scikit-learn pipeline with feature scaling, L2 normalization, and an `SGDClassifier` trained with log-loss. Prediction functions return both class labels and confidence scores, with threshold-based rejection to assign the `unknown` label.

## Project Structure

```text
cvproj_exc/
├── classifier.py          # Nearest-neighbor classifier for open-set evaluation
├── config.py              # Central project paths and mode definitions
├── dir_curve.py           # DIR curve plotting and operating-point selection
├── evaluation.py          # Open-set evaluation logic
├── face_detector.py       # MTCNN detection, template tracking, and alignment
├── face_recognition.py    # FaceNet embeddings, k-NN recognition, and k-means clustering
├── osr_learning.py        # SPL and MPL challenge models
├── training.py            # Training workflow for identification and clustering
├── test.py                # Runtime testing and visualization workflow
├── test_osr_learning.py   # Unit tests for SPL/MPL challenge functions
└── requirements.txt       # Python dependencies

data/
├── resnet50_128.onnx
├── train_data/
├── test_data/
├── recognition_gallery.pkl
├── clustering_gallery.pkl
├── evaluation_train_data.pkl
├── evaluation_test_data.pkl
└── challenge_train_data.csv
```

> Large datasets, trained galleries, and the ONNX model are expected in the `data/` directory and may be excluded from the repository depending on storage or licensing constraints.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include OpenCV, MTCNN with TensorFlow support, NumPy, pandas, matplotlib, and scikit-learn.

## Data and Model Setup

Place the required data and model files under the `data/` directory:

```text
data/
├── resnet50_128.onnx
├── train_data/
├── test_data/
├── evaluation_train_data.pkl
├── evaluation_test_data.pkl
└── challenge_train_data.csv
```

The paths are managed in `config.py`. Update `Config.PROJECT_DIR` or the file paths if your local project layout differs.

## Usage

The scripts can process either image-sequence/video input or live webcam input. Use `--video none` for webcam mode.

### Train an identification gallery

```bash
python -m cvproj_exc.training \
  --mode ident \
  --video data/train_data/Alan_Ball/%04d.jpg \
  --label Alan_Ball
```

Repeat the command with different labels and training sequences to build a multi-person gallery.

### Test face identification

```bash
python -m cvproj_exc.test \
  --mode ident \
  --video data/test_data/Alan_Ball/%04d.jpg \
  --show_aligned
```

### Train a clustering model

```bash
python -m cvproj_exc.training \
  --mode cluster \
  --video data/train_data/Alan_Ball/%04d.jpg
```

### Test face re-identification by clustering

```bash
python -m cvproj_exc.test \
  --mode cluster \
  --video data/test_data/Alan_Ball/%04d.jpg
```

### Run open-set evaluation and plot the DIR curve

```bash
python -m cvproj_exc.dir_curve
```

The script prints recommended threshold operating points and plots identification rate against false alarm rate.

### Run SPL/MPL challenge tests

```bash
python -m unittest cvproj_exc.test_osr_learning
```

## Controls

During video processing:

- Press `p` to pause.
- Press `Esc` to exit.

## Technologies Used

- Python
- OpenCV
- OpenCV DNN
- MTCNN
- NumPy
- Matplotlib
- pandas
- scikit-learn
- ONNX

## Limitations and Future Improvements

- Template matching assumes relatively small motion between frames and may require re-initialization for strong pose changes, occlusion, or rapid movement.
- The k-NN gallery grows linearly with the number of stored samples, so approximate nearest-neighbor indexing could improve scalability.
- k-means clustering is sensitive to initialization; the implementation mitigates this by trying multiple seeds and selecting the lowest final objective.
- Face recognition performance depends strongly on gallery diversity, lighting, pose coverage, and the quality of aligned face crops.
- Future improvements could include landmark-based alignment, temporal smoothing of predictions, confidence calibration, and a more scalable embedding index.

