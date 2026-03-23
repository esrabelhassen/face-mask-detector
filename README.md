# Face Mask Detector 

A real-time face mask detection system built with YOLOv8 — trained from scratch on a custom dataset and deployed on a live webcam feed.

---

## What it does

Detects whether a person is wearing a face mask or not in real time using a webcam. Draws a bounding box around each detected face with a label and confidence score.

- 🟢 **Green box** → mask detected
- 🔴 **Red box** → no mask detected

---

## Results

| Metric | Score |
|---|---|
| mAP50 (overall) | **89.3%** |
| Precision | **93.6%** |
| Recall | **80.9%** |
| with_mask mAP | **96.4%** |
| without_mask mAP | **82.2%** |
| Training time | **16 minutes** |
| Model size | **6.2 MB** |

Trained for 50 epochs with early stopping — best weights saved at epoch 36.

---

## Project Workflow

| Phase | Steps |
|---|---|
| **Data** | Kaggle dataset → visualize → convert XML to YOLO format → 80/10/10 split |
| **Training** | YOLOv8n pretrained → fine-tune on Google Colab T4 GPU → evaluate |
| **Deployment** | Export best.pt → run live inference with OpenCV on webcam |

---

## Tech Stack

- **Model** — YOLOv8n (Ultralytics)
- **Framework** — PyTorch
- **Training** — Google Colab (free T4 GPU)
- **Inference** — OpenCV + OpenVINO
- **Language** — Python 
- **Dataset** — [Face Mask Detection — Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/esrabelhassen/face-mask-detector.git
cd face-mask-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run live detection

```bash
python detect.py
```

> Make sure `best.pt` is in the same folder as `detect.py`. Press **Q** to quit the webcam window.

---

## Repository Structure

```
face-mask-detector/
├── README.md
├── detect.py                  # live webcam inference script
├── requirements.txt           # dependencies
├── full_project_workflow.html # visual project overview
├── notebook/
│   └── face_mask_detector.ipynb  
```

---

## Training from Scratch

Open `notebook/face_mask_detector.ipynb` in Google Colab and run all cells. The notebook covers:

1. Dataset download from Kaggle
2. Data exploration and visualization
3. XML → YOLO label conversion
4. Train/val/test split
5. YOLOv8n fine-tuning
6. Evaluation on test set
7. Model export


## Author

**Esra Belhassen**
*AI builder — always learning, always shipping.*

---

> Built end-to-end in a single session. From raw dataset to live webcam detection.
