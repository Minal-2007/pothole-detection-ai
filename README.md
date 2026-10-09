# 🛣️ Pothole Detection AI Using YOLOv11

An AI-powered computer vision project that detects potholes in road images using the YOLOv11 object detection model. The project explores how deep learning can support road condition monitoring and help identify damaged road surfaces.

## 📌 Project Overview

Potholes are a common road safety concern that can damage vehicles, affect traffic flow, and increase accident risks. Manual road inspection can be time-consuming, making automated road damage detection a useful application of computer vision.

This project uses **YOLOv11** to identify potholes in images by detecting their locations with bounding boxes. The goal is to demonstrate how object detection can support automated road inspection and infrastructure monitoring.

## 🎯 Objectives

* Detect potholes in road images using deep learning.
* Apply YOLOv11 for object detection.
* Visualize detected potholes with bounding boxes and confidence scores.
* Explore AI-based road condition monitoring.
* Build a foundation for future real-time pothole detection systems.

## ⚙️ Technologies Used

* **Python** — core programming
* **YOLOv11** — object detection
* **Ultralytics** — model training and inference tools
* **OpenCV** — image processing, if used
* **NumPy** — numerical operations
* **Streamlit** — interactive dashboard, if included
* **Google Colab / Jupyter Notebook** — model development, if used

## ✨ Key Features

* **Pothole detection:** Identify potholes in road images.
* **Bounding box visualization:** Mark detected potholes in the input image.
* **Confidence scores:** Display the model's confidence for each detection, where supported.
* **Image-based inference:** Analyze uploaded or selected road images.
* **Road safety application:** Demonstrate the potential of computer vision for road inspection.

*Features should match the functionality implemented in your project.*

## 🧠 How It Works

1. **Image input** — provide a road image to the model.
2. **Preprocessing** — resize and prepare the image for inference.
3. **Object detection** — YOLOv11 processes the image and predicts pothole locations.
4. **Postprocessing** — filter predictions using confidence and detection thresholds.
5. **Visualization** — draw bounding boxes around detected potholes.
6. **Output** — display the annotated image and detection results.

## 🏗️ System Architecture

```text
       Road Image
            ↓
    Image Preprocessing
            ↓
     YOLOv11 Model
            ↓
    Pothole Detection
            ↓
 Bounding Boxes & Scores
            ↓
   Annotated Image Output
```

## 📂 Project Structure

```text
Pothole-Detection-AI/
├── app.py                 # Application or dashboard (if included)
├── requirements.txt       # Project dependencies
├── best.pt                # Trained model weights (if included)
├── README.md
└── dataset/               # Training and validation data (if included)
```

*Adjust this structure to reflect your actual repository. Do not commit large datasets or model files unless appropriate; Git LFS or a release asset may be useful for large weights.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you are using the Ultralytics package and have not listed it in `requirements.txt`, install it with:

```bash
pip install ultralytics
```

### 4. Run inference

For a trained model named `best.pt`, run:

```bash
yolo predict model=best.pt source="path/to/road_image.jpg"
```

Replace the model path and image path with your actual file locations.

## 📊 Model Evaluation

Evaluate the model using metrics such as:

* **Precision:** How many predicted potholes were correct.
* **Recall:** How many actual potholes were detected.
* **mAP@0.5:** Detection performance at an intersection-over-union threshold of 0.5.
* **mAP@0.5:0.95:** Detection performance averaged across multiple IoU thresholds.

Add your actual validation results here after evaluating the trained model.

## 🌍 Applications

* Automated road inspection
* Smart city infrastructure monitoring
* Road maintenance planning
* Computer vision research
* Potential integration with vehicle-mounted cameras or mobile road survey systems

🌐 Live Demo

Streamlit Dashboard: pothole-detection-ai ∙ main ∙ app.py



## 🔭 Future Improvements

* Extend detection to video streams and real-time camera feeds.
* Integrate GPS tagging to record pothole locations.
* Develop a mobile or vehicle-mounted road inspection system.
* Classify road damage by severity.
* Optimize inference for edge devices.
* Integrate detections with a road maintenance reporting dashboard.

## ⚠️ Limitations

Detection accuracy depends on image quality, lighting, viewing angle, dataset diversity, and model training. The system may miss potholes or produce false detections. Results should be validated before using the system for operational road maintenance decisions.

## 👩‍💻 Author

**Minal Gunasekaran**
Mechatronics Engineering | AI/ML | Computer Vision

* GitHub: [Minal-2007](https://github.com/Minal-2007)

## 📄 License

Add a license file if you intend to distribute the project for reuse.
