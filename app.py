import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pothole Detection AI",
    page_icon="🕳️",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = Path(__file__).parent / "best.onnx"

CLASS_NAMES = [
    "Pothole",
    "Crack",
    "Normal Road"
]

INPUT_SIZE = 640


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "best.onnx was not found in the project folder."
    )

    st.stop()


# ============================================================
# LOAD YOLO ONNX MODEL USING OPENCV
# ============================================================

@st.cache_resource
def load_model():

    return cv2.dnn.readNetFromONNX(
        str(MODEL_PATH)
    )


try:

    net = load_model()

except Exception as e:

    st.error(
        "The YOLO ONNX model could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image, size=INPUT_SIZE):

    # PIL image → RGB
    image = image.convert("RGB")

    original_width, original_height = image.size

    # Convert PIL → NumPy
    image_array = np.array(image)

    # Resize
    resized = cv2.resize(
        image_array,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )

    # IMPORTANT:
    # The exported YOLO model expects RGB.
    #
    # Therefore:
    # RGB → RGB
    #
    # We do NOT convert to BGR here.

    blob = cv2.dnn.blobFromImage(
        resized,
        scalefactor=1.0 / 255.0,
        size=(size, size),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False
    )

    return (
        blob,
        original_width,
        original_height
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def detect(image, confidence_threshold):

    blob, original_width, original_height = preprocess(
        image
    )

    # Send image into network
    net.setInput(blob)

    # Run inference
    outputs = net.forward()

    # --------------------------------------------------------
    # YOLO11 exported output:
    #
    # (1, 7, 8400)
    #
    # 4 = x, y, width, height
    # 3 = class scores
    # --------------------------------------------------------

    output = outputs[0]

    # Remove batch dimension
    output = np.squeeze(output)

    # Convert:
    #
    # (7, 8400)
    #
    # into:
    #
    # (8400, 7)

    if output.ndim == 2:

        if output.shape[0] < output.shape[1]:

            predictions = output.T

        else:

            predictions = output

    else:

        predictions = output.reshape(
            -1,
            4 + len(CLASS_NAMES)
        )


    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    if predictions.shape[1] >= 5:

        all_class_scores = predictions[:, 4:]

        max_confidence = float(
            np.max(all_class_scores)
        )

    else:

        max_confidence = 0.0


    boxes = []
    confidences = []
    class_ids = []


    # Scaling from 640 × 640
    scale_x = original_width / INPUT_SIZE
    scale_y = original_height / INPUT_SIZE


    # ========================================================
    # PROCESS EACH PREDICTION
    # ========================================================

    for prediction in predictions:

        # First four values:
        # x center
        # y center
        # width
        # height

        x_center = float(
            prediction[0]
        )

        y_center = float(
            prediction[1]
        )

        width = float(
            prediction[2]
        )

        height = float(
            prediction[3]
        )


        # Class probabilities
        class_scores = prediction[4:]


        # Find highest scoring class
        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )


        # Ignore weak detections
        if confidence < confidence_threshold:

            continue


        # ----------------------------------------------------
        # Convert YOLO coordinates
        # ----------------------------------------------------

        x1 = int(
            (x_center - width / 2)
            * scale_x
        )

        y1 = int(
            (y_center - height / 2)
            * scale_y
        )

        x2 = int(
            (x_center + width / 2)
            * scale_x
        )

        y2 = int(
            (y_center + height / 2)
            * scale_y
        )


        # ----------------------------------------------------
        # Keep boxes inside image
        # ----------------------------------------------------

        x1 = max(
            0,
            min(x1, original_width - 1)
        )

        y1 = max(
            0,
            min(y1, original_height - 1)
        )

        x2 = max(
            0,
            min(x2, original_width - 1)
        )

        y2 = max(
            0,
            min(y2, original_height - 1)
        )


        # Ignore invalid boxes
        if x2 <= x1 or y2 <= y1:

            continue


        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        confidences.append(
            confidence
        )

        class_ids.append(
            class_id
        )


    # ========================================================
    # NO DETECTIONS
    # ========================================================

    if not boxes:

        return [], max_confidence


    # ========================================================
    # NON-MAXIMUM SUPPRESSION
    # ========================================================

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        confidence_threshold,
        0.45
    )


    detections = []


    # ========================================================
    # PROCESS NMS RESULTS
    # ========================================================

    if len(indices) > 0:

        for index in indices:

            # OpenCV may return:
            #
            # [0]
            #
            # or
            #
            # [[0]]
            #
            # or numpy scalar

            if isinstance(
                index,
                (list, tuple, np.ndarray)
            ):

                index = int(
                    np.asarray(index).flatten()[0]
                )

            else:

                index = int(index)


            x, y, width, height = boxes[index]


            detections.append({

                "class_id": class_ids[index],

                "confidence": confidences[index],

                "box": (
                    x,
                    y,
                    x + width,
                    y + height
                )

            })


    # Sort highest confidence first
    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )


    return detections, max_confidence


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    detections
):

    # PIL RGB → NumPy RGB
    image_array = np.array(
        image.convert("RGB")
    )

    # RGB → BGR for OpenCV drawing
    output_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )


    for detection in detections:

        class_id = detection["class_id"]

        confidence = detection["confidence"]

        x1, y1, x2, y2 = detection["box"]


        # Safety check
        if class_id >= len(CLASS_NAMES):

            continue


        class_name = CLASS_NAMES[
            class_id
        ]


        label = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )


        # ----------------------------------------------------
        # Label size
        # ----------------------------------------------------

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            2
        )


        label_top = max(
            0,
            y1 - text_height - baseline - 8
        )


        label_bottom = max(
            text_height + baseline + 8,
            y1
        )


        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        cv2.rectangle(

            output_image,

            (
                x1,
                label_top
            ),

            (
                x1 + text_width + 10,
                label_bottom
            ),

            (0, 0, 255),

            -1
        )


        # ----------------------------------------------------
        # Label text
        # ----------------------------------------------------

        cv2.putText(

            output_image,

            label,

            (
                x1 + 5,
                label_bottom - 6
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2,

            cv2.LINE_AA
        )


    # BGR → RGB
    output_image = cv2.cvtColor(
        output_image,
        cv2.COLOR_BGR2RGB
    )


    return output_image


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🕳️ Pothole & Road Damage Detection"
)

st.caption(
    "YOLO11-based detection of potholes, "
    "road cracks, and normal road surfaces."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Detection Settings"
)

confidence = st.sidebar.slider(

    "Confidence threshold",

    min_value=0.05,

    max_value=0.95,

    value=0.25,

    step=0.05
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a road image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a road image to start detection."
    )

    st.markdown(
        """
        ### Supported classes

        - 🕳️ **Pothole**
        - 🧱 **Crack**
        - 🛣️ **Normal Road**
        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    uploaded_file
).convert("RGB")


# ============================================================
# RUN MODEL
# ============================================================

with st.spinner(
    "Running YOLO11 detection..."
):

    detections, max_confidence = detect(

        image,

        confidence
    )


# ============================================================
# DRAW RESULTS
# ============================================================

annotated_image = draw_detections(

    image,

    detections
)


# ============================================================
# IMAGE DISPLAY
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Input Image"
    )

    st.image(
        image,
        use_container_width=True
    )


with col2:

    st.subheader(
        "Detection Result"
    )

    st.image(
        annotated_image,
        use_container_width=True
    )


# ============================================================
# DETECTION SUMMARY
# ============================================================

st.subheader(
    "Detection Summary"
)


if not detections:

    st.warning(
        "No objects were detected above "
        "the selected confidence threshold."
    )

    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    st.info(
        f"Highest raw model confidence: "
        f"{max_confidence * 100:.2f}%"
    )

    st.caption(
        "This value is shown for model validation. "
        "If the confidence is very low even for a clear "
        "pothole image, we need to inspect the ONNX "
        "preprocessing/output before deployment."
    )


else:

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    summary = {}


    for detection in detections:

        class_id = detection["class_id"]


        if class_id >= len(CLASS_NAMES):

            continue


        class_name = CLASS_NAMES[
            class_id
        ]


        if class_name not in summary:

            summary[class_name] = []


        summary[class_name].append(
            detection["confidence"]
        )


    # ========================================================
    # METRICS
    # ========================================================

    metric_columns = st.columns(
        max(1, len(summary))
    )


    for column, (
        class_name,
        values
    ) in zip(
        metric_columns,
        summary.items()
    ):

        with column:

            st.metric(

                class_name,

                len(values)
            )


    # ========================================================
    # INDIVIDUAL DETECTIONS
    # ========================================================

    st.subheader(
        "Individual Detections"
    )


    table_data = []


    for detection in detections:

        class_id = detection["class_id"]


        if class_id >= len(CLASS_NAMES):

            continue


        table_data.append({

            "Class": CLASS_NAMES[
                class_id
            ],

            "Confidence": (
                f"{detection['confidence'] * 100:.2f}%"
            )

        })


    st.dataframe(

        table_data,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Model: Custom YOLO11 model trained for "
    "pothole, crack, and normal-road detection."
)