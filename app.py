import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os

# Load the YOLO model
model = YOLO("models/best.pt")

# Prediction function
def yolo_predict(image):
    results = model(image)[0]
    annotated_img = results.plot()
    label_conf = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        label_conf[label] = max(conf, label_conf.get(label, 0))
    return Image.fromarray(annotated_img), label_conf or {"No object detected": 1.0}

# Collect example images
example_dir = "examples"
example_files = [
    os.path.join(example_dir, f)
    for f in os.listdir(example_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Gradio app using Blocks layout
with gr.Blocks() as demo:
    gr.Markdown("## Ocean Garbage Detector (YOLOv10)")
    gr.Markdown("""
                ### 🧭 Welcome to CleanSea: Ocean Garbage Detector

                This tool uses a fine-tuned YOLOv10 model to detect **marine debris** such as plastic bottles, bags, nets, and more in underwater images.

                To use:
                - Upload an underwater photo or choose an example image.
                - Click **Detect** to run the model.
                - View predictions (highlighted in the image) and confidence scores below.

                🔍 The tool can help monitor ocean pollution, support environmental research, or educate students about AI for sustainability.
                """)


    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Ocean Image")
        with gr.Column():
            bbox_output = gr.Image(label="Detected Objects")
            label_output = gr.Label(num_top_classes=5)

    gr.Examples(examples=example_files, inputs=image_input, cache_examples=False)

    submit_btn = gr.Button("Detect")
    submit_btn.click(fn=yolo_predict, inputs=image_input, outputs=[bbox_output, label_output])

    gr.Markdown("""
            ---

            ### 📈 Model Performance Visualizations

            These visualizations summarize the training and evaluation of our YOLOv10 model on a custom ocean trash dataset.

            - **Training Metrics**: Tracks loss and performance over 120 training epochs.
            - **Confusion Matrix**: Shows how well the model distinguishes among 16 garbage categories.
            - **F1 / Precision / PR Curves**: Evaluate how confident and accurate the model is across classes.

            📊 These help researchers assess strengths, weaknesses, and opportunities for improvement.
            """)

    with gr.Row():
        gr.Image("results/results.png", label="Training Metrics")
        gr.Image("results/confusion_matrix_normalized.png", label="Confusion Matrix")
    with gr.Row():
        gr.Image("results/F1_curve.png", label="F1-Confidence Curve")
        gr.Image("results/P_curve.png", label="Precision-Confidence Curve")
        gr.Image("results/PR_curve.png", label="Precision-Recall Curve")

# Launch app locally
demo.launch()
