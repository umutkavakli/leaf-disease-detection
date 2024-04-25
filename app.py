import gradio as gr
from detection import ObjectDetection

examples = [
    ['.test-images/toma_TMV_jpg.rf.fadfed6938fbeef46c39f78a02da3be4.jpg', 0.31],
    ['.test-images/99e886623c2080c22f6519b0e708c531_jpg.rf.83ea9a32bc50cfb2da6e4a39337532cc.jpg', 0.51],
    ['.test-images/early-blight-septoria-ls-fig-3_jpg.rf.7b2e29c077910e0930c16d7aed136121.jpg', 0.39],
    ['.test-images/glyphosate_jpg.rf.01dea2d24a1e3591855a68e077bb625e.jpg', 0.54],
    ['.test-images/image_jpg.rf.d37866429a917dd1bc5352ea1454a472.jpg', 0.41]
]

def get_predictions(img, threshold, box_color, text_color):
    v3_results = yolov3_detector.score_frame(img)
    v5_results = yolov5_detector.score_frame(img)
    v8_results = yolov8_detector.v8_score_frame(img)

    v3_frame = yolov3_detector.plot_bboxes(v3_results, img, float(threshold), box_color, text_color)
    v5_frame = yolov5_detector.plot_bboxes(v5_results, img, float(threshold), box_color, text_color)
    v8_frame = yolov8_detector.plot_bboxes(v8_results, img, float(threshold), box_color, text_color)


    return v3_frame, v5_frame, v8_frame


with gr.Blocks(title="Leaf Disease Detection", theme=gr.themes.Monochrome()) as interface:
    gr.Markdown("# Leaf Disease Detection")
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Input Image")
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    box_color = gr.ColorPicker(label="Box Color", value="#0000ff")
                with gr.Column():
                    text_color = gr.ColorPicker(label="Prediction Color", value="#ff0000")

            confidence = gr.Slider(maximum=1, step=0.01, value=0.4, label="Confidence Threshold", interactive=True)
            btn = gr.Button("Detect")
    
    with gr.Row():        
        v3_prediction = gr.Image(label="YOLOv3")  
        v5_prediction = gr.Image(label="YOLOv5")
        v8_prediction = gr.Image(label="YOLOv8")

        btn.click(
            get_predictions,
            [image, confidence, box_color, text_color],
            [v3_prediction, v5_prediction, v8_prediction]
        )
    
    with gr.Row():
        gr.Examples(examples=examples, inputs=[image, confidence])


yolov3_detector = ObjectDetection('yolov3')
yolov5_detector = ObjectDetection('yolov5')
yolov8_detector = ObjectDetection('yolov8')

interface.launch()

