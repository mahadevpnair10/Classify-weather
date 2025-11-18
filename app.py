import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer

# Setup class names
with open("class_names.txt", 'r') as f:
  classes = [name.strip() for name in f]

# Model and transforms
model, transform = create_effnetb2_model(
    num_classes=len(classes)
)

model.load_state_dict(
    torch.load(
        f="model_v3.pth",
        map_location=torch.device("cpu")
    )
)

# Predict function
def predict(img):
 
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = transform(img).unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        
        predictions = torch.softmax(model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio)
    pred_labels_and_probs = {classes[i]: float(predictions[0][i]) for i in range(len(classes))}
    
    pred_time = round(timer() - start_time, 4)
    
    return pred_labels_and_probs, pred_time

example_list = [["examples/" + example] for example in os.listdir("examples")]

# Gradio interface
title = "Weather image classification ⛅❄☔"
description = "Classifies the weather conditions from an image, capable of distinguishing among 12 distinct classes."
article = "See the code on [GitHub](https://github.com/georgescutelnicu/Weather-Image-Classification)."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=1, label="Predictions"),
        gr.Number(label="Prediction time (s)")
    ],
    examples=example_list,
    title=title,
    description=description,
    article="",                # remove text above footer
    allow_flagging="never",    # remove flag button
    css=".footer {display: none !important}"   # hide footer completely
)

demo.launch(debug=False,
            share=False)
