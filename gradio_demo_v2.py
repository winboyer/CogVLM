import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import base64
import json
import requests
import base64
import hashlib
import torch
import time
import re
import argparse


def clear_fn2(value):
    return default_chatbox


if __name__ == '__main__':
    gr.close_all()
    examples = []

    is_grounding = False
    example_ids = list(range(3)) if not is_grounding else list(range(3,6,1))
    with open("./examples/example_inputs.jsonl") as f:
        for i, line in enumerate(f):
            if i not in example_ids: continue
            data = json.loads(line)
            examples.append(data)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=7):
                # result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, What do you want to know about?")]).style(height=700)
                result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, What do you want to know about?")]).style(height=500)
                hidden_image_hash = gr.Textbox(visible=False)

            with gr.Column(scale=4):
                # image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                # video_prompt = gr.Video(type="file", label="Video Prompt", value=None)
                file_prompt = gr.File(label="File Prompt", file_types=["image",".mp4",".ts",".avi"], value=None)
                gallery_prompt = gr.Gallery(label='image', visible=False)

        with gr.Group():
            input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
            is_grounding = gr.Checkbox(label="Grounding")
            with gr.Row():
                run_button = gr.Button('Generate')
                clear_button = gr.Button('Clear')

            with gr.Row():
                temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                top_k = gr.Slider(maximum=50, value=5, minimum=1, step=1, label='Top K')

        gr_examples = gr.Examples(examples=[[example["text"], example["image"]] for example in examples], 
                                inputs=[input_text, file_prompt],
                                label="Example Inputs (Click to insert an example into the input box)",
                                examples_per_page=6)
        print(gr.__version__)
        # run_button.click(fn=chat_pred,inputs=[input_text, temperature, top_p, top_k, file_prompt, result_text, hidden_image_hash],
        #                     outputs=[input_text, result_text, hidden_image_hash])
        input_text.submit(fn=clear_fn2,inputs=[input_text, temperature, top_p, top_k, file_prompt, result_text, hidden_image_hash],
                            outputs=[input_text, result_text, hidden_image_hash])
        # clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, image_prompt])
        # image_prompt.upload(fn=clear_fn2, inputs=clear_button, outputs=[result_text])

        print(gr.__version__)


    # demo.queue(concurrency_count=10)
    # demo.launch(share=True)
    demo.launch(server_name="0.0.0.0", show_error=True, server_port=7860)
    # demo.launch(server_name="0.0.0.0", server_port=8088-8089)

