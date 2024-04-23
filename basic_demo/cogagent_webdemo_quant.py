import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import base64
import json
import requests
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import base64
import hashlib
import torch
import time
import re
import argparse
import cv2
import matplotlib.font_manager

from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
parser.add_argument("--chinese", action='store_true', help='Chinese interface')
parser.add_argument("--english", action='store_true', help='only output English')
parser.add_argument("--version", type=str, default="chat", help='version to interact with')
parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="/root/jinyfeng/models/CogVLM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="/root/jinyfeng/models/CogVLM/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--stream_chat", action="store_true")
parser.add_argument("--server_port", type=int, default=7861, help='the gradio server port')
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

MODEL_PATH = args.from_pretrained
if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True,
        )
    device_map = infer_auto_device_map(model, max_memory={0:'16GiB',1:'20GiB','cpu':'18GiB'}, no_split_module_classes='CogAgentDecoderLayer')
    model = load_checkpoint_and_dispatch(
        model,
        MODEL_PATH,
        device_map=device_map,
    )
    model = model.eval()

early_stop = False

def process_video(file_path, interval_time=4.0):
    videoCapture = cv2.VideoCapture(file_path)
    success, frame = videoCapture.read()
    frame_cnt = 0
    # 视频中，1s包含25帧
    ret_list = []
    while success :
        frame_cnt = frame_cnt + 1
        if frame_cnt%(25*interval_time) != 0: # 隔4s抽一帧
            success, frame = videoCapture.read()
            continue
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # pil_img = pil_img.convert('RGB')
        ret = (pil_img)
        ret_list.append(ret)

        success, frame = videoCapture.read()
    print('frame_cnt, len(ret_list)=========', frame_cnt, len(ret_list))
    videoCapture.release()

    return ret_list

from utils.utils.grounding_parser import text_to_dict
def draw_response2img(img, boxes):
    img = img.convert('RGB')
    width, height = img.size
    ratio = min(1920 / width, 1080 / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    new_img = img.resize((new_width, new_height), Image.LANCZOS)
        
    width, height = new_img.size
    absolute_boxes = [[(int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)) for box in b] for b in boxes]
    
    color_palette = sns.color_palette("husl", len(absolute_boxes))
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in color_palette]

    overlay = Image.new('RGBA', new_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    # draw = ImageDraw.Draw(new_img)
    # print('len(boxes)===========', len(boxes))
    for box, color in zip(absolute_boxes, colors):
        # print(len(box[0]), box[0])
        # draw.rectangle([box[0][0], box[0][1], box[0][2], box[0][3]], outline=color, width=box_width)
        draw.rectangle(box[0], outline=color, width=5)
    
    img_with_overlay = Image.alpha_composite(new_img.convert('RGBA'), overlay).convert('RGB')

    return img_with_overlay


default_chatbox = [("", "Hi, What do you want to know about from cogagent?")]

def http_post(
        input_text,
        temperature,
        top_p,
        top_k,
        file_prompt,
        result_previous,
        hidden_image,
        ):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    gallery_prompt = []
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][0] == None:
            del result_text[i]
    print(f"history {result_text}")

    filepath_extname = os.path.splitext(file_prompt.name)[-1]
    # print('filepath_extname, file_prompt.name============', filepath_extname, file_prompt.name)
    video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
    global early_stop
    early_stop = False
    try:
        if video_flag:
            print('web_video_chat')

            image_list = process_video(file_prompt.name)
            ret_img_list = []
            response_list = []
            ret_cnt=0
            for pil_img in image_list:
                if early_stop:
                    break

                input_by_model = model.build_conversation_input_ids(tokenizer, query=input_text, history=[], images=[pil_img])
                inputs = {
                    'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                    'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                    'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                    'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
                }
                if 'cross_images' in input_by_model and input_by_model['cross_images']:
                    inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

                # add any transformers params here.
                gen_kwargs = {"max_length": args.max_length,
                            "temperature": temperature,
                            "do_sample": False}

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    response = tokenizer.decode(outputs[0])
                    response = response.split("</s>")[0]
                    print("\nCogAgent:", response)

                # grounding
                if ' no ' in response or 'not visible' in response:
                    continue

                response_dic = text_to_dict(response)
                if not response_dic:
                    texts = []
                    boxes = []
                else:
                    texts, boxes = zip(*response_dic.items())
                if len(boxes)<1:
                    continue

                ret_cnt += 1
                draw_img = draw_response2img(pil_img, boxes)

                gallery_prompt.append(draw_img)
                answer = 'image '+str(ret_cnt)+': '+response
                result_text.append((input_text, answer))
                # print(result_text)

                yield "", result_text, hidden_image, gallery_prompt

            # image_path_grounding = './results/output.png'
            # show_image(cache_image, image_path_grounding)
            # result_text.append((None, (image_path_grounding,)))
            
    except Exception as e:
        print("error message", e)
        result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        # return "", result_text, hidden_image, gallery_prompt
        return "", result_text, hidden_image, gallery_prompt

    return "", result_text, hidden_image, gallery_prompt


def show_image(img, output_fn='./results/output.png'):
    img = img.convert('RGB')
    width, height = img.size
    ratio = min(1920 / width, 1080 / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    new_img = img.resize((new_width, new_height), Image.LANCZOS)

    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    img_with_overlay.save(output_fn)


def clear_fn(value):
    global early_stop
    early_stop = True
    return "", default_chatbox, None, None

def clear_fn2(value):
    return default_chatbox, None


def imagefile_upload_fn(file_prompt):
    filepath_extname = os.path.splitext(file_prompt.name)[-1]
    # print('filepath_extname, file_prompt.name============', filepath_extname, file_prompt.name)
    video_flag = filepath_extname in ('.mp4', 'avi', '.ts', '.mpg', '.mpeg', '.rm', '.rmvb', '.mov', '.wmv')
    if not video_flag:
        return default_chatbox, file_prompt.name
    else:
        return default_chatbox, None

def change_gallery(result_text, gallery_prompt):

    return result_text, gallery_prompt

if __name__ == '__main__':
    gr.close_all()
    examples = []
    # is_grounding = False
    # example_ids = list(range(3)) if not is_grounding else list(range(3,6,1))
    # with open("./examples/example_inputs.jsonl") as f:
    #     for i, line in enumerate(f):
    #         if i not in example_ids: continue
    #         data = json.loads(line)
    #         examples.append(data)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=7):
                result_text = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about from cogagent?")], 
                                                height=500)
                hidden_image_hash = gr.Textbox(visible=False)

            with gr.Column(scale=4):
                # image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                # video_prompt = gr.Video(type="file", label="Video Prompt", value=None)
                file_prompt = gr.File(label="file prompt", 
                # file_types=["image",".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"], 
                file_types=[".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"], 
                value=None)
                gallery_prompt = gr.Gallery(label='chat image', height=300)

        with gr.Group():
            input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
            with gr.Row():
                run_button = gr.Button('Generate')
                clear_button = gr.Button('Clear')

            with gr.Row():
                temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                top_k = gr.Slider(maximum=50, value=5, minimum=1, step=1, label='Top K')
            
        # gr_examples = gr.Examples(examples=[[example["text"], example["image"]] for example in examples], 
        #                         inputs=[input_text, file_prompt],
        #                         label="Example Inputs (Click to insert an example into the input box)",
        #                         examples_per_page=6)
        
        print(gr.__version__)
        run_button.click(fn=http_post,inputs=[input_text, temperature, top_p, top_k, file_prompt, result_text, hidden_image_hash],
                            outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        input_text.submit(fn=http_post,inputs=[input_text, temperature, top_p, top_k, file_prompt, result_text, hidden_image_hash],
                            outputs=[input_text, result_text, hidden_image_hash, gallery_prompt])
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, file_prompt, gallery_prompt])
        file_prompt.upload(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])
        # file_prompt.upload(fn=imagefile_upload_fn, inputs=file_prompt, outputs=[result_text, gallery_prompt])
        # file_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text, gallery_prompt])
        file_prompt.clear(fn=clear_fn2, inputs=file_prompt, outputs=[result_text, gallery_prompt])

        # gallery_prompt.change(fn=change_gallery, inputs=[result_text, gallery_prompt], outputs=[result_text, gallery_prompt])

        print(gr.__version__)

    demo.queue(concurrency_count=10)
    # demo.launch(share=True)
    demo.launch(server_name="0.0.0.0", show_error=True, server_port=args.server_port)
    # demo.launch(server_name="0.0.0.0", server_port=8088-8089)

