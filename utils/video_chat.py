# -*- encoding: utf-8 -*-
'''
@File    :   chat.py
@Time    :   2023/05/08 19:10:08
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import requests
from PIL import Image
from io import BytesIO

import torch
from sat.generation.autoregressive_sampling import filling_sequence, get_masks_and_position_ids_default
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.mpu import get_model_parallel_rank

import cv2


def process_image(image_path, img_processor, image):
    if image is None:
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)

    if image is not None and isinstance(image, Image.Image):
        pil_img = image.convert('RGB')
        img_dict = img_processor(pil_img)
        ret = (img_dict, pil_img)
    else:
        ret = image
    return ret

def process_video(image_path, img_processor, image):
    
    assert image is None, 'in process_video, the image is not None !!!!!!'
    
    videoCapture = cv2.VideoCapture(image_path)
    success, frame = videoCapture.read()
    frame_cnt = 0
    # 视频中，1s包含25帧
    ret_list = []
    while success :
        frame_cnt = frame_cnt + 1
        if frame_cnt%20 != 0:
            success, frame = videoCapture.read()
            continue
        
        pil_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        # pil_img = pil_img.convert('RGB')
        img_dict = img_processor(pil_img)
        ret = (img_dict, pil_img)
        ret_list.append(ret)
        success, frame = videoCapture.read()
    # print('frame_cnt, len(ret_list)=========', frame_cnt, len(ret_list))

    return ret_list

def chat(image_path, model, text_processor, img_processor,
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 4096, top_p=0.95, top_k=5, temperature=0.95, repetition_penalty=1.0,
        invalid_slices=[], no_prompt=False
        ):
    if image is None:
        assert image_path is not None
    if not history:
        history = []

    if no_prompt:
        query = ''
    prompt = text_processor.history_to_prompt(query, history)

    (torch_image, pil_img) = process_image(image_path, img_processor, image)

    if torch_image is not None:
        for k in torch_image:
            if type(torch_image[k]) is torch.Tensor and torch_image[k].dtype is not torch.int and torch_image[k].dtype is not torch.long:
                torch_image[k] = torch_image[k].to(next(model.parameters()).dtype)
            if type(torch_image[k]) is torch.Tensor:
                torch_image[k] = torch_image[k].to(next(model.parameters()).device)

    inputs_dic = text_processor(prompt)
    for k in inputs_dic:
        if type(inputs_dic[k]) is torch.Tensor and inputs_dic[k].dtype is not torch.int and inputs_dic[k].dtype is not torch.long:
            inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).dtype)
        if type(inputs_dic[k]) is torch.Tensor:
            inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).device)
    input_ids = inputs_dic['input_ids'].to(model.parameters().__next__().device)[0]
    
    if max_length-len(input_ids) <= 1:
        response = "The prompt exceeds the context length limit, please try again."
        return response, history, (torch_image, pil_img)
    
    seq = torch.cat(
        [input_ids, torch.tensor([-1]*(max_length-len(input_ids)), device=input_ids.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
                            invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
    # use beam search to get a better result
    # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
    #                               num_beams=5, consider_end=True, repetition_penalty=repetition_penalty)
    get_func = text_processor.get_func(input_ids, **inputs_dic) if hasattr(text_processor, 'get_func') else get_masks_and_position_ids_default

    img_inputs = {'vision_'+k: v for k, v in torch_image.items()}
    inputs_dic.pop('input_ids')
    inputs = {**img_inputs, **inputs_dic}

    output = filling_sequence(
        model, seq,
        batch_size=1,
        get_masks_and_position_ids=get_func,
        strategy=strategy,
        **inputs
    )[0] # drop memory
    
    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq
    if type(output) is not list:
        output_list = output.tolist()
    else:
        output_list = output

    response = text_processor.tokenizer.decode(output_list[0])
    # print('original:', response)
    if hasattr(text_processor, 'process_response'):
        response = text_processor.process_response(response)
    response = response.split(text_processor.sep)[-1].strip()
    if get_model_parallel_rank() == 0:
        from utils.parser import parse_response
        parse_response(pil_img, response)
    history = history + [(query, response)]
    return response, history, (torch_image, pil_img)

def video_chat(image_path, model, text_processor, img_processor,
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 4096, top_p=0.95, top_k=5, temperature=0.95, repetition_penalty=1.0,
        invalid_slices=[], no_prompt=False
        ):
    if image is None:
        assert image_path is not None
    if not history:
        history = []

    if no_prompt:
        query = ''
    prompt = text_processor.history_to_prompt(query, history)

    image_list = process_video(image_path, img_processor, image)
    ret_torch_image, ret_pil_img = None, None
    image_cnt = 0
    # print('len(image_list)=========', len(image_list))
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
                            invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
    # use beam search to get a better result
    # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
    #                               num_beams=5, consider_end=True, repetition_penalty=repetition_penalty)
    for torch_image, pil_img in image_list:
        # image_cnt += 1
        if torch_image is not None:
            for k in torch_image:
                if type(torch_image[k]) is torch.Tensor and torch_image[k].dtype is not torch.int and torch_image[k].dtype is not torch.long:
                    torch_image[k] = torch_image[k].to(next(model.parameters()).dtype)
                if type(torch_image[k]) is torch.Tensor:
                    torch_image[k] = torch_image[k].to(next(model.parameters()).device)

        img_inputs = {'vision_'+k: v for k, v in torch_image.items()}

        inputs_dic = text_processor(prompt)
        for k in inputs_dic:
            if type(inputs_dic[k]) is torch.Tensor and inputs_dic[k].dtype is not torch.int and inputs_dic[k].dtype is not torch.long:
                inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).dtype)
            if type(inputs_dic[k]) is torch.Tensor:
                inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).device)
        input_ids = inputs_dic['input_ids'].to(model.parameters().__next__().device)[0]
        if max_length-len(input_ids) <= 1:
            response = "The prompt exceeds the context length limit, please try again."
            return response, history, (torch_image, pil_img)

        seq = torch.cat(
            [input_ids, torch.tensor([-1]*(max_length-len(input_ids)), device=input_ids.device)], dim=0
        )
        
        get_func = text_processor.get_func(input_ids, **inputs_dic) if hasattr(text_processor, 'get_func') else get_masks_and_position_ids_default
        inputs_dic.pop('input_ids')
        inputs = {**img_inputs, **inputs_dic}

        output = filling_sequence(
            model, seq,
            batch_size=1,
            get_masks_and_position_ids=get_func,
            strategy=strategy,
            **inputs
        )[0] # drop memory
    
        # ---------------
        # port from inference_glm.py, more general than chat mode
        # clip -1s and fill back generated things into seq
        if type(output) is not list:
            output_list = output.tolist()
        else:
            output_list = output

        response = text_processor.tokenizer.decode(output_list[0])        
        # print('query, original response ===========:', query, response)

        if hasattr(text_processor, 'process_response'):
            response = text_processor.process_response(response)
        response = response.split(text_processor.sep)[-1].strip()
        # print('ori response=========', response)
        if get_model_parallel_rank() == 0:
            from utils.parser import parse_response
            parse_response(pil_img, response)
            # print('get_model_parallel_rank()========', get_model_parallel_rank())
            # print('parse response=========', response)

        # verb_noun_list, det_list = parse_query_response(response)
        # 判断 response中的关键字
        if ' no ' in response or 'not visible' in response:
            # print('verb_noun_list, det_list========', verb_noun_list, det_list)
            ret_torch_image, ret_pil_img = torch_image, pil_img
            # history = history + [(query, response)]
            continue

        ret_torch_image, ret_pil_img = torch_image, pil_img
        history = history + [(query, response)]
        # print('history===========', history)

        break
    # print('image_cnt=======', image_cnt)

    return response, history, (ret_torch_image, ret_pil_img)
