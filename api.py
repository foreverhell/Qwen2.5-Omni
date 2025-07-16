#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import os
import queue
import signal
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.request import urlopen

import librosa
import numpy as np
import psutil
import requests
import resampy
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from vllm.engine.arg_utils import nullable_str
from vllm.engine.async_llm_engine import AsyncEngineArgs
from vllm.engine.omni_llm_engine import OmniLLMEngine # OptimizedOmniLLMEngine as 
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.multimodal.processing_omni import fetch_image, fetch_video
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
from io import BytesIO
import subprocess
import base64,io
logger = init_logger('vllm.omni')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="/mnt/diskhd/Backup/DownloadModel/Qwen2.5-Omni-7B/")
parser.add_argument('--thinker-model', type=str, default=None)
parser.add_argument('--talker-model', type=str, default=None)
parser.add_argument('--code2wav-model', type=str, default=None)
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--legacy-omni-video', action='store_true')
parser.add_argument("--thinker-only", action="store_true")
parser.add_argument("--text-only", action="store_true")
parser.add_argument("--do-wave", action="store_true")
parser.add_argument('--max-num-seqs', type=int, default=4)#64
parser.add_argument('--block-size', type=int, default=16)#16
parser.add_argument('--enforce-eager', action='store_true') #调试用的，正式环境不要
parser.add_argument('--thinker-enforce-eager', action='store_true')
parser.add_argument('--talker-enforce-eager', action='store_true')
parser.add_argument('--enable-prefix-caching', action='store_true')
parser.add_argument('--thinker-quantization',
                    type=nullable_str,
                    choices=QUANTIZATION_METHODS)
parser.add_argument('--talker-quantization',
                    type=nullable_str,
                    choices=QUANTIZATION_METHODS)
parser.add_argument('--enable-torch-compile', action='store_true')#静态图，不能流式返回
parser.add_argument('--enable-torch-compile-first-chunk', action='store_true')
parser.add_argument("--odeint-method",
                    type=str,
                    default="euler",
                    choices=["euler", "rk4"])#默认rk4
parser.add_argument("--odeint-method-relaxed", action="store_true")
parser.add_argument('--code2wav-steps', type=int, default=5)
parser.add_argument("--batched-chunk", type=int, default=2)#50Hz时默认为2
parser.add_argument("--code2wav-frequency",
                    type=str,
                    default='50hz',
                    choices=['50hz'])
parser.add_argument('--voice-type', type=nullable_str, default='m02')
parser.add_argument('--warmup-voice-type', type=nullable_str, default='m02')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--max-tokens", type=int, default=2048)#2048
parser.add_argument("--num-prompts", type=int, default=1)#1
parser.add_argument('--sample-rate', type=int, default=24000)#生成的采样率
parser.add_argument('--use-torchvision', action='store_true')
parser.add_argument('--prompt',
                    choices=[
                        'text', 'audio', 'audio-long', 'audio-long-chunks',
                        'audio-long-expand-chunks', 'image', 'video',
                        'video-frames', 'audio-in-video', 'audio-in-video-v2',
                        "audio-multi-round", "badcase-vl", "badcase-text",
                        "badcase-image-early-stop", "badcase-two-audios",
                        "badcase-two-videos", "badcase-multi-round",
                        "badcase-voice-type", "badcase-voice-type-v2",
                        "badcase-audio-tower-1", "badcase-audio-only"
                    ],
                    default='audio-in-video-v2')

parser.add_argument('--thinker-devices', type=json.loads, default="[0]")
parser.add_argument('--talker-devices', type=json.loads, default="[1]")
parser.add_argument('--code2wav-devices', type=json.loads, default="[1]")
parser.add_argument('--code2wav-dynamic-batch',
                    action='store_true',
                    help='Enable code2wav dynamic batch')
parser.add_argument('--thinker-gpu-memory-utilization',
                    type=float,
                    default=0.9)
parser.add_argument('--talker-gpu-memory-utilization', type=float, default=0.9)

parser.add_argument('-o',
                    '--output-dir',
                    type=str,
                    default='output_wav',
                    help="Audio output directory")
#server
parser.add_argument('--host', type=str, default='0.0.0.0', help="服务器监听地址")
parser.add_argument('--port', type=int, default=8901, help="服务器监听端口")
args = parser.parse_args()
if args.thinker_model is None or not os.path.exists(args.thinker_model):
    if os.path.exists(f'{args.model}/thinker'):
        args.thinker_model = f'{args.model}/thinker'
    else:
        args.thinker_model = f"{args.model}"
if args.talker_model is None or not os.path.exists(args.talker_model):
    if os.path.exists(f'{args.model}/talker'):
        args.talker_model = f'{args.model}/talker'
    else:
        args.talker_model = f"{args.model}"
if args.code2wav_model is None or not os.path.exists(args.code2wav_model):
    if os.path.exists(f'{args.model}/code2wav'):
        args.code2wav_model = f'{args.model}/code2wav'
    else:
        args.code2wav_model = f"{args.model}"

omni = None
processor = None
tokenizer = None
@asynccontextmanager
async def lifespan(app:FastAPI):
    global omni, processor, tokenizer
    # init engine
    omni = init_omni_engine()
    processor = AutoProcessor.from_pretrained(args.thinker_model)
    tokenizer = AutoTokenizer.from_pretrained(args.thinker_model)
    warmup()
    yield
    #关闭时清理
    if omni:
        omni.shutdown()
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            os.kill(child.pid, signal.SIGTERM)
def warmup():
    # warmup
    prompt = make_omni_prompt(audio_path="/home/yiyangzhe/Qwen2.5-Omni/test.wav")
    run_omni_engine(prompt, omni, num_prompts=args.num_prompts, is_warmup=True)
        
app = FastAPI(
    title="Qwen-Omni Compatible API", 
    description="API compatible with OpenAI format for Qwen-Omni model",
    lifespan=lifespan
)

def pcm_2_waveform(pcm_data: bytes) -> np.array:
    if len(pcm_data) & 1:
        pcm_data = pcm_data[:-1] # 保证偶数个字节
    int16_array = np.frombuffer(pcm_data, dtype=np.int16)
    waveform = int16_array.astype(np.float32) / (1<<15)
    return waveform

def resample_wav_to_16khz(input_filepath: str):
    if input_filepath.startswith("data:audio"):
        audio_bytes = base64.b64decode(input_filepath.split(',', 1)[1])
        with io.BytesIO(audio_bytes) as f:
            data, original_sample_rate = sf.read(f)            
    elif os.path.exists(input_filepath):
        data, original_sample_rate = sf.read(input_filepath)
    # Only use the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    # resample to 16kHz
    data_resampled = resampy.resample(data,
                                      sr_orig=original_sample_rate,
                                      sr_new=16000)
    return data_resampled


def fetch_and_read_video(video_url: str, fps=2):
    import torchvision.io

    def read_video_with_torchvision(video_file_name: str):
        video, audio, info = torchvision.io.read_video(
            video_file_name,
            start_pts=0.0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW",
        )

        total_frames, video_fps = video.size(0), info["video_fps"]
        total_duration = round(total_frames / video_fps, 3)
        nframes = int(total_frames / video_fps * fps)

        frame_timestamps = total_duration * torch.arange(1,
                                                         nframes + 1) / nframes
        grid_timestamps = frame_timestamps[::2]
        second_per_grid = grid_timestamps[1] - grid_timestamps[0]

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        video_height, video_width = video.shape[2:]
        video = video[idx]

        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid.item()]
        else:
            return video

    def read_video_with_transformers(video_file_name: Union[str, List[str]]):
        video, total_duration, nframes, second_per_grid = fetch_video(
            {'video': video_file_name})
        if total_duration is None and nframes is None:
            nframes = len(video)
            total_duration = 0.5 * nframes
            second_per_grid = 1.0
        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid]
        else:
            return video

    def read_video(video_file_name: str):
        if args.use_torchvision:
            return read_video_with_torchvision(video_file_name)
        else:
            return read_video_with_transformers(video_file_name)

    if isinstance(video_url, str) and video_url.startswith("http"):
        with tempfile.NamedTemporaryFile(delete=True) as temp_video_file:
            resp = requests.get(video_url)
            assert resp.status_code == requests.codes.ok, f"Failed to fetch video from {video_url}, status_code:{resp.status_code}, resp:{resp}"

            temp_video_file.write(urlopen(video_url).read())
            temp_video_file_path = temp_video_file.name
            video_file_name = temp_video_file_path
            return read_video(video_file_name)
    else:
        video_file_name = video_url
        return read_video(video_file_name)


def make_inputs_qwen2_omni(
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
    use_audio_in_video: Optional[bool] = False,
    tokenize: bool = args.tokenize,
) -> Union[TokensPrompt, TextPrompt]:
    try:
        config = AutoConfig.from_pretrained(args.thinker_model)
        if 'Qwen2_5OmniModel' in config.architectures:
            args.legacy_omni_video = False
        else:
            args.legacy_omni_video = True
    except:
        args.legacy_omni_video = True

    audios, images, videos = [], [], []
    for message in messages:
        if not isinstance(message['content'], list):
            message['content'] = [{
                'type': 'text',
                'text': message['content'],
            }]
        index, num_contents = 0, len(message['content'])
        while index < num_contents:
            ele = message['content'][index]
            if 'type' not in ele:
                if 'text' in ele:
                    ele['type'] = 'text'
                elif 'audio' in ele:
                    ele['type'] = 'audio'
                elif 'audio_url' in ele:
                    ele['type'] = 'audio_url'
                elif 'image' in ele:
                    ele['type'] = 'image'
                elif 'image_url' in ele:
                    ele['type'] = 'image_url'
                elif 'video' in ele:
                    ele['type'] = 'video'
                elif 'video_url' in ele:
                    ele['type'] = 'video_url'
                else:
                    raise ValueError(f'Unknown ele: {ele}')
            if ele['type'] == 'audio' or ele['type'] == 'audio_url':
                if 'audio_url' in ele:
                    audio_key = 'audio_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_audio_file:
                        temp_audio_file.write(urlopen(ele[audio_key]).read())
                        temp_audio_file_path = temp_audio_file.name
                        audios.append(
                            resample_wav_to_16khz(temp_audio_file_path))
                        ele['audio'] = temp_audio_file_path
                elif 'audio' in ele:
                    audio_key = 'audio'
                    audios.append(resample_wav_to_16khz(ele[audio_key]))
                else:
                    raise ValueError(f'Unknown ele {ele}')
            elif use_audio_in_video and (ele['type'] == 'video'
                                         or ele['type'] == 'video_url'):
                # use video as audio as well
                if 'video_url' in ele:
                    audio_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele[audio_key]).read())
                        temp_video_file_path = temp_video_file.name
                        ele[audio_key] = temp_video_file_path
                        audios.append(
                            librosa.load(temp_video_file_path, sr=16000)[0])
                        videos.append(
                            fetch_and_read_video(temp_video_file_path))
                        ele['video'] = temp_video_file_path
                elif 'video' in ele:
                    audio_key = 'video'
                    audios.append(librosa.load(ele[audio_key], sr=16000)[0])
                    videos.append(fetch_and_read_video(audio_key))
                else:
                    raise ValueError("Unknown ele {}".format(ele))
                # insert a audio after the video
                message['content'].insert(index + 1, {
                    "type": "audio",
                    "audio": ele[audio_key],
                })
                # no need to load the added audio again
                index += 1
            elif ele['type'] == 'video' or ele['type'] == 'video_url':
                if 'video_url' in ele:
                    video_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele['video_url']).read())
                        temp_video_file_path = temp_video_file.name
                        videos.append(fetch_and_read_video(temp_video_file))
                        ele['video'] = temp_video_file_path
                else:
                    video_key = 'video'
                    videos.append(fetch_and_read_video(ele[video_key]))
            elif ele['type'] == 'image' or ele['type'] == 'image_url':
                images.append(fetch_image(ele))

            # move to the next content
            index += 1
    prompt = processor.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    
    audios = audios if len(audios) > 0 else None
    images = images if len(images) > 0 else None
    videos = videos if len(videos) > 0 else None

    logger.info(f'{prompt}, '
                f'audios = {len(audios) if audios else None}, '
                f'images = {len(images) if images else None}, '
                f'videos = {len(videos) if videos else None}')

    multi_modal_data = {}
    if audios:
        multi_modal_data["audio"] = audios
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    # pass through the use_audio_in_video to llm engine
    multi_modal_data["use_audio_in_video"] = use_audio_in_video

    if isinstance(prompt, list) and isinstance(prompt[0], (list, str)):
        prompt = prompt[0]

    if tokenize:
        return TokensPrompt(
            prompt_token_ids=prompt,
            multi_modal_data=multi_modal_data,
        )
    else:
        return TextPrompt(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
        )

default_system = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
def get_system_prompt(system:str = None):
    if args.text_only:
        return {
            'role': 'system',
            'content': [{
                'type': 'text',
                'text': system if system else 'You are a helpful assistant.'
            }]
        }
    else:
        return {
            'role':
            'system',
            'content': [{
                'type':
                'text',
                'text':
                system if system else default_system
            }]
        }


def make_text_prompt(system:str = None):
    messages = [
        get_system_prompt(system),
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Who are you?"
                },
            ]
        },
    ]

    prompt = make_inputs_qwen2_omni(messages, )
    return prompt


def make_audio_in_video_v2_prompt(system:str=None, audio_path:str="/home/yiyangzhe/Qwen2.5-Omni/test.wav"):
    messages = [
        get_system_prompt(system),
        {
            "role":
            "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                #{"type":"video_url","video_url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw_small.mp4"},
            ]
        },
    ]
    prompt = make_inputs_qwen2_omni(
        messages,
        use_audio_in_video=True,
    )
    return prompt


def init_omni_engine():
    thinker_engine_args = AsyncEngineArgs(
        model=args.thinker_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.thinker_gpu_memory_utilization,
        tensor_parallel_size=len(args.thinker_devices),
        enforce_eager=args.enforce_eager or args.thinker_enforce_eager,
        distributed_executor_backend="mp",
        limit_mm_per_prompt={
            'audio': 32,
            'image': 960,
            'video': 32
        },
        max_model_len=32768,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        quantization=args.thinker_quantization,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    talker_engine_args = AsyncEngineArgs(
        model=args.talker_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.talker_gpu_memory_utilization,
        tensor_parallel_size=1,#必须能被head_num 28整除，必须比GPU数量少
        enforce_eager=args.enforce_eager or args.talker_enforce_eager,
        distributed_executor_backend="mp",
        limit_mm_per_prompt={
            'audio': 32,
            'image': 960,
            'video': 32
        },
        max_model_len=32768,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        quantization=args.talker_quantization,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    if args.thinker_only:
        return OmniLLMEngine(
            thinker_engine_args,
            thinker_visible_devices=args.thinker_devices,
        )
    elif not args.do_wave:
        return OmniLLMEngine(
            thinker_engine_args,
            talker_engine_args,
            thinker_visible_devices=args.thinker_devices,
            talker_visible_devices=args.talker_devices,
        )
    else:
        return OmniLLMEngine(
            thinker_engine_args,
            talker_engine_args,
            args.code2wav_model,
            code2wav_enable_torch_compile=args.enable_torch_compile,
            code2wav_enable_torch_compile_first_chunk=args.
            enable_torch_compile_first_chunk,
            code2wav_odeint_method=args.odeint_method,
            code2wav_odeint_method_relaxed=args.odeint_method_relaxed,
            code2wav_batched_chunk=args.batched_chunk,
            code2wav_frequency=args.code2wav_frequency,
            thinker_visible_devices=args.thinker_devices,
            talker_visible_devices=args.talker_devices,
            code2wav_visible_devices=args.code2wav_devices,
            code2wav_dynamic_batch=args.code2wav_dynamic_batch,
            code2wav_steps=args.code2wav_steps
        )


def make_omni_prompt(messages:list=None,system:str=None,audio_path="/home/yiyangzhe/Qwen2.5-Omni/test.wav") -> Union[TokensPrompt, List[TokensPrompt]]:
    if args.prompt == 'text':
        prompt = make_text_prompt(system)
    elif args.prompt == 'audio-in-video-v2':
        if messages:
            prompt = make_inputs_qwen2_omni(
                messages,
                use_audio_in_video=True,
            )
        else:
            prompt = make_audio_in_video_v2_prompt(system=system,audio_path=audio_path)
    else:
        raise ValueError(f'Unsupported prompt type: {args.prompt}')
    return prompt


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

async def process_response(
    i: int,
    output_queue: queue.Queue[Union[RequestOutput, np.ndarray]],
    modalities = ["audio"],
    request:Request = None,
    request_id:str = None
):
    #last_output: RequestOutput = None
    #waveforms: List[np.ndarray] = []

    j=0
    is_special = False
    while True:
        output = output_queue.get()
        if await request.is_disconnected():
            omni.abort_request(request_id)
            #omni.abort_request(request_id[:-1]+chr(ord(request_id[-1]) + 1))
        if output is None:
            break

        if isinstance(output, RequestOutput):
            #print(output.outputs[0].token_ids)
            if len(output.outputs[0].token_ids) and output.outputs[0].token_ids[-1] == 151665:
                is_special = True
            if output.outputs[0].text:
                print(
                    f'[R-{i}][{now()}] Input: [{len(output.prompt_token_ids)}], '
                    f'Output: [{len(output.outputs[0].token_ids)}] {output.outputs[0].text}'
                )
                #yield output.outputs[0].text
                if "text" in modalities:
                    yield json.dumps({"type": "text", "data": output.outputs[0].text}) + "\n"
            else:
                print(
                    f'[R-{i}][{now()}] Input: [{len(output.prompt_token_ids)}], '
                    f'Output: [{len(output.outputs[0].token_ids)}] {output.outputs[0].token_ids}'
                )
                #yield output.outputs[0].token_ids
                if "text" in modalities:
                    yield json.dumps({"type": "text_token_ids", "data": output.outputs[0].token_ids}) + "\n"
        elif isinstance(output, tuple) and isinstance(output[0], np.ndarray):
            if is_special:
                #omni.abort_request(request_id)
                #yield ""#不能返回空或者None，解析不了
                return #直接输出，没有字段，没有返回
            output, output_tokens = output
            tmp_wav_path = os.path.join(args.output_dir,
                            f"{now()}-chunk{j}.wav") #request_id
            sf.write(tmp_wav_path, output, samplerate=args.sample_rate)
            print(f"[R-{i}][{now()}] Generated: {tmp_wav_path}")
            j += 1
            
            if "text" in modalities:
                yield json.dumps({"type": "audio", "data": (output * 32767).astype(np.int16).tobytes()}) + "\n"
            else:
                yield (output * 32767).astype(np.int16).tobytes()
        else:
            raise ValueError(f'[R-{i}] Unknown output type: {output}')

class Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
    
    def dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

class Delta:
    def __init__(self, role=None, content=None, audio=None):
        self.role = role
        self.content = content
        self.audio = audio
    
    def dict(self):
        result = {}
        if self.role is not None:
            result["role"] = self.role
        if self.content is not None:
            result["content"] = self.content
        if self.audio is not None:
            result["audio"] = self.audio
        return result

class Choice:
    def __init__(self, index, delta, finish_reason=None):
        self.index = index
        self.delta = delta
        self.finish_reason = finish_reason
    
    def dict(self):
        return {
            "index": self.index,
            "delta": self.delta.dict(),
            "finish_reason": self.finish_reason
        }

class ChatCompletionChunk:
    def __init__(self, id, created, model, choices, usage=None):
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    
    def dict(self):
        result = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.dict() for choice in self.choices]
        }
        if self.usage is not None:
            result["usage"] = self.usage.dict()
        return result

async def process_output_queue(output_queue, response_queue, request_id, model_name, include_audio, request:Request):
    """处理输出队列，将结果转换为OpenAI兼容格式"""
    try:
        text_content = ""
        token_count = 0
        j=0
        is_special = False
        while True:
            output = await asyncio.to_thread(output_queue.get)
            #if j:
            #    print(j,output)
            if await request.is_disconnected():
                omni.abort_request(request_id)
                #omni.abort_request(request_id[:-1]+chr(ord(request_id[-1]) + 1))
            if output is None:
                # 请求完成，发送用量信息
                if token_count > 0:
                    usage = Usage(
                        prompt_tokens=len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0,
                        completion_tokens=token_count,
                        total_tokens=(len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0) + token_count
                    )
                    
                    await response_queue.put({
                        "type": "usage",
                        "data": usage.dict()
                    })
                
                # 结束流
                await response_queue.put(None)
                break
            
            if hasattr(output, 'outputs') and output.outputs:
                #print(output.outputs[0].token_ids)
                if len(output.outputs[0].token_ids) and output.outputs[0].token_ids[-1] == 151665:
                    is_special = True
                # 处理文本输出
                if output.outputs[0].text:
                    new_text = output.outputs[0].text[len(text_content):]
                    text_content = output.outputs[0].text
                    print(now(),text_content)
                    if new_text:
                        token_count += len(output.outputs[0].token_ids) - token_count
                        
                        chunk = ChatCompletionChunk(
                            id=f"chatcmpl-{request_id}",
                            created=int(time.time()),
                            model=model_name,
                            choices=[
                                Choice(
                                    index=0,
                                    delta=Delta(
                                        content=new_text
                                    ),
                                    finish_reason=None
                                )
                            ]
                        )
                        
                        await response_queue.put({
                            "type": "text",
                            "data": chunk.dict()
                        })
            
            elif isinstance(output, tuple) and isinstance(output[0], np.ndarray) and include_audio:
                if is_special:
                    # 结束流
                    #omni.abort_request(request_id)
                    await response_queue.put(None)
                    break
                # 处理音频输出
                audio_data, output_tokens = output
                audio_stream = (audio_data * 32767).astype(np.int16).tobytes()
                #本地保存，看是传输的问题，还是推理的问题
                tmp_wav_path = os.path.join(args.output_dir,f"{now()}-chunk{j}.wav") #request_id
                sf.write(tmp_wav_path, audio_data, samplerate=args.sample_rate)
                j += 1
                print(f"[{now()}] Generated: {tmp_wav_path}")
                # 将音频数据编码为base64
                with io.BytesIO() as audio_io:
                    #sf.write(audio_io, audio_data, 24000, format='RAW', subtype="PCM_16")#不能以WAV打包，包含头信息，只能分段播放
                    audio_io.write(audio_stream)
                    audio_io.seek(0)
                    audio_base64 = base64.b64encode(audio_io.read()).decode('ascii')
                
                # 创建音频响应块
                chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{request_id}",
                    created=int(time.time()),
                    model=model_name,
                    choices=[
                        Choice(
                            index=0,
                            delta=Delta(
                                audio={
                                    "data": audio_base64,
                                    "transcript": text_content
                                }
                            ),
                            finish_reason=None
                        )
                    ]
                )
                
                await response_queue.put({
                    "type": "audio",
                    "data": chunk.dict()
                })
    
    except Exception as e:
        logger.exception(f"Error processing output queue: {e}")
        # 发送错误到响应队列
        await response_queue.put({
            "type": "error",
            "data": str(e)
        })
        # 结束流
        await response_queue.put(None)

async def stream_response(queue):
    """流式响应生成器"""
    try:
        while True:
            item = await queue.get()
            if item is None:
                # 流结束
                yield "data: [DONE]\n\n"
                break
            
            if item["type"] == "error":
                # 错误信息
                yield f"data: {json.dumps({'error': item['data']})}\n\n"
                break
            
            # 正常数据
            yield f"data: {json.dumps(item['data'])}\n\n"
    
    except Exception as e:
        logger.exception(f"Error in stream response: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        
def run_omni_engine(
    prompt: Union[TokensPrompt, List[TokensPrompt]],
    omni: OmniLLMEngine,
    num_prompts: int = 1,
    is_warmup: bool = False,
    modalities:list = ["audio"],
    openai:bool = False,
    request: Request = None,
    request_id:str = None,
    is_first:bool = False,
    voice_type:str = "m02"
):
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        repetition_penalty=1.1,
        max_tokens=args.max_tokens,
        detokenize=True,
        seed=args.seed,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_k=40,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,#2048
        detokenize=False,
        seed=args.seed,
    )
    #if not is_first and request_id:#只要不传request_id，这部分就永远不起作用
        #print('before',prompt)
        #prompt['prompt'] = prompt['prompt'][:-33] #如果中途停掉，想要做续写，把apply_chat_template加在最后的special_token去掉
        #print('after',prompt)
    if not isinstance(prompt, list):
        prompts = [copy.deepcopy(prompt) for _ in range(num_prompts)]
    else:
        prompts = prompt
    
    # add request
    output_queues = []
    for i, prompt in enumerate(prompts):
        request_id = request_id if request_id else str(uuid.uuid4())
        logger.info(f'[R-{i}][{now()}] Adding request {request_id}')
        output_queue = omni.add_request(#这个是同步队列
            request_id,
            prompt,
            sampling_params,
            **{
                "talker_params": talker_sampling_params,
            } if not args.thinker_only else {},
            voice_type=voice_type
            if voice_type else args.voice_type,
            #is_first=is_first
        )
        logger.info(f'[R-{i}][{now()}] Added request {request_id}')
        output_queues.append(output_queue)
        
        if openai:
            # 创建用于流式响应的队列
            response_queue = asyncio.Queue()
            # 启动后台任务处理输出队列
            asyncio.create_task(process_output_queue(
                output_queue, 
                response_queue, 
                request_id, 
                "qwen-omni",
                "audio" in modalities and not args.text_only,
                request
            ))
            
            # 返回流式响应
            return StreamingResponse(
                stream_response(response_queue),
                media_type="text/event-stream"
            )
        if "text" in modalities:
            return StreamingResponse(
                        process_response(i,output_queue,modalities,request,request_id),
                        #media_type="application/octet-stream",#控制格式，输出音频。
                        media_type="application/x-ndjson",#json.dumps({"": ""})，就可以返回不同格式数据（文本，音频等）
                    )            
        else:
            return StreamingResponse(
                        process_response(i,output_queue,modalities,request,request_id),
                        media_type="application/octet-stream",#控制格式，输出音频。
                        #media_type="application/x-ndjson",#json.dumps({"": ""})，就可以返回不同格式数据（文本，音频等）
                    )


def parse_voice_type(voice_type) -> str:
    if not voice_type:
        return voice_type

    voice_types = {
        "晨煦": "m02",
        "ethan": "m02",
        "千雪": "f030",
        "chelsie": "f030",
    }
    voice_type = voice_type.lower()
    return voice_types.get(voice_type, voice_type)

args.voice_type = parse_voice_type(args.voice_type)
args.warmup_voice_type = parse_voice_type(args.warmup_voice_type)

@app.post("/")
async def chat_completions(request: Request):
    """兼容OpenAI格式的聊天补全API"""
    try:
        # 解析请求体
        body = await request.json()
        # 获取必需字段
        audio_path = body.get("audio_path", None)
        system = body.get("system", None)
        messages = body.get("messages",None)
        #print(type(messages),messages)
        modalities = body.get("modalities",["audio"])
        openai = body.get("openai",False)
        request_id = body.get("request_id",None)
        is_first = body.get("is_first",False)
        voice_type = body.get("voice_type","m02")
        # 准备提示
        try:
            #start = time.time()
            prompt = make_omni_prompt(messages=messages, system=system, audio_path=audio_path)
            #print('make prompt costs', time.time()-start)
            # 返回流式响应
            return run_omni_engine(prompt, omni, args.num_prompts, is_warmup=False, modalities=modalities, openai=openai, request=request, request_id=request_id,is_first=is_first, voice_type=voice_type)
        except Exception as e:
            logger.exception('Error {e} in run_omni_engine')
        
    except Exception as e:
        logger.exception(f"解析请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/completions")
async def chat_completions(request: Request):
    """兼容OpenAI格式的聊天补全API"""
    try:
        # 解析请求体
        body = await request.json()
        # 获取必需字段
        audio_path = body.get("audio_path", None)
        system = body.get("system", None)
        messages = body.get("messages",None)
        #print(type(messages),messages)
        modalities = body.get("modalities",["text","audio"])
        openai = body.get("openai",True)
        request_id = body.get("request_id",None)
        is_first = body.get("is_first",False)
        voice_type = body.get("voice_type","m02")
        # 准备提示
        try:
            #start = time.time()
            prompt = make_omni_prompt(messages=messages, system=system, audio_path=audio_path)
            #print('make prompt costs', time.time()-start)
            # 返回流式响应
            return run_omni_engine(prompt, omni, args.num_prompts, is_warmup=False, modalities=modalities, openai=openai, request=request, request_id=request_id, is_first=is_first, voice_type=voice_type)
        except Exception as e:
            logger.exception('Error {e} in run_omni_engine')
        
    except Exception as e:
        logger.exception(f"解析请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def chat_completions(messages:list=None, audio_path:str=None, system:str=None, modalities:list=["audio"], openai:bool=False, request_id:str=None, is_first:bool=False, voice_type:str="m02", media_type:str="raw"):
    """兼容OpenAI格式的聊天补全API"""
    try:             
        # 准备提示
        try:
            prompt = make_omni_prompt(messages=messages, system=system, audio_path=audio_path)
            # 返回流式响应
            return run_omni_engine(prompt, omni, args.num_prompts, is_warmup=False, modalities=modalities, openai=openai, request_id=request_id, is_first=is_first, voice_type=voice_type)
        except Exception as e:
            logger.exception('Error {e} in run_omni_engine')
    except:
        logger.exception('Error {e} in run_omni_engine')

def main():
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
