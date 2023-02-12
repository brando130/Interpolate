import sys
import os
import subprocess
import string
import random
import glob
import shutil
import modules
import gradio as gr
import numpy as np
import modules.scripts as scripts
from tqdm import trange
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed, process_images
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from PIL import Image


# Interpolate v1 (C) 2023 Brandon Anderson. MIT License.
# Tested with Automatic 1111 WebUI w/ python: 3.10.6  •  torch: 1.12.1+cu113  •  xformers: N/A  •  gradio: 3.16.2  •  commit: e33cace2

def generate_random_string(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class Script(scripts.Script):

    def title(self):
        return "Interpolate"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):     

        # Seed  
        slider_seed = gr.Checkbox(label='Interpolate Seed', value=False)
        slider_seed_min = gr.Number(value=0)
        slider_seed_max = gr.Number(value=0)

        # Subseed
        slider_subseed = gr.Checkbox(label='Interpolate Subseed', value=False)
        slider_subseed_min = gr.Number(value=0)
        slider_subseed_max = gr.Number(value=0)

        # Subseed strength
        slider_subseed_strength = gr.Checkbox(label='Interpolate Subseed Strength', value=False)
        slider_subseed_strength_min = gr.Slider(label='Subseed Strength Min', mininum=0, maximum=1, step=0.01, value=0)
        slider_subseed_strength_max = gr.Slider(label='Subseed Strength Max', mininum=0, maximum=1, step=0.01, value=0)

        # Steps
        slider_steps = gr.Checkbox(label='Interpolate Steps', value=False)
        slider_steps_min = gr.Slider(label='Steps Min', mininum=1, maximum=150, step=1, value=1)
        slider_steps_max = gr.Slider(label='Steps Max', mininum=1, maximum=150, step=1, value=1)

        # CFG Scale
        slider_scale = gr.Checkbox(label='Interpolate Scale', value=False)
        slider_scale_min = gr.Slider(label='Scale Min', mininum=1, maximum=30, step=1, value=1)
        slider_scale_max = gr.Slider(label='Scale Max', mininum=1, maximum=30, step=1, value=1)

        # Denoising Strength
        slider_denoising_strength = gr.Checkbox(label='Interpolate Denoising Strength', value=False, visible=is_img2img)
        slider_denoising_strength_min = gr.Slider(label='Denoising Strength Min', mininum=0.01, maximum=1, step=0.01, value=0, visible=is_img2img)
        slider_denoising_strength_max = gr.Slider(label='Denoising Strength Max', mininum=0.01, maximum=1, step=0.01, value=0, visible=is_img2img)

        # Duration/FPS
        slider_seconds = gr.Slider(label='Seconds', mininum=1, maximum=60, step=1, value=1)
        slider_FPS = gr.Slider(label='FPS', mininum=1, maximum=60, step=1, value=30)

        # Is img2img? (Hidden, used to pass is_img2img to run())
        slider_is_img2img = gr.Checkbox(label="Is Img2Img?", value=is_img2img, visible=False)

        # Show all pictures in the UI at the end of generation?
        show = gr.Checkbox(label='Show grid in UI at end of generation (Disable for many frames)', value=False)

        # Create video at the end of generation?
        create_video = gr.Checkbox(label='Create video after generating? (Requires FFMPEG in your PATH)', value=False)

        # Smooth the video with minterpolate
        smooth = gr.Checkbox(label='Smooth the video after creation with minterpolate?', value=False)

        return [slider_seed, slider_seed_min, slider_seed_max, slider_subseed, slider_subseed_min, slider_subseed_max, slider_subseed_strength, 
                slider_subseed_strength_min, slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
                slider_scale, slider_scale_min, slider_scale_max, slider_denoising_strength,
                slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, slider_is_img2img, show, create_video, smooth]

    def run(self, p, slider_seed, slider_seed_min, slider_seed_max, slider_subseed, slider_subseed_min, slider_subseed_max, slider_subseed_strength, 
        slider_subseed_strength_min, slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
        slider_scale, slider_scale_min, slider_scale_max, slider_denoising_strength,
        slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, slider_is_img2img, show, create_video, smooth):

        # Total frames
        total_frames = slider_seconds * slider_FPS

        # Initialize arrays
        all_images = []
        all_prompts = []
        infotexts = []
        files = []

        # For each frame..
        for i in range(total_frames):

            # Initial model settings
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.batch_count = 1

            # Adjust the model using linear interpolation
            if slider_seed:
                p.seed = slider_seed_min + (slider_seed_max - slider_seed_min) * (i / total_frames)
            if slider_subseed:
                p.subseed = slider_subseed_min + (slider_subseed_max - slider_subseed_min) * (i / total_frames)
            if slider_subseed_strength:
                p.subseed_strength = slider_subseed_strength_min + (slider_subseed_strength_max - slider_subseed_strength_min) * (i / total_frames)
            if slider_steps:
                steps = slider_steps_min + (slider_steps_max - slider_steps_min) * (i / total_frames)
                p.steps = int(steps)
            if slider_scale:
                p.cfg_scale = slider_scale_min + (slider_scale_max - slider_scale_min) * (i / total_frames)
            if slider_denoising_strength:
                p.denoising_strength = slider_denoising_strength_min + (slider_denoising_strength_max - slider_denoising_strength_min) * (i / total_frames)
 
           # If img2img, save first image in outputs folder
            if slider_is_img2img:      
                if i == 0:
                    images.save_image(p.init_images[0], p.outpath_samples, "", p.seed, p.prompt)
        
            # Use the model to process the image
            processed = process_images(p)
            all_images += processed.images
            all_prompts += processed.prompt
            infotexts += processed.infotexts

        # Create video after generation (if enabled)
        # (Some code here is modified from https://github.com/memes-forever/Stable-diffusion-webui-video and related forks)
        if create_video:

            fps = slider_FPS

            # Get all files in the output directory, sort by modified, and select the number equal to total_frames
            files = [os.path.join(p.outpath_samples, f) for f in os.listdir(p.outpath_samples) if f.endswith('.png')]
            files.sort(key=lambda f: os.path.getmtime(f))
            files = files[-int(total_frames):]
               # Duplicate last frame for minterpolate.
            if smooth:
                files.append(files[-1])    
           
            # Make path OS agnostic
            files = [i.replace('/', os.path.sep) for i in files]

            # Setup filename / directory for video.
            path = modules.paths.script_path
            video_name = os.path.splitext(os.path.basename(files[-1]))[0] + '.mp4'
            save_dir = os.path.join(os.path.split(os.path.abspath(p.outpath_samples))[0], 'img2img-videos')
            os.makedirs(save_dir, exist_ok=True)
            video_name = os.path.join(save_dir, video_name)

            # Save the filenames to text.
            txt_name = video_name + '.txt'
            open(txt_name, 'w').write('\n'.join(["file '" + os.path.join(path, f) + "'" for f in files]))

            # Use FFMPEG to create video.
            if smooth:
                ffmpeg_command = ["ffmpeg", "-r", str(fps), "-f", "concat", "-safe", "0", "-i", str(txt_name), "-vcodec", "libx264", "-filter:v", "minterpolate", "-crf", "10", "-pix_fmt", "yuv420p", str(video_name)]
            else:
                ffmpeg_command = ["ffmpeg", "-r", str(fps), "-f", "concat", "-safe", "0", "-i", str(txt_name), "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(video_name)]       
            subprocess.call(ffmpeg_command)

        return Processed(p, all_images if show else [], p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
