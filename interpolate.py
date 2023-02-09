import numpy as np
import sys
from tqdm import trange
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed, process_images
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
import modules
from PIL import Image

# Interpolate v1 (C) 2023 Brandon Anderson. MIT License.
# Tested with Automatic 1111 WebUI w/ python: 3.10.6  •  torch: 1.12.1+cu113  •  xformers: N/A  •  gradio: 3.16.2  •  commit: e33cace2

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
        slider_subseed_strength_min = gr.Slider(label="Subseed Strength Min", mininum=0, maximum=1, step=0.01, value=0)
        slider_subseed_strength_max = gr.Slider(label="Subseed Strength Max", mininum=0, maximum=1, step=0.01, value=0)

        # Steps
        slider_steps = gr.Checkbox(label='Interpolate Steps', value=False)
        slider_steps_min = gr.Slider(label="Steps Min", mininum=1, maximum=150, step=1, value=1)
        slider_steps_max = gr.Slider(label="Steps Max", mininum=1, maximum=150, step=1, value=1)

        # CFG Scale
        slider_cfg_scale = gr.Checkbox(label='Interpolate CFG Scale', value=False)
        slider_cfg_scale_min = gr.Slider(label="CFG Scale Min", mininum=0.1, maximum=10, step=0.1, value=1)
        slider_cfg_scale_max = gr.Slider(label="CFG Scale Max", mininum=0.1, maximum=10, step=0.1, value=1)

        # Denoising Strength
        slider_denoising_strength = gr.Checkbox(label='Interpolate Denoising Strength', value=False, visible=is_img2img)
        slider_denoising_strength_min = gr.Slider(label="Denoising Strength Min", mininum=0.01, maximum=1, step=0.01, value=0, visible=is_img2img)
        slider_denoising_strength_max = gr.Slider(label="Denoising Strength Max", mininum=0.01, maximum=1, step=0.01, value=0, visible=is_img2img)

        # Duration/FPS
        slider_seconds = gr.Slider(label="Seconds", mininum=1, maximum=60, step=1, value=1)
        slider_FPS = gr.Slider(label="FPS", mininum=1, maximum=60, step=1, value=30)

        # Is img2img? (Hidden, used to pass is_img2img to run())
        slider_is_img2img = gr.Checkbox(label="Is Img2Img?", value=is_img2img, visible=False)

        # Show all pictures in the UI at the end of generation?
        show = gr.Checkbox(label='Show grid in UI at end of generation (Disable for many frames)', value=False)

        return [slider_seed, slider_seed_min, slider_seed_max, slider_subseed, slider_subseed_min, slider_subseed_max, slider_subseed_strength, 
                slider_subseed_strength_min, slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
                slider_cfg_scale, slider_cfg_scale_min, slider_cfg_scale_max, slider_denoising_strength,
                slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, slider_is_img2img, show]

    def run(self, p, slider_seed, slider_seed_min, slider_seed_max, slider_subseed, slider_subseed_min, slider_subseed_max, slider_subseed_strength, 
        slider_subseed_strength_min, slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
        slider_cfg_scale, slider_cfg_scale_min, slider_cfg_scale_max, slider_denoising_strength,
        slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, slider_is_img2img, show):

        # Total frames
        total_frames = slider_seconds * slider_FPS

        # Initialize arrays
        all_images = []
        all_prompts = []
        infotexts = []

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
            if slider_cfg_scale:
                p.cfg_scale = slider_cfg_scale_min + (slider_cfg_scale_max - slider_cfg_scale_min) * (i / total_frames)
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

        return Processed(p, all_images if show else [], p.seed, "", all_prompts=all_prompts, infotexts=infotexts)