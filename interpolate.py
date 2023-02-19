import sys
import os
import subprocess
import string
import random
import glob
import shutil
import re
import math
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

# Gradio control visibility toggler
def gr_show(**kwargs):
    updates = [{"id": control_id, "visible": visible, "__type__": "update"} for control_id, visible in kwargs.items()]
    return updates

# Interpolate 'Prompt Blending' weights
def blend_prompts(outputs, fps, current_frame):
    
    # Determine how much work has already been done
    grow_step = current_frame / fps
    output_step = current_frame / (fps * 2)

    completed_output_steps = math.floor(output_step)
    completed_grow_steps = math.floor(grow_step)

    # Determine interpolation value
    current_step_value = grow_step % 1

    # Determine which output we're interpolating
    current_output = (completed_output_steps % outputs) 
    
    # Determine whether we are growing or shrinking the interpolated value
    growing = False
    if completed_grow_steps % 2 == 0:
        growing = True

    # Initialize the output
    interpolated_outputs = [0] * outputs

    #   # Interpolate the values (from 0 -> 1 if growing, else 1 -> 0)
        # Growing
    if (growing):       
        # First output
        if (current_output == 0):
            interpolated_outputs[1] = current_step_value
            interpolated_outputs[0] = 1.0 
        # Last output   
        elif current_output == (outputs - 1):
            interpolated_outputs[0] = current_step_value
            interpolated_outputs[outputs-1] = 1.0 
        # Other outputs
        else:
            interpolated_outputs[current_output+1] = current_step_value
            interpolated_outputs[current_output] = 1.0 
    # Shrinking
    else:
        if current_output == 0:
                interpolated_outputs[0] = 1.0 - current_step_value
                interpolated_outputs[1] = 1.0
        # Last output
        elif current_output == outputs - 1:
            interpolated_outputs[outputs-1] = 1.0 - current_step_value
            interpolated_outputs[0] = 1.0
        # Other outputs
        else:
            interpolated_outputs[current_output] = 1.0 - current_step_value
            interpolated_outputs[current_output+1] = 1.0

    
    return interpolated_outputs

def extract_nested_list(input_str, fps, current_frame):
    # First, we use regular expressions to find all text within curly braces
    brace_regex = r'{(.*?)}'
    brace_matches = re.findall(brace_regex, input_str)

    result = []
    # For each match, we split the text by the | character and discard anything after a colon or at-symbol, unless it's within []
    for match in brace_matches:
        pipe_split = match.split('|')
        cleaned_split = []
        for pipe_str in pipe_split:
            if '[' in pipe_str and ']' in pipe_str:
                # If the string is within [], leave it intact
                cleaned_split.append(pipe_str)
            else:
                # Otherwise, discard anything after a colon or at-symbol
                colon_index = pipe_str.find(':')
                at_index = pipe_str.find('@')
                if colon_index != -1 and (at_index == -1 or colon_index < at_index):
                    cleaned_split.append(pipe_str[:colon_index])
                elif at_index != -1:
                    cleaned_split.append(pipe_str[:at_index])
                else:
                    cleaned_split.append(pipe_str)
        # Call blend_prompts to get the weights for the options
        weights = blend_prompts(len(cleaned_split), fps, current_frame)
        result.append([(option + "@" + str(weight)) for option, weight in zip(cleaned_split, weights)])
    return result

class Script(scripts.Script):

    # Initialize class variables
    starting_seed = None
    slider_seed = None
    slider_seed_min = None
    slider_seed_max = None
    slider_subseed = None
    slider_subseed_min = None
    slider_subseed_max = None
    slider_subseed_strength = None
    slider_subseed_strength_min = None
    slider_subseed_strength_max = None

    def title(self):
        return "Interpolate"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):     

        # Continuous Interpolation
        continuous = gr.Checkbox(label='Continuously move through latent space', value=False)
        self.starting_seed = gr.Number(label='Starting Seed', value=0, visible=False)

        # Seed  
        self.slider_seed = gr.Checkbox(label='Interpolate Seed', value=False)
        self.slider_seed_min = gr.Number(value=0)
        self.slider_seed_max = gr.Number(value=0)

        # Subseed
        self.slider_subseed = gr.Checkbox(label='Interpolate Subseed', value=False)
        self.slider_subseed_min = gr.Number(value=0)
        self.slider_subseed_max = gr.Number(value=0)

        # Subseed strength
        self.slider_subseed_strength = gr.Checkbox(label='Interpolate Subseed Strength', value=False)
        self.slider_subseed_strength_min = gr.Slider(label='Subseed Strength Min', mininum=0, maximum=1, step=0.01, value=0)
        self.slider_subseed_strength_max = gr.Slider(label='Subseed Strength Max', mininum=0, maximum=1, step=0.01, value=0)

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
        slider_seconds = gr.Number(value=0, label='Seconds')
        slider_FPS = gr.Number(value=0, label='FPS')

        # Is img2img? (Hidden, used to pass is_img2img to run())
        slider_is_img2img = gr.Checkbox(label="Is Img2Img?", value=is_img2img, visible=False)

        # Show all pictures in the UI at the end of generation?
        show = gr.Checkbox(label='Show grid in UI at end of generation (Disable for many frames)', value=False)

        # Create video at the end of generation?
        create_video = gr.Checkbox(label='Create video after generating? (Requires FFMPEG in your PATH)', value=False)

        # Smooth the video with minterpolate?
        smooth = gr.Checkbox(label='Smooth the video after creation with minterpolate?', value=False)

        # Support Prompt Blending?
        prompt_blending = gr.Checkbox(label='Support Prompt Blending? (Requires Prompt Blending Script)')
        prompt_blending_01 = gr.Number(label='Frames to cycle prompt blend 1', value=0, visible=False)
        prompt_blending_02 = gr.Number(label='Frames to cycle prompt blend 2', value=0, visible=False)

        # Bind the 'continuous' checkbox to the visibility for relevant Gradio controls
        continuous.change(fn=lambda x: gr_show(starting_seed=True, slider_seed=False, slider_seed_min=False, slider_seed_max=False, slider_subseed=False, slider_subseed_min=False, slider_subseed_max=False, slider_subseed_strength = False, slider_subseed_strength_min=False, slider_subseed_strength_max=False) if x else gr_show(starting_seed=False, slider_seed=True, slider_seed_min=True, slider_seed_max=True, slider_subseed=True, slider_subseed_min=True, slider_subseed_max=True, slider_subseed_strength = True, slider_subseed_strength_min=True, slider_subseed_strength_max=True), inputs=[continuous], outputs=[self.starting_seed, self.slider_seed, self.slider_seed_min, self.slider_seed_max, self.slider_subseed, self.slider_subseed_min, self.slider_subseed_max, self.slider_subseed_strength, self.slider_subseed_strength_min, self.slider_subseed_strength_max], show_progress=False)
       
        return [continuous, self.starting_seed, self.slider_seed, self.slider_seed_min, self.slider_seed_max, self.slider_subseed, self.slider_subseed_min, self.slider_subseed_max, self.slider_subseed_strength, 
                self.slider_subseed_strength_min, self.slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
                slider_scale, slider_scale_min, slider_scale_max, slider_denoising_strength,
                slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, slider_is_img2img, show, create_video, smooth, prompt_blending]

    def run(self, p, continuous, starting_seed, slider_seed, slider_seed_min, slider_seed_max, slider_subseed, slider_subseed_min, slider_subseed_max, slider_subseed_strength, 
        slider_subseed_strength_min, slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
        slider_scale, slider_scale_min, slider_scale_max, slider_denoising_strength,
        slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, slider_is_img2img, show, create_video, smooth, prompt_blending):

        #Debug
        #print(extract_nested_list(p.prompt))
        current_frame = 40
        fps_values = [60, 120, 30]
        values_list = [['man', 'woman', 'child'], ['tall', 'short'], ['blonde', 'brown', 'red', 'gray']]
        values = [[0,0,0], [0,0], [0,0,0,0]]
        
        #for i in range(1000):        
        #    print(blend_prompts(7, 30, i))

        #result = interpolate_values(current_frame, fps_values, extract_nested_list(p.prompt))
        #for i in range(1000):
            #result = interpolate(4, 30, i)
            #print(result)  # should output [1.0, 0.5, 0.0]

        # Total frames
        total_frames = (int)(slider_seconds * slider_FPS)

        # Initialize arrays
        all_images = []
        all_prompts = []
        infotexts = []
        files = []

        frames_drawn_this_second = 0

        # For each frame..
        for i in range(total_frames):

            if state.interrupted:
                # Interrupt button pressed in WebUI
                break

            # Initial model settings
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.batch_count = 1
            reset = False

            # Keep track of frame count                           
            frames_drawn_this_second += 1
            if (frames_drawn_this_second > slider_FPS):
                frames_drawn_this_second = 0
                reset = True 

            # Continuous Iteration
            if continuous:
                # Set the seed and subseed on the first frame
                if i == 0:
                    p.seed = starting_seed
                    p.subseed = starting_seed + 1
                # If the frame count > FPS, increment the seeds. 
                if (reset):
                    p.seed = p.subseed
                    p.subseed += 1              
                # Interpolate subseed_strength at 0 -> 1 over the course of the frames in slider_FPS
                p.subseed_strength = 0 + (1 - 0) * (frames_drawn_this_second / slider_FPS)
            else:
                # Adjust the seed using linear interpolation      
                if slider_seed:
                    p.seed = slider_seed_min + (slider_seed_max - slider_seed_min) * (i / total_frames)
                if slider_subseed:
                    p.subseed = slider_subseed_min + (slider_subseed_max - slider_subseed_min) * (i / total_frames)
                if slider_subseed_strength:
                    p.subseed_strength = slider_subseed_strength_min + (slider_subseed_strength_max - slider_subseed_strength_min) * (i / total_frames)
           
            # Adjust other parameters using linear interpolation    
            if slider_steps:
                steps = slider_steps_min + (slider_steps_max - slider_steps_min) * (i / total_frames)
                p.steps = int(steps)
            if slider_scale:
                p.cfg_scale = slider_scale_min + (slider_scale_max - slider_scale_min) * (i / total_frames)
            if slider_denoising_strength:
                p.denoising_strength = slider_denoising_strength_min + (slider_denoising_strength_max - slider_denoising_strength_min) * (i / total_frames)

            if prompt_blending:
                # Extract the nested list with weights
                nested_list = extract_nested_list(p.prompt, slider_FPS, i)

                # Construct the new prompt with weights inserted
                new_prompt = p.prompt
                brace_regex = r'{(.*?)}'
                for i, match in enumerate(re.findall(brace_regex, new_prompt)):
                    options = nested_list[i]
                    new_prompt = new_prompt.replace("{" + match + "}", "{" + "|".join(options) + "}", 1)
                print(new_prompt)
                p.prompt = new_prompt

           # If img2img, save first image in outputs folder
            if slider_is_img2img:      
                if i == 0:
                    images.save_image(p.init_images[0], p.outpath_samples, "", p.seed, p.prompt)
        
            # Use the model to process the image
            try:
                if not state.interrupted:
                    processed = process_images(p)
            except Exception as e:
                break

            processed = process_images(p)
            all_images += processed.images
            all_prompts += processed.prompt
            infotexts += processed.infotexts

        # Create video after generation (if enabled)
        # (Some code here is modified from https://github.com/memes-forever/Stable-diffusion-webui-video and related forks)
        if not state.interrupted:
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
