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
from scipy.interpolate import CubicHermiteSpline


# Interpolate v1.1 (C) 2023 Brandon Anderson. All rights reserved.
# Tested with Automatic 1111 WebUI w/ python: 3.10.6  •  torch: 1.12.1+cu113  •  xformers: N/A  •  gradio: 3.16.2  •  commit: e33cace2
# TO DO: 
# - Random order for prompt blending (BROKE!)
# - Multiple music brackets, multiple analysis files (Unsupported)
# - Use Bezier Curve (BROKE!)


# Gradio control visibility toggler
def gr_show(**kwargs):
    updates = [{"id": control_id, "visible": visible, "__type__": "update"} for control_id, visible in kwargs.items()]
    return updates

# Interpolate 'Prompt Blending' weights
def interpolate_weights(outputs, fps, current_frame, first_output=None, second_output=None):
    
    # Determine how much work has already been done
    grow_step = current_frame / fps
    output_step = current_frame / (fps * 2)

    completed_output_steps = math.floor(output_step)
    completed_grow_steps = math.floor(grow_step)

    # Determine interpolation value
    current_step_value = grow_step % 1

    # Determine which output we're interpolating
    current_output = None
    ordered = False
    if first_output == None:
        current_output = (completed_output_steps % outputs)
        ordered = True
    else:
        current_output = first_output
    
    # Determine whether we are growing or shrinking the interpolated value
    growing = False
    if completed_grow_steps % 2 == 0:
        growing = True

    # Initialize the output
    interpolated_outputs = [0] * outputs

    #   # Interpolate the values (from 0 -> 1 if growing, else 1 -> 0)
    # Growing
    if growing:
        if ordered:  
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
        else:
            interpolated_outputs[first_output] = 1.0
            interpolated_outputs[second_output] = current_step_value
        # Shrinking
    else:
        if ordered:
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
        else:
            interpolated_outputs[first_output] = 1.0 - current_step_value
            interpolated_outputs[second_output] = 1.0
    
    return interpolated_outputs

def get_bezier_interpolated_colors(seconds_list, colors, fps, duration, current_frame, p1, p2):
    
    # Calculate the total number of frames in the video
    total_frames = int(fps * duration)

    # Calculate the timestamp for the current frame
    current_time = current_frame / fps

    # Find the index of the last beat that occurred before the current time
    last_beat_index = 0
    for i in range(len(seconds_list)):
        if seconds_list[i] <= current_time:
            last_beat_index = i

    # Find the index of the next beat after the current time
    next_beat_index = (last_beat_index + 1) % len(seconds_list)

    # Calculate the timestamp of the last and next beats
    last_beat_time = seconds_list[last_beat_index]
    next_beat_time = seconds_list[next_beat_index]

    # Calculate the fraction of time between the last and next beats
    time_fraction = (current_time - last_beat_time) / (next_beat_time - last_beat_time)

    # Use Bezier curve interpolation to get the color values at the current time
    n_colors = len(colors)
    x = np.linspace(0, 1, n_colors)
    y = np.zeros(n_colors)
    y[last_beat_index] = 1 - time_fraction
    y[next_beat_index] = time_fraction
    spline = CubicHermiteSpline(x, y, [p1, p2])
    y_interp = spline(np.linspace(0, 1, n_colors))
    y_interp = np.clip(y_interp, 0, 1)

    # Create a dictionary to hold the interpolated color values
    interpolated_colors = {}

    # Loop through each color and interpolate its value
    for i in range(n_colors):
        # Add the color and its value to the dictionary
        interpolated_colors[colors[i]] = y_interp[i]

    return interpolated_colors

def get_interpolated_colors(seconds_list, colors, fps, duration, current_frame):
    
    # Calculate the total number of frames in the video
    total_frames = int(fps * duration)

    # Calculate the timestamp for the current frame
    current_time = current_frame / fps

    # Find the index of the last beat that occurred before the current time
    last_beat_index = 0
    for i in range(len(seconds_list)):
        if seconds_list[i] <= current_time:
            last_beat_index = i

    # Find the index of the next beat after the current time
    next_beat_index = (last_beat_index + 1) % len(seconds_list)

    # Calculate the timestamp of the last and next beats
    last_beat_time = seconds_list[last_beat_index]
    next_beat_time = seconds_list[next_beat_index]

    # Calculate the fraction of time between the last and next beats
    time_fraction = (current_time - last_beat_time) / (next_beat_time - last_beat_time)

    # Create a dictionary to hold the interpolated color values
    interpolated_colors = {}

    # Loop through each color and interpolate its value
    for i in range(len(colors)):
        # Calculate the index of the color in the colors list
        color_index = i % len(colors)

        # Calculate the value of the color based on the time fraction
        if color_index == last_beat_index % len(colors):
            value = 1.0 - time_fraction
        elif color_index == next_beat_index % len(colors):
            value = time_fraction
        else:
            value = 0.0

        if value > 1.0: value = 1.0
        if value < 0.0: value = 0.0

        # Add the color and its value to the dictionary
        interpolated_colors[colors[color_index]] = value

    return interpolated_colors

def extract_nested_list(input_str, fps, current_frame, first_output=None, second_output=None):
    # First, we use regular expressions to find all text within curly braces 
    # (unless it has ♪, then ignore, as that will be handled by the music_blending code)
    brace_regex = r'{((?:(?!♪).)*)}'
    brace_matches = re.findall(brace_regex, input_str)

    result = []
    # For each match, we recursively call extract_nested_list() to handle any nested brackets
    for match in brace_matches:
        cleaned_match = match.split(':')[-1].split('@')[0].strip()
        if '{' in cleaned_match:
            result.append(extract_nested_list(cleaned_match, fps, current_frame, first_output, second_output))
        else:
            pipe_split = cleaned_match.split('|')
            cleaned_split = []
            for pipe_str in pipe_split:
                if '[' in pipe_str and ']' in pipe_str:
                    # If the string is within [], leave it intact
                    cleaned_split.append(pipe_str)
                else:
                    # Otherwise, discard anything after an at-symbol
                    at_index = pipe_str.find('@')
                    if at_index != -1:
                        cleaned_split.append(pipe_str[:at_index])
                    else:
                        cleaned_split.append(pipe_str)
            # Call interpolate_weights to get the weights for the options
            weights = interpolate_weights(len(cleaned_split), fps, current_frame, first_output, second_output)
            # Build the p.prompt
            result.append([(option + "@" + str(weight)) for option, weight in zip(cleaned_split, weights)])
    return result

def read_file(file):
    lines = []
    with open(file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines

def convert_to_float(lines):
    return [float(line) for line in lines]

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
    prompt_blending_random = None
    music_analysis_file = None
    use_bezier_curve = None
    bezier_p1 = None
    bezier_p2 = None
    slider_is_img2img = None

    def title(self):
        return "Interpolate"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):     

        # Continuous Interpolation
        continuous = gr.Checkbox(label='Continuously progress seed and subseed', value=False)
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
        self.slider_is_img2img = gr.Checkbox(label="Is Img2Img?", value=is_img2img, visible=False)

        # Show all pictures in the UI at the end of generation?
        show = gr.Checkbox(label='Show grid in UI at end of generation (Disable for many frames)', value=False)

        # Create video at the end of generation?
        create_video = gr.Checkbox(label='Create video after generating (Requires FFMPEG in your PATH)', value=False)

        # Smooth the video with minterpolate?
        smooth = gr.Checkbox(label='Smooth the video after creation with minterpolate', value=False)

        # Support Prompt Blending?
        prompt_blending = gr.Checkbox(label='Support Prompt Blending? (Requires Prompt Blending Script)', value=False)
        self.prompt_blending_random = gr.Checkbox(label='Randomize blend order?', value=False, visible=False)

        # Support Music Blending?
        music_blending = gr.Checkbox(label='Support Music Blending? (Requires Prompt Blending Script)', value=False)
        self.music_analysis_file = gr.Textbox(label='Music Analysis File', visible=False)
        self.use_bezier_curve = gr.Checkbox(label='Use Bezier Curve', value=False, visible=False)
        self.bezier_p1 = gr.Number(value=0, label='p1', visible=False)
        self.bezier_p2 = gr.Number(value=0, label='p1', visible=False)

        # Support img2img loopback?
        loopback = gr.Checkbox(label='Update the input image with the generated image (Loopback)', value=False, visible=is_img2img)

        # Bind the checkboxes to the visibility of Gradio controls
        continuous.change(fn=lambda x: gr_show(starting_seed=True, slider_seed=False, slider_seed_min=False, slider_seed_max=False, slider_subseed=False, slider_subseed_min=False, slider_subseed_max=False, slider_subseed_strength = False, slider_subseed_strength_min=False, slider_subseed_strength_max=False) if x else gr_show(starting_seed=False, slider_seed=True, slider_seed_min=True, slider_seed_max=True, slider_subseed=True, slider_subseed_min=True, slider_subseed_max=True, slider_subseed_strength = True, slider_subseed_strength_min=True, slider_subseed_strength_max=True), inputs=[continuous], outputs=[self.starting_seed, self.slider_seed, self.slider_seed_min, self.slider_seed_max, self.slider_subseed, self.slider_subseed_min, self.slider_subseed_max, self.slider_subseed_strength, self.slider_subseed_strength_min, self.slider_subseed_strength_max], show_progress=False)
        prompt_blending.change(fn=lambda x: gr_show(prompt_blending_random=True, slider_is_img2img=False) if x else gr_show(prompt_blending_random=False, slider_is_img2img=False), inputs=[prompt_blending], outputs=[self.prompt_blending_random, self.slider_is_img2img], show_progress=False)
        music_blending.change(fn=lambda x: gr_show(music_analysis_file=True, use_bezier_curver=True) if x else gr_show(music_analysis_file=False, use_bezier_curve=False), inputs=[music_blending], outputs=[self.music_analysis_file, self.use_bezier_curve], show_progress=False)
        self.use_bezier_curve.change(fn=lambda x: gr_show(bezier_p1=True, bezier_p2=True) if x else gr_show(bezier_p1=False, bezier_p2=False), inputs=[self.use_bezier_curve], outputs=[self.bezier_p1, self.bezier_p2], show_progress=False)
       
        return [continuous, self.starting_seed, self.slider_seed, self.slider_seed_min, self.slider_seed_max, self.slider_subseed, self.slider_subseed_min, self.slider_subseed_max, self.slider_subseed_strength, 
                self.slider_subseed_strength_min, self.slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
                slider_scale, slider_scale_min, slider_scale_max, slider_denoising_strength,
                slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, self.slider_is_img2img, show, create_video, smooth, 
                prompt_blending, self.prompt_blending_random, loopback, music_blending, self.music_analysis_file, self.use_bezier_curve, self.bezier_p1, self.bezier_p2]

    def run(self, p, continuous, starting_seed, slider_seed, slider_seed_min, slider_seed_max, slider_subseed, slider_subseed_min, slider_subseed_max, slider_subseed_strength, 
        slider_subseed_strength_min, slider_subseed_strength_max, slider_steps, slider_steps_min, slider_steps_max, 
        slider_scale, slider_scale_min, slider_scale_max, slider_denoising_strength,
        slider_denoising_strength_min, slider_denoising_strength_max, slider_seconds, slider_FPS, slider_is_img2img, show, create_video, smooth, 
        prompt_blending, prompt_blending_random, loopback, music_blending, music_analysis_file, use_bezier_curve, bezier_p1, bezier_p2):

        #Debug
        #print(extract_nested_list(p.prompt))
        #current_frame = 40
        #fps_values = [60, 120, 30]
        #values_list = [['man', 'woman', 'child'], ['tall', 'short'], ['blonde', 'brown', 'red', 'gray']]
        #values = [[0,0,0], [0,0], [0,0,0,0]]
        
        #for i in range(1000):        
        #    print(interpolate_weights(7, 30, i))

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

        first_output = -1
        second_output = -1
        completed_steps_last_frame = -1

        # Keep track of the original prompt
        og = p.prompt

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
                # Reset the prompt
                p.prompt = og
                # Identify all the nested lists in the input string
                nested_list_regex = r'{((?:(?!♪|\{|\}).)*)}'
                nested_lists = re.findall(nested_list_regex, p.prompt)

                # Loop over each nested list
                for nl in nested_lists:
                    print(nl)
                    # Extract the options and weights using your extract_nested_list function
                    options_and_weights = extract_nested_list("{" + nl + "}", slider_FPS, i)
                    print(options_and_weights)
                    # Build a dictionary mapping each option to its interpolated weight
                    option_weights = {}
                    for o in options_and_weights:
                        weights = interpolate_weights(len(o), slider_FPS, i)
                        for j in range(len(o)):
                           option_weights[o[j]] = weights[j]

                   # Replace the nested list with the interpolated option
                    interpolated_option = max(option_weights, key=option_weights.get)
                    p.prompt = p.prompt.replace("{" + nl + "}", "{" + '|'.join(options_and_weights[0]) + "}")

                # Print the final p.prompt string
                print(p.prompt)

            if music_blending:
                try:               
                    # Reset the prompt (so it doesn't grow like a weed)
                    p.prompt = og
                    # Load the musical analysis (a list of floats specifying where a change happens in the song (in seconds))
                    beats = convert_to_float(read_file(music_analysis_file))
                    print(beats)
                    # Load the colors we'll be interpolating
                    pattern = re.compile('♪(.*?)♪')
                    colors = pattern.findall(p.prompt)
                    # Calculate the interpolated value of our colors for this frame
                    if use_bezier_curve:
                        color_values = get_bezier_interpolated_colors(beats,colors,slider_FPS,total_frames/slider_FPS,i,bezier_p1,bezier_p2)
                    else:
                        color_values = get_interpolated_colors(beats,colors,slider_FPS,total_frames/slider_FPS,i)
                    print(color_values)

                    # For each color, add our interpolated value to the prompt
                    for k, v in color_values.items():
                        try:
                            print('old prompt:', p.prompt)
                            p.prompt = p.prompt.replace(k, str(k) + '@' + str(v))
                            print('new prompt', p.prompt)
                        except:
                            pass
                    
                    p.prompt = p.prompt.replace("♪", "")
                    print("final prompt: ", p.prompt)
                except:
                    pass
                    
           # If img2img, save first image in outputs folder
            if slider_is_img2img:      
                if i == 0:
                    images.save_image(p.init_images[0], p.outpath_samples, "", p.seed, p.prompt)
        
            # Use the model to process the image
            try:
                if not state.interrupted:
                    processed = process_images(p)
                    all_images += processed.images
                    all_prompts += processed.prompt
                    infotexts += processed.infotexts
            except Exception as e:
                break

            if loopback:
                p.init_images[0] = processed.images[0]

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
