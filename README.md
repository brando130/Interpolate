# Interpolate
A script for Automatic 1111's Web UI which allows you to interpolate the sliders to create transitions

https://user-images.githubusercontent.com/25804985/217944337-07072b97-3223-401c-82f7-0fd311afec2a.mp4

https://user-images.githubusercontent.com/25804985/217940047-24d67139-6909-4929-b23c-44f84737e195.mp4

# Requirements
Automatic 1111's WebUI for Stable Diffusion (https://github.com/AUTOMATIC1111/stable-diffusion-webui)

Optional: ffmpeg for video creation (must be linked in your environment PATH) 

# Installation / Getting Started
Drop interpolate.py in the scripts folder of your stable-diffusion-webui installation. Restart WebUI or go to Settings -> Reload UI

Once installed, select 'Interpolate' from Scripts. 

![interpolate](https://user-images.githubusercontent.com/25804985/217945736-bd9bec3f-523d-4a51-9fc4-d9bab34d7ec7.jpg)

You can set the FPS and duration (in seconds) of the video you'd like to generate frames for, then select the checkbox for any settings you want to interpolate, and set the min and max values for the animation.

![interpolate2](https://user-images.githubusercontent.com/25804985/217946340-31ffc8f1-ccc1-4a76-bd02-e2e28033a58b.jpg)
