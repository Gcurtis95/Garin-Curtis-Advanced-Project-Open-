from diffusers import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import AutoPipelineForText2Image,LCMScheduler,AutoencoderKL,AutoencoderTiny
from diffusers import DPMSolverMultistepScheduler
from streamdiffusion.image_utils import postprocess_image


import numpy as np
import cv2 as cv
import NDIlib as ndi

from pythonosc import dispatcher, osc_server, udp_client
from threading import Thread
from typing import List, Any, Tuple
import json
import time
import torch
import threading

    
shared_message = None
seed_message = 0

torch.backends.cuda.matmul.allow_tf32 = True

analogDiffusion = "C:/Users/Garin/Downloads/analogDiffusion_10Safetensors.safetensors"
dreamLike = "dreamlike-art/dreamlike-photoreal-2.0"
sdTurbo = "stabilityai/sd-turbo"
sd1_5  = "c:/Users/Garin/Downloads/sd.webui/webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors"
sdXLTurbo = "stabilityai/sdxl-turbo"
lykon = "Lykon/dreamshaper-7"
jugganault = "C:/Users/Garin/Downloads/juggernaut_reborn.safetensors"
Trained_2001 = "C:/Users/Garin/Desktop/Trained_Checkpoints/2001_Trained"

Lora1 = "C:/Users/Garin/Desktop/Trained_Checkpoints/2001_Lora_Trained/pytorch_lora_weights.safetensors"
lora2 = "latent-consistency/lcm-lora-sdv1-5"
lora3 = "c:/Users/Garin/Downloads/rz884n4l0g.safetensors"

pipeline = StableDiffusionPipeline.from_single_file(
    analogDiffusion,
    torch_dtype=torch.bfloat16, 
    use_safetensors=True
).to("cuda")


### compiling torch

# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
# pipeline.unet.to(memory_format=torch.channels_last)
# pipeline.vae.to(memory_format=torch.channels_last)

# pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
# pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)






# pipeline = AutoPipelineForText2Image.from_pretrained(
#     Trained_2001, torch_dtype=torch.bfloat16, use_safetensors=True
# ).to("cuda")

#vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.bfloat16).to("cuda")

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.bfloat16).to("cuda")
pipeline.vae = vae
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

#pipeline.load_lora_weights(lora2, weight_name="pytorch_lora_weights.safetensors")

pipeline.load_lora_weights(lora2)
pipeline.fuse_lora()



def oscprompt(address, *args):
    global shared_message
    if address == "/prompt1":
        shared_message = args[0]
        
def oscseed(address, *args):
    global seed_message
    if address == "/seed":
        seed_message = args[0]


# pipeline = StableDiffusionPipeline.from_pretrained(
#     analogDiffusion, torch_dtype=torch.float16, use_safetensors=True
# ).to("cuda")

#pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

#pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

#pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

#print(pipeline)

#prompt = "analog style, photograph of young Harrison Ford as Han Solo, star wars behind the scenes" #185589077

prompt = "analog style jellyfish"

#prompt = "analog style photograph of young Harrison Ford as Han Solo, star wars behind the scenes." 3466625277

#prompt = "analog style portrait of beautiful young Zoe Saldana 1950s hollywood glamour"

#prompt = "analog style an amazing vista of 2001_S_O"
#prompt = "analog style film still of Audrey Hepburn at a neon convenience storefront"

#prompt = "analog style at night a campsite under the stars"
#prompt = "analog style portrait of cosmonaut Johnny Cash"

#prompt = "analog style portrait of Heath Ledger as a 1930s baseball player"

#prompt = "analog style, portrait of artifical intelligence as a cosmonaut"


#### good seed 32746 for 2001

#prompt = "analog style, portrait of 2001_S_O"


SEED = 3466625277




generator = torch.Generator(device="cuda").manual_seed(SEED)
image = pipeline(
    prompt, 
    generator=generator,
    negative_prompt="blur haze",
    guidance_scale=1.0,
    #strength = 0.6,
    num_inference_steps=4
    ).images[0]




frame_buffer_size = 1

# NDI
send_settings = ndi.SendCreate()
send_settings.ndi_name = 'SD-NDI'
ndi_send = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()

# OSC
def OSCMain():

    #OSC SERVER
    _dispatcher = dispatcher.Dispatcher()
    _dispatcher.map("/prompt1", oscprompt)
    _dispatcher.map("/seed", oscseed)
    global osc_server_running
    
    server = osc_server.ThreadingOSCUDPServer(
    ("127.0.0.1", 10000), _dispatcher)
    #print("Serving on {}".format(server.server_address))
    server.serve_forever()
    
    while osc_server_running:
        server.handle_request()

    print("OSC server shut down.")






osc_thread = threading.Thread(target=OSCMain)
osc_thread.start()   







# Run the stream infinitely
try:
    while True:
        
        if seed_message != 0:
            SEED = seed_message
           # print("Seed is: ", SEED)
            
        
        if shared_message is not None:
             # Check for new prompt from OSC
             
      
            prompt = str(shared_message)

            # Process the received message within the loop as needed
            print(f"Prompt: {prompt}")
            # Reset the shared_message variable
            shared_message = None
            
        start_time = time.time()
        # Apply Stable Diffusion
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        output_images = image = pipeline(
                prompt, 
                generator=generator,
                negative_prompt="blur haze",
                guidance_scale=1.0,
                #strength = 0.6,
                num_inference_steps=8
                ).images[0]
        
        if frame_buffer_size == 1:
            output_images = [output_images]
        for output_image in output_images:
            # Post-process and send NDI frame  
            #stream.postprocess_image(output_image, output_type="pil").show()
            output_image_np = np.array(output_image)

            #open_cv_image = (output_image * 255).round().astype("uint8")
            
            # Convert RGB to BGR
            bgr_image = cv.cvtColor(output_image_np, cv.COLOR_RGB2BGR)

            # Add alpha channel to make it BGRA
            bgra_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2BGRA)


            video_frame.data = bgra_image
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

            ndi.send_send_video_v2(ndi_send, video_frame)
            fps = 1 / (time.time() - start_time)

            print("FPS:", fps)

except KeyboardInterrupt:
    # Handle KeyboardInterrupt (Ctrl+C)
    print("KeyboardInterrupt: Stopping the server")
finally:
    # Stop the server when the loop exits
    ndi.send_destroy(ndi_send)
    ndi.destroy()
    osc_thread.shutdown()
    osc_thread.join()








