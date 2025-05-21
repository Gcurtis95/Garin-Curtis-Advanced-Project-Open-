import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import postprocess_image
#from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

import time
import numpy as np
import cv2 as cv
import NDIlib as ndi

from pythonosc import dispatcher, osc_server, udp_client
from threading import Thread
import threading


from typing import List, Any, Tuple
import json


shared_message = None
seed_message = 0

def oscprompt(address, *args):
    global shared_message
    if address == "/prompt1":
        shared_message = args[0]
        
def oscseed(address, *args):
    global seed_message
    if address == "/seed":
        seed_message = args[0]

def process_image(image_np: np.ndarray, range: Tuple[int, int] = (-1, 1)) -> Tuple[torch.Tensor, np.ndarray]:
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image.unsqueeze(0), image_np


def np2tensor(image_np: np.ndarray) -> torch.Tensor:
    height, width, _ = image_np.shape
    imgs = []
    img, _ = process_image(image_np)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear", align_corners=False
    )
    image_tensors = images.to(torch.float16)
    return image_tensors



def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

# Load config
config_data = load_config('config.json')
sd_model = config_data['sd_model']
t_index_list = config_data['t_index_list']
engine = config_data['engine']
min_batch_size = config_data['min_batch_size']
max_batch_size = config_data['max_batch_size']
ndi_name = config_data['ndi_name']
osc_out_adress = config_data['osc_out_adress']
osc_out_port = config_data['osc_out_port']
osc_in_adress = config_data['osc_in_adress']
osc_in_port = config_data['osc_in_port']
print(config_data)

analogDiffusion = "./models/Model/analogDiffusion_10Safetensors.safetensors"
ReAL_dREAM = "C:/Users/Garin/Downloads/real-dream-15.safetensors"
EoicPhotoGasm = "C:/Users/Garin/Downloads/epicphotogasm_ultimateFidelity.safetensors"
sdTurbo = "stabilityai/sd-turbo"
sd1_5 = "stable-diffusion-v1-5"

dreamLike = "dreamlike-art/dreamlike-photoreal-2.0"
sdTurbo = "stabilityai/sd-turbo"
jugganault = "C:/Users/Garin/Downloads/juggernaut_reborn.safetensors"
Lora1 = "C:/Users/Garin/Desktop/Trained_Checkpoints/pytorch_lora_weights.safetensors"
lora2 = "latent-consistency/lcm-lora-sdv1-5"
photon = "C:/Users/Garin/Downloads/photon_v1.safetensors"
Trained_2001 = "C:/Users/Garin/Desktop/Trained_Checkpoints/2001_Trained"
PolarPic = "C:/Users/Garin/Desktop/Trained_Checkpoints/PolarPic_Trained"
MyPortrait = "C:/Users/Garin/Desktop/Trained_Checkpoints/MyPortraitTrained"

# # You can load any models using diffuser's StableDiffusionPipeline
# pipe = StableDiffusionPipeline.from_pretrained(sd_model).to(
#     device=torch.device("cuda"),
#     dtype=torch.float16,
# )

frame_buffer_size = 1
SEED = 56655
#[0, 16, 32, 45]
[8, 21, 32, 37, 42, 44]

#2/3 606w 341h
#1/2 455w 170.5



stream = StreamDiffusionWrapper(
        model_id_or_path=analogDiffusion,
        use_lcm_lora =True,
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=575,
        height=320,
        warmup=10,
        acceleration="none",
        do_add_noise=True,
        mode="img2img",
        use_tiny_vae=False,
        output_type="np",
        enable_similar_image_filter=True,
        similar_image_filter_threshold=1.0,
        use_denoising_batch=True,
        seed= SEED,
        cfg_type='none')


# Wrap the pipeline in StreamDiffusion


prompt = "analog style portrait of a person"
# Prepare the stream


# NDI
ndi_find = ndi.find_create_v2()

source = ''
while True:
    if not ndi.find_wait_for_sources(ndi_find, 5000):
        print('NDI: No change to the sources found.')
        continue
    sources = ndi.find_get_current_sources(ndi_find)
    print('NDI: Network sources (%s found).' % len(sources))
    for i, s in enumerate(sources):
        print('%s. %s' % (i + 1, s.ndi_name))
        if s.ndi_name == ndi_name:
            source = s
    if source != '':
        print(f'NDI: Connected to {source.ndi_name}')
        break   

ndi_recv_create = ndi.RecvCreateV3()
ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
ndi_recv = ndi.recv_create_v3(ndi_recv_create)
ndi.recv_connect(ndi_recv, source)
ndi.find_destroy(ndi_find)
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

stream.prepare(
    prompt=prompt,
    num_inference_steps=50,
    negative_prompt = "blur, haze, naked, nude",
)




# Run the stream infinitely
try:
    while True:
        if seed_message != 0:
            SEED = seed_message
            
            
        
        if shared_message is not None:
             # Check for new prompt from OSC
             
      
            prompt = str(shared_message)

            # Process the received message within the loop as needed
            print(f"Prompt: {prompt}")
            # Reset the shared_message variable
            shared_message = None

        t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)

        if t == ndi.FRAME_TYPE_VIDEO:

            frame = np.copy(v.data)
            framergb = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            inputs = []

            inputs.append(np2tensor(framergb))

            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
            start_time = time.time()
            sampled_inputs = []
            
            
            
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()

            #stream.preprocess_im, age()
            #stream(seed=SEED)
            output_images = stream(image=input_batch, prompt=prompt,seed=SEED)
           # print("Seed is: ", SEED)
            #  = stream(
            #     input_batch.to(device=stream.device, dtype=stream.dtype)
            # ).cpu()
            
            
            
            
            
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:
                
                # output_image = stream.preprocess_image(output_image)
                # output_image = postprocess_image(output_image, output_type="np")[0]

                open_cv_image = (output_image * 255).round().astype("uint8")

                #img = cv.cvtColor(open_cv_image, cv.COLOR_RGB2RGBA)
                
                            # Convert RGB to BGR
                bgr_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2BGR)

                # Add alpha channel to make it BGRA
                bgra_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2BGRA)
                ndi.recv_free_video_v2(ndi_recv, v)

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
    ndi.recv_destroy(ndi_recv)
    ndi.send_destroy(ndi_send)
    ndi.destroy()
