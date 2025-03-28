########    Reference             ###################################
#
#    This code has been adapted from https://github.com/olegchomp/StreamDiffusion-NDI
#    to send output over NDI and to receive OSC messages
#
#
#

###  Import all relevent modules 

from streamdiffusion.image_utils import postprocess_image
from utils.wrapper import StreamDiffusionWrapper
import cv2 as cv
import NDIlib as ndi
from pythonosc import dispatcher, osc_server
from typing import List, Any, Tuple
import json
import time
import threading

### Declare Global OSC messages 

shared_message = None
seed_message = 0

#### OSC functions

def oscprompt(address, *args):
    global shared_message
    if address == "/prompt1":
        shared_message = args[0]
        
def oscseed(address, *args):
    global seed_message
    if address == "/seed":
        seed_message = args[0]

def load_settings(file_path):
    with open(file_path, 'r') as file:
        settings = json.load(file)
    return settings

# Load config
settings_data = load_settings('settings.json')
sd_model = settings_data['sd_model']
Width = settings_data['width']
Height = settings_data['height']
#Use_Tiny_Vae = settings_data['use_tiny_vae']


print(settings_data)


### Model safetensors file location
analogDiffusion = "C:/Users/Garin/Downloads/analogDiffusion_10Safetensors.safetensors"


#### 512, 288 wide screen dimension for low resolution

frame_buffer_size = 1
SEED = 3466625277

####  referenced from https://github.com/cumulo-autumn/StreamDiffusion/tree/main

stream = StreamDiffusionWrapper(
        model_id_or_path=sd_model,
        use_lcm_lora =True,
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=Width,
        height=Height,
        warmup=10,
        acceleration="none",
        #vae_id = "stabilityai/sd-vae-ft-mse",
        do_add_noise=True,
        mode="txt2img",
        use_tiny_vae=False,
        output_type="pil",
        enable_similar_image_filter=True,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=True,
        seed= SEED,
        cfg_type='none')


prompt = "Analog style portrait of a person"


# NDI
send_settings = ndi.SendCreate()
send_settings.ndi_name = "USER-LAPTOP (Touchdesigner)"
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
           # print("Seed is: ", SEED)
            
        
        if shared_message is not None:
             # Check for new prompt from OSC
             
      
            prompt = str(shared_message)

            # Process the received message within the loop as needed
            print(f"Prompt: {prompt}")
            # Reset the shared_message variable
            shared_message = None
            
        # Apply Stable Diffusion

        start_time = time.time()
            
        output_images = stream(prompt=prompt, seed = SEED)
        output_images = stream.preprocess_image(output_images)
        if frame_buffer_size == 1:
            output_images = [output_images]
        for output_image in output_images:
            # Post-process and send NDI frame  
            #stream.postprocess_image(output_image, output_type="pil").show()
            output_image = stream.postprocess_image(output_image, output_type="np")

            open_cv_image = (output_image * 255).round().astype("uint8")
            
            # Convert RGB to BGR
            bgr_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2BGR)

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
