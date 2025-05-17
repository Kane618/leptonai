import os
import io
import time
import base64
import requests
from PIL import Image

def encode_image(image_path):
    with Image.open(image_path).convert("RGB") as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Initialize session
session = requests.Session() 

# Get API token and endpoint URL
api_token = os.environ.get('LEPTON_API_TOKEN')
base_url = "endpoint_url"
run_url = f"{base_url}/run"
headers = {
    "Authorization": f"Bearer {api_token}"
}

# Encode image to base64, image url is also supported
img_b64 = encode_image("path_to_image")

# Prepare params
params = {
    "image": f"data:image/png;base64,{img_b64}", # or "image": image_url
    "prompt": "Use the left hand to pick up pink dragonfruit from blue plate to pale turquoise plate."
}

# Run the task
response = session.post(run_url, json=params, headers=headers)
if response.status_code != 200:
    raise Exception(f"Failed to get task id: {response.status_code}")
else:
    task_id = response.json()
    while True:
        task_url = f"{base_url}/task"
        response = session.get(task_url, params={"task_id": task_id}, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get task status: {response.status_code}")
        else:
            ret_dict = response.json()
            print(f"{ret_dict=}")
            # Get the task status
            status = ret_dict['status']
            if status == "CREATED":
                print(f"Task {task_id} is created")
            elif status == "RUNNING":
                print(f"Task {task_id} is running")
            elif status == "FAILED":
                raise Exception(f"Task {task_id} failed")
            elif status == "SUCCESS":
                print(f"Task {task_id} is completed")
                result = ret_dict["result"]
                # Get the video and dataset URLs
                video_url = result['url']
                dataset_url = result['dataset_url']
                print(f"{video_url=}")
                print(f"{dataset_url=}")
                break
            else:
                raise Exception(f"Unknown status: {status}")
            # Wait for 20 seconds before checking again
            time.sleep(20)

