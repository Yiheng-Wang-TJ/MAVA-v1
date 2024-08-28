import gradio as gr
import os
import tempfile
from serve.gradio_utils import title_markdown, block_css
import numpy as np
import cv2
import base64
import time
from openai import OpenAI
import shutil
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from pytube import YouTube
from sentence_transformers import SentenceTransformer, util
import logging
import re
from pathlib import Path
import chromadb

def save_video_to_local(video_path):
    filename = os.path.join(tempfile._get_default_tempdir(), video_path.split("/")[-1])
    shutil.copyfile(video_path, filename)
    return filename
    
def save_image_to_local(image):
    filename = os.path.join(tempfile._get_default_tempdir(), 'gradio/'+image.split("/")[-1])
    image = Image.open(image)
    image.save(filename)
    return filename
    
def process_video(video):
    textbox_out = ''
    if video.split("/")[-1] not in stored_video.keys():
        video_key = video.split("/")[-1].replace(".", "")
        PICS_FRA = Path(f'examples/{video_key}')
        PICS_FRA.mkdir(parents=True, exist_ok=True)
        del_list = os.listdir(PICS_DIR )
        for f in del_list:
            file_path = os.path.join(PICS_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        base64Frames = []
        frame_index = 0
        video_cv = cv2.VideoCapture(video)
        start = time.perf_counter()
        selected_indices = []
        while video_cv.isOpened():
            success, frame = video_cv.read()
            if not success:
                break
            if frame_index % 24 == 0:
                save_path = "{}/{:>06d}.png".format(PICS_DIR, frame_index)
                cv2.imwrite(save_path, frame)
                save_path = "{}/{:>06d}.png".format(PICS_FRA, frame_index)
                cv2.imwrite(save_path, frame)
                _, buffer = cv2.imencode(".png", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                selected_indices.append(int(frame_index/24))
            frame_index += 1
        video_cv.release()
        end = time.perf_counter()
        print('frame capture执行时间: ', end - start)
        logging.info('frame capture cost:%s s', end - start)
        print(len(base64Frames), "")
        logging.info('%s frames read.', len(base64Frames))
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        transforms.Resize(size=(224, 224))])
        dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        start = time.perf_counter()
        count = 0
        for images, labels in dataloader:
            images = images.to(device)
            batch_features = model(images)
            if count == 0:
                frames_features_b = batch_features
                count += 1
            else:
                frames_features_b = torch.cat([frames_features_b, batch_features], dim=0)
        end = time.perf_counter()
        print('resnet cost: ', (end - start) * 1000)
        logging.info('resnet cost:%s ms', (end - start) * 1000)
        start = time.perf_counter()
        end = time.perf_counter()
        print('video summarization cost: ', end - start)
        logging.info('video summarization cost:%s s', end - start)
        selected_indices.sort()
        img_names = []
        for frame_index in selected_indices:
            img_names.append("{}/{:>06d}.png".format(PICS_DIR , frame_index * 24))
        start = time.perf_counter()
        img_emb = model_clip.encode([Image.open(filepath) for filepath in img_names], batch_size=32,
                                    convert_to_tensor=True, show_progress_bar=True)
        end = time.perf_counter()
        print('clip cost: ', end - start)
        logging.info('clip cost:%s s', end - start)
        selected_frames = [base64Frames[i] for i in selected_indices]
        stored_video[video.split("/")[-1]] = [selected_frames, img_emb, 0]

def process_prevideo():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    prevideo =["ego4d1.mp4","ego4d2.mp4","ego4d3.mp4"]
    for p_v in prevideo:
        video = f"{cur_dir}/examples/{p_v}"
        process_video(video)
        
def generate(video, textbox_in, request:gr.Request):
    process_video(video)
    video_key = video.split("/")[-1].replace(".", "")
    query_emb = model_clip.encode([textbox_in], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, stored_video[video.split("/")[-1]][1],
                                top_k=10 if len(stored_video[video.split("/")[-1]][0]) < 300 else 30)[0]
    select_index = []
    logging.info('hits:%s', hits)
    for hit in hits:
        select_index.append(hit['corpus_id'])
    select_index.sort()
    print(select_index)
    logging.info('select_index:%s', select_index)
    select_frames = [stored_video[video.split("/")[-1]][0][index] for index in select_index]
    data_uris = [
        f"data:image/jpeg;base64,{frame}" for frame in select_frames
    ]
    image_dicts = [
        {
            "type": "image_url",
            "image_url": {
                "url": data_uri,
                "detail": "low"
            }
        }
        for data_uri in data_uris
    ]
    print(video)
    logging.info('video:%s', video)
    filename = save_video_to_local(video)
    print(filename)
    logging.info('video:%s', filename)
    if video not in last_video:
        collection = client_db.get_or_create_collection(name="tizi365")
        if collection.count():
            client_db.delete_collection(name="tizi365")
    collection = client_db.get_or_create_collection(name="tizi365")
    query_embedding = client.embeddings.create(model='text-embedding-ada-002', input=textbox_in).data[0].embedding
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    query_num = len(query_result.get('documents')[0])
    query_content = ""
    for i in range(0, query_num):
        query_content += query_result.get('documents')[0][i]
    show_images = f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={video}"></video>'
    key = request.client.host
    if not type(request.headers).__name__ == "Obj":
        x_forwarded_for = request.headers.get('x-forwarded-for')
        if x_forwarded_for:
            key = x_forwarded_for
    print(key)
    logging.info('key:%s', key)
    if key not in history.keys():
        history[key] = []
    if len(history[key]) > 0 and video.split("/")[-1] not in history[key][0]:
        history.pop(key)
        history[key] = []
    history[key].append(textbox_in + "\n")
    logging.info('last_video:%s', last_video)
    if "video" not in history[key][0] or video not in last_video:
        history[key][0] += show_images
        last_video[0] = video
    start = time.perf_counter()
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "You are my AI assistant to support memory augmentation. Use the following frames from a video to answer "
                "the question at the end. Remember answer the question like a real man and don't mention frames or images or AI assistant,"
                "Question:" + textbox_in + "These are frames in order from a video:",
                *image_dicts,
                "Here is the previous conversation:" + query_content,
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 512,
        "stream": True
    }
    response = client.chat.completions.create(**params)
    end = time.perf_counter()
    print('generate video description cost: ', end - start)
    logging.info('generate video description:%s s', end - start)
    partial_message = ""
    token_counter = 0
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            if token_counter == 0:
                history[key].append(" " + partial_message)
            else:
                history[key][-1] = partial_message + "\n"
            token_counter += 1
            chat = [(history[key][i], history[key][i + 1]) for i in range(0, len(history[key]) - 1, 2)]  # convert to tuples of list
            yield (chat, gr.update(value=None, interactive=True), gr.update(value=video, interactive=True))
    doucuments = "Question:" + textbox_in + "Answer:" + partial_message
    embedding = client.embeddings.create(model='text-embedding-ada-002', input=doucuments).data[0].embedding
    collection.add(ids=[str(collection.count())], embeddings=embedding, documents=doucuments)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": ["Question:" + textbox_in + "These are frames in order from a video:",
                        *image_dicts,
                        "Please provide the three most relevant frames based on the given question, with frame numbers starting from 1."
                        "Please only provide the numbers, not any other information."],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 512,
    }
    result = client.chat.completions.create(**params)
    print(result.usage)
    logging.info('usage:%s', result.usage)
    numbers = re.findall(r'\d+', result.choices[0].message.content)
    print(result.choices[0].message.content)
    logging.info('content:%s', result.choices[0].message.content)
    print(numbers)
    logging.info('numbers:%s', numbers)
    show_images = ""
    for number in numbers:
        if int(number) <= len(select_index):
            image = "{}/{:>06d}.png".format(f'examples/{video_key}', select_index[int(number) - 1] * 24)
            filename = save_image_to_local(image)
            show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    history[key][-1] = partial_message + "\n" + show_images
    chat = [(history[key][i], history[key][i + 1]) for i in range(0, len(history[key]) - 1, 2)]
    yield  (chat, gr.update(value=None, interactive=True), gr.update(value=video, interactive=True))


def download(url):
    path = VIDEO_DIR
    name = url[-10:-1] + ".mp4"
    video = f'{path}/{name}'
    chat = None
    if name not in os.listdir(path):
        yt = YouTube(url)
        try:
            start = time.perf_counter()
            yt.streams.filter(file_extension="mp4").first().download(path, filename=name)
            end = time.perf_counter()
            print('video download执行时间: ', end - start)
            logging.info('video download:%s s', end - start)
            process_video(video)
            # text=None
        except Exception as e:
            logging.error(f'An error occurred: {str(e)}')
            # text=f"An error occurred: {str(e)}"
            chat = [(f"An error occurred: {str(e)}"," ")]
    return (chat, gr.update(value=video if os.path.exists(video) else None, interactive=True), gr.update(value=None, interactive=True))

def clear_history(request:gr.Request):
    key = request.client.host
    if not type(request.headers).__name__ == "Obj":
        x_forwarded_for = request.headers.get('x-forwarded-for')
        if x_forwarded_for:
            key = x_forwarded_for
    if key in history.keys():
        history.pop(key)
    return (None, None)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
client = OpenAI(api_key = )
model_clip = SentenceTransformer('clip-ViT-B-32')

client_db = chromadb.PersistentClient(path="/data/tizi365.db")
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, 1024)
model.eval()
model.requires_grad_(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

textbox = gr.Textbox(
    show_label=False, placeholder="Enter Question and Press Send", container=False, label="Question"
)
url = gr.Textbox(
        show_label=False, placeholder="Enter YouTube URL and Press Download", container=False, label="Input URL"
    )
    
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with gr.Blocks(title='Long Form Video QA🚀', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(title_markdown)
    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Video(label="Video")
            with gr.Row():
                with gr.Column(scale=7):
                    url.render()
                with gr.Column(scale=2, min_width=50):
                    downvote_btn = gr.Button(
                        value="Download", variant="primary", interactive=True
                    )
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/ego4d1.mp4",
                        "What can this person see?",
                    ],
                    [
                        f"{cur_dir}/examples/ego4d2.mp4",
                        "Where is the person?"
                    ],
                    [
                        f"{cur_dir}/examples/ego4d3.mp4",
                        "what is this person doing?",
                    ],
                ],
                inputs=[video, textbox],
            )
            gr.Examples(
                examples=[
                    [
                        "https://www.youtube.com/watch?v=TwGqQyzhPyk",
                        "What is video mainly about?",
                    ],
                    [
                        "https://www.youtube.com/watch?v=Lds9NVpH5_g",
                        "Describe the story in the story.",
                    ],
                ],
                inputs=[url, textbox],
            )
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Video QA", bubble_full_width=True, height=750)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=True
                    )
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                # downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
    try:
        submit_btn.click(generate, [video, textbox], [chatbot, textbox, video])
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
    downvote_btn.click(download, [url], [chatbot, video, url])
    # regenerate_btn.click(regenerate, [], [chatbot]).then(
    #     generate, [image1, video, textbox],
    #     [chatbot, textbox, image1, video])
    #
    clear_btn.click(clear_history, [], [video, chatbot])


stored_video = {}
history = {}
last_video = [" "]
PICS_DIR = Path("pics_embedding/pics")
PICS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("pics_embedding")
DATA_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = Path("dt")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_TMP = Path("temp")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
process_prevideo()
del_list = os.listdir(VIDEO_DIR)
for f in del_list:
    file_path = os.path.join(VIDEO_DIR, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
demo.queue().launch(server_name="0.0.0.0", share=True)
