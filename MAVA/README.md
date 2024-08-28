## Overview

## Project Structure
The folder structure should be organized as follows before training.
```plaintext
MAVA
├── config
│   └── Config.yaml
├── data
│   └── processed
├── eval
├── serve
├── mava
│   ├── compression
│   ├── encoding
│   ├── conversation
│   ├── retrieval
│   ├── storage
```

## How to Run Demo Locally

Firstly, set the `compression`, `encoding`, `retrieval`, `storage` and `conversation` in [config/Config.yaml](./config/Config.yaml).
Then run the script:
```
python inference.py \
    --config config/Config.yaml \
    --query "What is this video about?" \
    --prompt "You are an assistant to help with the problem." \
    --video_path needle_0.mp4 \
    --interval 4 \
    --compression_rate 5 \
    --retrieval_number 5
```

## Zero-shot Evaluation

The previous eval code is no longer usable, and the new eval files are yet to be developed.