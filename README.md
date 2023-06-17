# How strong is reasoning of LLMs in Video Games?

## Requirements
- Python

### Installing Python dependencies
1. (recommended) Create a virtual environment
2. Install dependencies from `requirements.txt` file
    
    ```pip install -r requirements.txt```


## Getting the model
There are two models, LLama-based model fine-tuned for instruction tuning and a replication of OpenAI’s GPT-3

### High-level API for LLama-based model
We are using a high-level API for the LLama-based model. The API is downloaded with all the dependencies when running the code for the first time. The reposority is located at https://github.com/abetlen/llama-cpp-python

### Downloading the LLama-based model
1. Go to https://huggingface.co/TheBloke/wizardLM-7B-GGML/tree/main
2. Download `wizardLM-7B.ggmlv3.q4_1.bin` model.
3. Create a folder `model` (if it does not exist and place the model inside of the folder.

### Downloading the GPT-3 replication

