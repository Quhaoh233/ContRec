# ContRec
This is an early access release of the code and data for the paper "Diffusion Generative Recommendation with Continuous Tokens."


## Setup Environment

You can install the useful wheels by:
```shell
pip install -r requirements.txt
```

Please download the any open-source LLM, such as LLaMA-3.2-1B-Instruct, to the path: "ContRec/code/huggingface_path/".


### A simple Example of Implementation
All code for a complete implementation of ContRec (including finetuning, validation, and testing) is included in the "code" folder. 

**Please change your path to the "code" folder.**
```shell
cd code
```
**Also replace input a correct "hf_token" in Line 228 of the "model_interface.py" file**
```shell
hf_token = "Your_HF_TOKEN"
```

**We can run the "main.py" file to fintune a Llama model for EV charging data prediction**:
```shell
python main.py
```

More configurations can be found in the "parse.py" file.
