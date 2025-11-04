# ContRec
This is an early access release of the code and data for the paper "Diffusion Generative Recommendation with Continuous Tokens."

<img width="2346" height="1088" alt="66127427-d92b-4874-9b6d-582963fca85c" src="https://github.com/user-attachments/assets/d0567314-154f-4223-b327-b3184c119088" />
ContRec represents users\&items as latent vector representations using a not-quantized tokenizer and leverages the exceptional continuous-valued generation capability of diffusion models to operate within continuous spaces and generate implicit user preferences conditioned on the reasoning content of LLMs.

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
