import os
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

class ResponseFormat(BaseModel):
    name : str
    age: int
    
num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", 
          trust_remote_code=True,
          tensor_parallel_size=num_gpus,
          hf_overrides={"architectures": ["Qwen2ForCausalLM"]}, )

json_schema = ResponseFormat.model_json_schema()

guided_decoding_params = GuidedDecodingParams(json=json_schema)

sampling_params = SamplingParams(
        temperature=0.1,  
        top_p=0.9,
        max_tokens=512,
        guided_decoding=guided_decoding_params
        )

inputs = [
        "give me name and age of the person as json. \"I\'m vishva, I'm 26 years old '\"",
        "give me name and age of the person as json. \"I\'m Vimal, I'm 45 years old '\"",
        "give me name and age of the person as json. \"I\'m Ragul, I'm 31 years old '\""
    ]

outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)
