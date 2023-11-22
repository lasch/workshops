from fms.models import llama
from fms.models.hf import to_hf_api
from transformers import LlamaForCausalLM

save_path = "/lustre/t5/workshops/pretraining/ckpt/hf/hf_fms_hf"

raw_hf_model = LlamaForCausalLM.from_pretrained("/lustre/llama_weights/hf/7B")

converted_fms_model = llama.convert_hf_llama(raw_hf_model)

hf_model = to_hf_api(converted_fms_model)

hf_model.save_pretrained(save_path)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/lustre/llama_weights/hf/7B")
tokenizer.save_pretrained(save_path)
