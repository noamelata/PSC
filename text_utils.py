import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import zlib

device = torch.device("cuda")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
blip2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco").to(device)

def compress_text(input_text):
    input_bytes = input_text.encode('utf-8')
    return zlib.compress(input_bytes, level=zlib.Z_BEST_COMPRESSION)

def decompress_text(compressed_text):
    decompressed_bytes = zlib.decompress(compressed_text)
    return decompressed_bytes.decode('utf-8')

@torch.no_grad()
def caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = blip2.generate(**inputs, max_length=32)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text