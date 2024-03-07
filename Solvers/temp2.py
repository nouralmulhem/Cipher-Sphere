from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import time

start_time = time.time()

# prepare image + question
url = "./bears1.png"
image = Image.open(url)
text = "How many white bears are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
end_time = time.time()
print("Predicted answer:", model.config.id2label[idx])

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the result
print("Elapsed time: {:.4f} seconds".format(elapsed_time))
