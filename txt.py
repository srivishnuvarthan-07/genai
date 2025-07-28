from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Input text (you can modify this for any task)
input_text = input()

# Tokenize input
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)

# Decode and print result
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Output:", output_text)
