import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download('punkt')
nltk.download('punkt_tab')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "vennify/t5-base-grammar-correction"
print("Loading model... This may take a minute.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
print("Ready!\n")

def correct_sentence(sentence):
    try:
        input_text = "grammar: " + sentence.strip()
        input_ids = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error: {e}")
        return sentence

def correct_grammar(text):
    sentences = nltk.sent_tokenize(text.strip())
    corrected = []
    for i, s in enumerate(sentences):
        corrected.append(correct_sentence(s))
        print(f"  [{i+1}/{len(sentences)}] processed")
    return " ".join(corrected)

print("Type a sentence or paragraph to correct it.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Bye!")
        break
    if not user_input:
        continue
    print(f"Corrected: {correct_grammar(user_input)}\n")