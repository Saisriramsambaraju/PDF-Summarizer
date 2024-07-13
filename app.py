import argparse
from pdf2image import convert_from_path
import re
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from datasets import load_dataset
from melo.api import TTS
# import scipy

parser=argparse.ArgumentParser()
parser.add_argument('path', help='Path to the file to be extracted', type=str, default=None)
parser.add_argument('mode', help='Enter the mode',type=str,default="summarize")
parser.add_argument('--output', help='Path to the output file',default="")
args=parser.parse_args()

def extract(path):
    print('Extracting the file: ', path)
    images=convert_from_path(path)
    text=''
    for img in images:
        text+=pytesseract.image_to_string(img)
    print('Extraction complete')
    return text

def summarize(text):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return summary

def voice(text):
    model = TTS(language='EN', device="auto")
    speaker_ids = model.hps.data.spk2id
    output_path = 'en-india.wav'
    model.tts_to_file(text, speaker_ids['EN_INDIA'], output_path, speed=1.0)


def main():
    text=extract(args.path)
    if args.mode=="extract":
        pass
    elif args.mode=="summarize":
        text =summarize(text)
    elif args.mode=="voice":
        text = voice(text)
    if args.output and args.mode!= "voice":
        with open(args.output,'w') as f:
            f.write(text)
    elif args.mode!= "voice":
        print(text)

if __name__ == "__main__":
    main()