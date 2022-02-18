from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    IntervalStrategy,
)
import torch

DEVICE = torch.device("cpu")
BASE_MODEL = "facebook/mbart-large-cc25"
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="en_XX",)
MODEL_FILE = "/home/pihatonttu/code/poetry_generation/models/first-line-en-mbart/"


# load fine-tuned model
#model_tuned = MBartForConditionalGeneration.from_pretrained("./models/first-line-en-mbart/pytorch_model.bin")
#model_tuned.cuda()


def get_tokenizer_and_model():

    model = MBartForConditionalGeneration.from_pretrained(MODEL_FILE)
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id["en_XX"]

    #model.resize_token_embeddings(len(tokenizer))  # is this really necessary here?
    #print("Model vocab size is {}".format(model.config.vocab_size))
    #model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device(DEVICE)))
    #model.to(DEVICE)

    return tokenizer, model


def generate_lines(keywords, model, tokenizer):
    encoded = tokenizer.encode(
        keywords, padding="max_length", max_length=32, truncation=True
    )
    encoded = torch.tensor(encoded).unsqueeze(0).to(DEVICE)

    sample_outputs = model.generate(
        encoded,
        do_sample=True,
        max_length=32,
        num_beams=5,
        repetition_penalty=5.0,
        early_stopping=True,
        num_return_sequences=5,
    )

    return get_candidates(sample_outputs)


def get_candidates(sample_outputs):
    candidates = [
        tokenizer.decode(sample_output, skip_special_tokens=True)
        for sample_output in sample_outputs
    ]
    return candidates


if __name__ == "__main__":
    tokenizer, model = get_tokenizer_and_model()
    keyword_list = ['love hate', 'love hate', 'anarchy chaos', 'loving anarchy', 'cats chaos', 'running away', 'German shepherd, cat']
    for keywords in keyword_list:
        print("KWs: {}".format(keywords))
        lines = generate_lines(keywords, model, tokenizer)
        print(lines)
