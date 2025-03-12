import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None


def load_and_test_model_streaming(model_dir="./merged_sft_model", max_new_tokens=1024):
    # Load the tokenizer and model (using eager attention as recommended)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # attn_implementation="eager"  # Use eager attention for stable training/inference
        # attn_implementation="flash_attention_2",
    )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    # Define a sample prompt that mimics your training format (but uses different content)
    query = "Ancient Roman architecture"
    initial_results = (
        "### Text snippet from document at k=1\n"
        ": The Roman Forum remains as a witness to the political and social life of ancient Rome.\n\n"
        "### Text snippet from document at k=3\n"
        ": The Colosseum is celebrated for its grand design and remarkable engineering techniques.\n\n"
        "### Text snippet from document at k=5\n"
        ": Innovations such as arches and concrete construction revolutionized Roman building practices."
    )

    instruction = (
        f"Original Search Query: {query}\n\n"
        f"Initial Results Snippets:\n{initial_results}\n\n"
        "Please analyze these initial results and brainstorm an augmented query to improve retrieval."
    )
    prompt = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        f"{tokenizer.bos_token}<start_of_turn>user\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set up the streamer to print tokens as they are generated
    streamer = TextStreamer(tokenizer)

    # Generate the output while streaming it in real-time
    model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        streamer=streamer,
    )


if __name__ == "__main__":
    load_and_test_model_streaming()
