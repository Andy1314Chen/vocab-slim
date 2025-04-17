# VocabSlim

[![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)](#) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-red)](#) 

**A package to reduce the size of 🤗 Hugging Face models via Vocabulary Slimming.**

Large language models (LLMs) increasingly adopt extensive vocabularies to better handle diverse tasks such as multilingual processing, code generation, and function-calling. Recent research highlights that expanding both model and vocabulary size enhances performance, following scaling laws with vocabulary.

However, larger vocabularies come with significant drawbacks. For smaller LLMs designed for specific tasks like English or Chinese chat applications on mobile devices, a large vocabulary can be inefficient. It consumes computational resources as the model must predict token probabilities at each decoding step. Additionally, token embeddings occupy memory and lead to larger activations during inference and fine-tuning, exacerbating the issue for smaller models(e.g., 26.53% for Qwen2.5-0.5B).

For instance, the Qwen2.5 series illustrates this challenge:

| Model                                | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B | Qwen2.5-72B |
| ------------------------------------ | ------------ | ------------ | ---------- | ---------- | ----------- | ----------- | ----------- |
| Number of Parameters                 | 0.49B        | 1.54B        | 3.09B      | 7.61B      | 14.7B       | 32.5B       | 72.7B       |
| Number of Parameters (Non-Embedding) | 0.36B        | 1.31B        | 2.77B      | 6.53B      | 13.1B       | 31.0B       | 70.0B       |
| Ratio (Embeding Parameters)          | 26.53%       | 14.93%       | 10.36%     | 14.19%     | 10.88%      | 4.61%       | 3.71%       |


Reducing the vocabulary can significantly shrink the model size and enhance memory efficiency for both inference and fine-tuning. This benefit is particularly pronounced when models are quantized, as most quantization methods avoid quantizing token embeddings and the language modeling head to preserve accuracy.



## Installation

### From PyPI (recommended)
```bash
pip install vocab-slim
```

### From source
```bash
git clone https://github.com/Andy1314Chen/vocab-slim.git
cd vocab-slim
pip install -r requirements.txt
pip install -e .
```


## Usage


### Command Line Interface
```bash
vocab-slim --model_name_or_path Qwen/Qwen2.5-0.5B --vocab_size 32000
```

### Python API
```python
from vocabslim import VocabSlim

# load pretrained tokenizer and model, train new BPE tokenizer
vocSlim = VocabSlim(model_name_or_path="Qwen/Qwen2.5-0.5B",
                    save_path=f"Qwen2.5-0.5B-Vocab-Slimed-32K",
                    dataset_config={"name": "wikitext",
                                "config": "wikitext-103-raw-v1",
                                "split": "train",
                                "text_column": "text"},
                    target_vocab_size=32_000)

# vocabulary slimming and adjust embedding weight
vocSlim.prune()

# compare outputs between original and slimmed models with test text
vocSlim.check("What is the capital of France?")

# slimmed model and tokenizer are saved in save_path
```


### Evaluation

To evaluate the performance after vocabulary slimming, we used [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to compare on several common datasets.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

model_path = "./Qwen2.5-0.5B-Vocab-Slimed-32K"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)

results = simple_evaluate(model=lm, tasks=["arc_challenge", "hellaswag", "piqa"])
results = {key: value for key, value in results.items() if key == "results"}
print("Accuracy:", results)
```

### Model Size Reduction
| Model                      | Vocabulary Size | #params            | Accuracy(arc_challenge,hellaswag,piqa) | Model Size(GB) |
| -------------------------- | --------------- | ------------------ | -------------------------------------- | -------------- |
| Qwen2.5-0.5B               | 151,936         | 494.03M            | 0.2952,0.4062,0.7024                   | 0.94           |
| Qwen2.5-0.5B-Slimed-32K    | 32,000          | 386.57M (-21.75%)  | 0.2688,0.3658,0.6099                   | 0.73           |
| Qwen2.5-0.5B-Slimed-32K-FT | 32,000          | 386.57M (-21.75%)  | 0.2798,0.3915,0.6507                   | 0.73           |
| SmolLM2-360M               | 49,152          | 361.82M            | 0.3652,0.4316,0.7214                   | 0.69           |
| SmolLM2-360M-Slimed-32K    | 32,000          | 345.36M (-4.56%)   | 0.2824,0.2728,0.5120                   | 0.66           |
| SmolLM2-360M-Slimed-32K-FT | 32,000          | 345.36M (-4.56%)   | 0.3370,0.4055,0.6610                   | 0.66           |
| Llama-3.2-1B               | 128,256         | 1235.81M           | 0.3131,0.4772,0.7448                   | 2.40           |
| Llama-3.2-1B-Slimed-32K    | 32,000          | 1038.68M (-15.95%) | 0.3020,0.4234,0.6453                   | 2.00           |
| Llama-3.2-1B-Slimed-32K-FT | 32,000          | 1038.68M (-15.95%) | 0.3063,0.4603,0.6904                   | 2.00           |



## Limitations
- Only Hugging Face models with BPE tokenizer are supported.
- May cause performance degradation due to reduced vocabulary size.
- Currently only supports causal language models

## References

### Papers
1. **Efficient Vocabulary Reduction for Small Language Models**
   - Authors: Yuta Nozaki, Dai Nakashima, Ryo Sato, Naoki Asaba, Shintaro Kawamura
   - Published in: Proceedings of the 31st International Conference on Computational Linguistics: Industry Track 2025
   - [[Paper]](https://arxiv.org/abs/2305.15020)

2. **Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies**
   - Authors: Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong
   - Published in: NeurIPS 2024
   - [[Paper]](https://arxiv.org/abs/2407.13623)

3. **Efficient Multilingual Language Model Compression through Vocabulary Trimming**
   - Authors: Asahi Ushio, Yi Zhou, Jose Camacho-Collados
   - Published in: EMNLP 2023
   - [[Paper]](https://arxiv.org/abs/2305.15020)

### Related Projects
1. **Introducing Minivoc: Faster and Memory-Efficient LLMs Through Vocabulary Reduction [WIP]**
   - [[Link]](https://kaitchup.substack.com/p/introducing-minivoc-faster-and-memory-llms)

## Citation

If you use this software, please cite it as given below;
```bibtex
@software{vocab-slim,
author = {Andy1314Chen},
license = {Apache-2.0},
title = {{vocab-slim: A Tool for Hugging Face Model Vocabulary Slimming}}
url = {https://github.com/Andy1314Chen/vocab-slim}
}
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.