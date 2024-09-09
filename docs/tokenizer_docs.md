# Tokenizer

## Description
The tokenizer used in this project is a based on the paper "Neural Machine Translation with Byte=Level Subwords" by Wang et al. It is a regex + byte-pair encoding tokenizer.

In the LLaMA 3.1 technical report, Meta says they used a vocab size of 128K. 100K tokens were trained using the TikToken tokenizer written by OpenAI, and the other 28K were trained with a different approach for multilingual accuracy. They did not provide any insights into the multilingual approach, so I used the tiktoken approach of a regex and byte-pair encoding algorithm.

I made the choice to use C++ for this tokenizer. Mainly because the rest of the implementation will be written in CUDA and C++, I thought that it would make things easier, while python would be easier to implement, it may slow things down.  OpenAI originally implemented it in Rust with Python bindings and frontend. In this tokeinzer, I used the regex expression used by the GPT-4o tokenizer, which is also referred to as o200k_base in the code. I am also using the vocabulary provided by the GPT-4o tokenizer. I would train my own, but I wanted to make this implementation as good as possible and GPT-4o seems to be one of the best tokenizers out there. I assume that because Meta made the switch from SentencePiece in LLaMA 2 to this tiktoken approach in LLaMA 3 & 3.1.

This file contains a walkthrough of the code in the tokenizer, explaining piece by piece. Before reading through this, I highly recommend watching Andrej Karpathy's ["Let's build the GPT Tokenizer"](https://youtu.be/zduSFxRajkE?si=O4DB8YQ51MRv_OtD) it's a great introduction to what tokenizers are and how they work.'

## Tokenizers
A tokenizer is a 
