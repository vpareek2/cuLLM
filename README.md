# cuLlama

Work in progress Llama 3.1 implementation in pure cuda.

Gameplan:
Thank you for providing that summary. I'll refine it based on the information in the paper to more accurately reflect what we know about Llama 3's architecture:

The Llama 3 architecture builds on previous Llama models, using a standard dense Transformer architecture with some modifications. Here are the key components and characteristics:

### 1. Input Embedding Layer
- Uses a vocabulary of 128K tokens, combining 100K tokens from the tiktoken tokenizer with 28K additional tokens to better support non-English languages.

### 2. Multi-Head Self-Attention (MHSA)
- Employs Generalized Query Attention (GQA) with 8 key-value heads for increased efficiency.

### 3. Feed-Forward Network (FFN)
- Uses SwiGLU activation function.

### 4. Positional Encoding
- Utilizes Rotary Positional Embedding (RoPE) with a base frequency hyperparameter of 500,000.

### 5. Model Sizes
- Llama 3 comes in different sizes: 8B, 70B, and 405B parameters.
- The 405B model has 126 layers, a token representation dimension of 16,384, and 128 attention heads.

### 6. Context Length
- Supports context lengths up to 128K tokens.

### 7. Attention Mask
- Implements an attention mask that prevents self-attention between different documents within the same sequence.

### 8. Other Characteristics
- Does not use mixture-of-experts architectures.
- Focuses on scaling up standard dense Transformer architecture rather than introducing complex architectural changes.

### 9. Training and Optimization
- Uses AdamW optimizer with a cosine learning rate schedule.
- Implements various parallelism techniques (tensor, pipeline, context, and data parallelism) for efficient training at scale.

It's important to note that Llama 3 does not include some of the components you mentioned in your original summary, such as specific memory layers or parameter-efficient fine-tuning techniques like LoRA. The paper emphasizes that Llama 3's improvements come primarily from increased scale, better data quality and diversity, and optimized training processes rather than architectural innovations.