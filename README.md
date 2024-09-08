# cuLlama

Work in progress Llama 3.1 implementation in pure cuda.

Finished the tokenizer in cpp. It's a GPT-4o (o200k_base) tokenizer. LLaMA 3.1 uses a mix of the tiktoken3 (GPT-4o) tokenizer and another one that is not specified for other languages. I implemented the GPT-4o tokenizer which is regex + byte pair encoding. It is also using the the vocabulary provided by OpenAI for Tiktoken.

1. Input Embedding Layer
   • Converts input tokens to dense vector representations

2. Positional Encoding
   - Rotary Positional Embedding (RoPE)

3. Transformer Block (repeat N times):
   a. Layer Normalization
   b. Multi-Head Self-Attention (MHSA)
      - Generalized Query Attention (GQA)
   c. Residual Connection
   d. Layer Normalization
   e. Feed-Forward Network (FFN)
      - Linear layer
      - SwiGLU activation
      - Linear layer
   f. Residual Connection

4. Final Layer Normalization
   • Normalizes the output of the last Transformer block

5. Output Layer
   - Linear layer
   - Softmax function

Key points:
- Use pre-normalization (Layer Norm before attention and FFN)
- Implement GQA in the attention mechanism
- Apply residual connections around both attention and FFN blocks
- Use SwiGLU activation in the FFN

This structure should be repeated for the desired number of layers, depending on the model size you're implementing.


cuLLM/
├── LICENSE
├── README.md
├── Makefile
├── assets/
│   └── llama3_1.png
├── includes/
│   ├── model/
│   │   ├── config.hpp           # Model configuration parameters
│   │   ├── llama_model.cuh      # Main model header (CUDA)
│   │   ├── embedding.cuh        # Input embedding layer (CUDA)
│   │   ├── positional_encoding.cuh  # RoPE implementation (CUDA)
│   │   ├── attention.cuh        # Multi-Head Attention with GQA (CUDA)
│   │   ├── feed_forward.cuh     # Feed-Forward Network with SwiGLU (CUDA)
│   │   ├── layer_norm.cuh       # Layer Normalization (CUDA)
│   │   └── output_layer.cuh     # Output layer (CUDA)
│   └── tokenizer/
│       └── tokenizer.hpp
├── src/
│   ├── main.cpp                 # Main entry point
│   ├── model/
│   │   ├── llama_model.cu       # Main model implementation
│   │   ├── embedding.cu         # Input embedding implementation
│   │   ├── positional_encoding.cu   # RoPE implementation
│   │   ├── attention.cu         # Multi-Head Attention with GQA
│   │   ├── feed_forward.cu      # Feed-Forward Network with SwiGLU
│   │   ├── layer_norm.cu        # Layer Normalization
│   │   └── output_layer.cu      # Output layer implementation
│   └── tokenizer/
│       ├── tokenizer.cpp
│       └── vocab/
│           └── o200k_base.tiktoken
└── tests/
    ├── model/
    │   ├── model_test.cu        # CUDA tests for the model
    │   ├── embedding_test.cu
    │   ├── attention_test.cu
    │   └── feed_forward_test.cu
    └── tokenizer/
        └── tokenizer_test.hpp
