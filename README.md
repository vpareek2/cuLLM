# cuLlama

Work in progress Llama 3.1 implementation in pure cuda.


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