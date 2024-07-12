# Transformer from Scratch with Numpy/PyTorch

Code the Transformer from scratch using only the standard libraries of Linear Algebra function

The Transformer library consists of multiple blocks:

1. Self Attention Block
2. Multi Head Attention Block
3. Feed Forward Neural Network
4. Residual Network
5. Prediction Block

### Self Attention 

Equation 1:
$$ attn = softmax \left( \frac{K \cdot Q^T}{\sqrt{d}} \cdot V \right) $$ 


The inputs $K$, $Q$ and $V$ are the same input embeddings multiplied by $M_K$, $M_Q$ and $M_V$ respectively

Transformer flow

Input Text Processing
1. An input text is given
2. Text is tokenized and converted to token based on the vocabulary
3. Each token is represented by the index in the Vocabulary
4. Each Token has an embedding representation

Embedding input to Transformer


