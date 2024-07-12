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

### Transformer flow

#### Input Text Processing
1. An input text is given
2. Text is tokenized and converted to token based on the vocabulary
3. Each token is represented by the index in the Vocabulary
4. Each Token has an embedding representation

```
input_text = "The cat sat on the mat"
input_tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
input_tokens_id = [454, 8793, 1294, 9892, 22243, 3429, 0]
input_embedding_list = [vocab[idx] for idx in input_tokens_id]
input_embeddings = np.array(input_embedding_list)
```



#### Embedding input to Transformer
1. Each Token Embedding is transformed by $M_K$, $M_Q$ and $M_V$ respectively
2. We get a list of embeddings as $K$, $Q$ and $V$
3. This is fed into the Self Attention Module to get the Attention Head Embedding

```

```
