# Let's Build GPT

This project is based on the YouTube video "**Let's build GPT: from scratch, in code, spelled out.**" by Andrej Karpathy. It provides a step-by-step implementation of a Generative Pre-trained Transformer (GPT)-like language model from scratch using Python and PyTorch, mirroring the development process shown in the video [1].

## Overview

This repository contains the code for building a **decoder-only Transformer** [2] capable of language modeling. Following the video, we start with basic data processing and a simple character-level model, and progressively build up to a multi-layered Transformer with self-attention, ultimately training it on the **Tiny Shakespeare dataset** [3]. The goal is to demystify the inner workings of large language models like ChatGPT by constructing a fundamental version [4].

The implementation focuses on a **character-level language model** [3] where the model learns to predict the next character in a sequence of Shakespeare's works [3]. The final Transformer model learns the patterns within this data and can generate text that resembles it [5].

## Key Concepts and Implementation Steps

This project covers the following key concepts and implementation steps, as detailed in the video:

*   **Language Modeling:** The fundamental task of predicting the next element (character in this case) in a sequence [4]. The model completes a given sequence of text [4].
*   **Tokenization:** Converting raw text into a sequence of integers [6] based on a vocabulary of unique characters in the dataset [1, 6]. This project uses a simple **character-level tokenizer** [6] with an encoder and decoder to map characters to integers and back [6].
*   **Data Preparation:**
    *   Downloading and loading the **Tiny Shakespeare dataset**, a concatenation of all of Shakespeare's works [1, 3].
    *   Creating a vocabulary of all unique characters in the text [1].
    *   Encoding the entire text into a single sequence of integers using the character-level tokenizer [7].
    *   Splitting the data into **training (90%) and validation (10%) sets** to monitor for overfitting [7, 8].
*   **Batching:** Dividing the training data into smaller chunks of sequences with a fixed length called **block size** (e.g., 8 initially, then 256) [8-10]. During training, random chunks are sampled from the training set [8, 11]. Each batch consists of multiple independent sequences processed in parallel for efficiency [11]. The input `X` and target `Y` within a batch are offset by one, creating multiple training examples within a single block [8, 9].
*   **Byram Language Model:** A simple initial neural network where the prediction for the next character is based solely on the embedding of the current character, without considering the preceding context [12]. This model uses an **embedding table** to look up vector representations for each character [12, 13]. The loss is calculated using **cross-entropy** between the model's predictions (logits) and the target characters [13, 14].
*   **Generation from the Byram Model:** A `generate` function is implemented to sample the next character based on the probability distribution predicted by the model [15, 16].
*   **Token and Position Embeddings:**
    *   **Token Embedding:** Each character is represented by a low-dimensional vector looked up from an embedding table (`nn.Embedding`) [17]. The size of the embedding dimension (`n_embed`) is a hyperparameter [10, 17].
    *   **Position Embedding:** Each position in the sequence (from 0 to `block_size - 1`) also has an associated embedding vector (`nn.Embedding`) [17]. These embeddings are added to the token embeddings, providing the model with information about the order of tokens [17, 18].
*   **Self-Attention:** A key mechanism allowing tokens to interact and understand relationships within the sequence [18, 19].
    *   **Queries, Keys, and Values:** Each token's embedding is linearly transformed to produce query, key, and value vectors [19].
    *   **Attention Scores (Weights):** Calculated by taking the dot product of query vectors with key vectors of all tokens in the sequence [19, 20].
    *   **Masked Self-Attention:** A lower triangular mask is applied to the attention scores to prevent a token from attending to future tokens, ensuring the autoregressive nature of the decoder [20-23].
    *   **Scaled Dot-Product Attention:** Attention scores are scaled by the inverse square root of the head size to stabilize training by preventing the dot products from becoming too large [24].
    *   **Softmax:** Applied to the masked and scaled attention scores to obtain attention probabilities (weights that sum to 1) [22, 25].
    *   **Value Aggregation:** The attention weights are used to perform a weighted sum of the value vectors, producing the output of the attention mechanism [25].
*   **Multi-Head Attention:** Running multiple independent self-attention "**heads**" in parallel [26]. Each head learns different types of relationships. The outputs of all heads are concatenated and then linearly projected back to the original embedding dimension [27, 28]. The number of heads (`n_head`) and the head size (`head_size`) are hyperparameters [10, 26, 29].
*   **Feedforward Networks:** A simple multi-layer perceptron (MLP) applied to the output of the attention mechanism on a per-token basis [30]. It consists of a linear layer, a non-linearity (ReLU initially implied, later GELU in `nanogpt` [30, 31]), and another linear layer [30, 32]. The inner layer typically has a larger dimension (e.g., 4 times `n_embed`) [32].
*   **Residual Connections (Skip Connections):** The input of each sub-layer (attention or feedforward) is added to its output (`x = x + self_attention(x)`) [28, 29]. This helps with gradient flow and enables the training of deeper networks [29].
*   **Layer Normalization:** Normalizing the activations within each layer before applying attention and feedforward networks (**pre-norm formulation**) [33]. LayerNorm normalizes across the features of each token [32, 33]. Two layer normalization layers are used in each Transformer block [33].
*   **Transformer Block:** A fundamental building block consisting of layer normalization, multi-head self-attention with a residual connection, another layer normalization, and a feedforward network with a residual connection [29, 30, 33]. Multiple such blocks are stacked (`n_layer`) to create the full Transformer model [10, 29].
*   **Decoder-only Architecture:** The implemented Transformer lacks the encoder component of the original Transformer and only uses the decoder part with masked self-attention [2]. This architecture is suitable for generative tasks like language modeling [2].
*   **Training Loop:** Iterating over the training data in batches for a specified number of steps [34, 35]. For each batch, the loss (cross-entropy between the model's logits and the target characters) is calculated, gradients are computed using backpropagation, and the model's parameters are updated using an optimizer like AdamW [34]. Learning rate scheduling and gradient clipping are common practices in more advanced training loops (as seen in `nanogpt`'s `train.py` [31]).
*   **Generation:** Sampling new text from the trained model by repeatedly predicting the next character, conditioned on the previously generated sequence, up to a maximum number of new tokens [5, 15, 16]. The context provided to the model during generation is cropped to the `block_size` [26].
*   **Scaling Up:** Increasing the model's capacity by increasing the embedding dimension (`n_embed`), the number of attention heads (`n_head`), and the number of layers (`n_layer`) [10]. Larger models trained on more data generally exhibit improved performance [10, 36].
*   **Dropout:** A regularization technique (`nn.Dropout`) applied after attention and feedforward layers to prevent overfitting by randomly setting a fraction of the neurons' outputs to zero during training [10, 37].

The repository also includes "**nanog GPT**" [5], a simplified and efficient implementation of a Transformer in two main files:
*   `model.py`: Defines the GPT model (Transformer architecture) [5, 31].
*   `train.py`: Contains the code for training the model on a given text dataset [5, 31]. This script in `nanogpt` includes more advanced features like GPU support, evaluation loss estimation over multiple batches, saving and loading checkpoints, learning rate decay, and potentially distributed training [31, 38].

## Usage

To use the code in this repository:

1.  Ensure you have **Python 3.x** and **PyTorch** installed on your system. You can find installation instructions on the official PyTorch website.
2.  Clone this repository to your local machine:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
3.  Download the **Tiny Shakespeare dataset** (`input.txt`) if it's not already included. The `train.py` script in `nanogpt` typically handles this [1].
4.  Run the training script (`train.py`) with the desired hyperparameters. You can modify these directly in the script or potentially pass them as command-line arguments (refer to the `train.py` script for specific options) [31, 35, 38]. For example:
    ```bash
    python train.py
    ```
5.  Once the model is trained, you can use the generation functionality within the `model.py` or `train.py` (depending on the specific implementation) to generate new text [5, 15, 16]. This usually involves loading a trained checkpoint and providing a starting context.

Refer to the comments and documentation within the `model.py` and `train.py` files for more detailed usage instructions and hyperparameter explanations.

## Results

The video demonstrates a clear progression in the model's ability to generate text as more sophisticated components of the Transformer are added and the model is scaled up [10, 26, 30, 32, 35].

*   Training a **Byram language model** results in a decreasing loss and the generation of some basic character sequences [35].
*   Introducing **self-attention**, then **multi-head self-attention**, and **feedforward networks** further reduces the validation loss, indicating improved learning of dependencies in the data [26, 30].
*   The incorporation of **residual connections** and **layer normalization** enables the training of deeper and more powerful Transformer models [29, 32, 33].
*   Scaling up the model to approximately **10 million parameters** with an embedding dimension of 384, 6 attention heads, and 6 layers, and training on the Tiny Shakespeare dataset for about 15 minutes on an A100 GPU, achieved a validation loss of **1.48** [10]. The generated text, while still nonsensical, started to exhibit more recognizable Shakespearean-like patterns and vocabulary [2, 10].

These results demonstrate the effectiveness of the Transformer architecture for language modeling, even on a relatively small dataset. Scaling up the model size and training data significantly, as done in models like GPT-3 (which has 175 billion parameters and was trained on 300 billion tokens [36]), leads to much more coherent and capable language models. This project provides a foundational understanding of the core Transformer architecture that underlies these powerful systems [3].

## Further Learning

*   **Attention is All You Need:** The original paper introducing the Transformer architecture [3, 24, 27, 39].
*   **Andrej Karpathy's Previous Videos (especially the "make more series"):** These videos cover simpler language models and provide a strong foundation for understanding the concepts presented here [1, 12, 13, 32, 39].
*   **The nanog GPT repository:** Explore the `model.py` and `train.py` files for a clean and efficient implementation of the Transformer [5, 31].
*   **Pre-training and Fine-tuning:** Understand the two main stages involved in developing large language models like ChatGPT. This project focuses on **pre-training** [36, 40].
*   **Encoder-Decoder vs. Decoder-Only Architectures:** Learn about the differences and use cases of these Transformer variants [2, 23].