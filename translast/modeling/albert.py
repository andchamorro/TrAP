import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlbertConfig:
    """
    This is the configuration class to store the configuration of an ALBERT model. It is used to instantiate an ALBERT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the ALBERT architecture.

    Args:
        vocab_size (int, optional, defaults to None): 
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the 
            `input_ids` passed when calling the model.
        embedding_size (int, optional, defaults to 128): 
            Dimensionality of the token embeddings.
        hidden_size (int, optional, defaults to 4096): 
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (int, optional, defaults to 12): 
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (int, optional, defaults to 64): 
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (int, optional, defaults to 16384): 
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        max_position_embeddings (int, optional, defaults to 512): 
            The maximum sequence length that this model might ever be used with. Typically set this to something large 
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (int, optional, defaults to 2): 
            The vocabulary size of the `token_type_ids` passed when calling the model.
    """
    def __init__(self, vocab_size=None, 
                 embedding_size=128,
                 hidden_size=4096,
                 num_hidden_layers=12,
                 num_attention_heads=64, 
                 intermediate_size=16384,
                 max_position_embeddings=512,
                 type_vocab_size=2):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

class AlbertEmbeddings(nn.Module):
    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.embedding_to_hidden = nn.Linear(config.embedding_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape) # (S,) -> (B, S)

        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.embedding_to_hidden(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)
        return layer_output

class AlbertModel(nn.Module):
    def __init__(self, config):
        super(AlbertModel, self).__init__()
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = nn.ModuleList([AlbertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids)
        hidden_states = embedding_output
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states