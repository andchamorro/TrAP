import json
from math import ceil
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

def check_for_shape(tensor, name):
    print(f"shape {tensor.shape} of {name}")

def check_for_type(tensor, name):
    print(f"type {tensor.type()} of {name}")

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")

def check_all_padding_sequences(input_ids, pad_token_id):
    # Check if each sequence in the batch is entirely padding
    all_padding_mask = (input_ids == pad_token_id).all(dim=1)
    return all_padding_mask
        
# Example usage
# check_for_nans(input_ids, "input_ids")
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
                 hidden_dropout_prob = 0.1,
                 type_vocab_size=2):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob # in albert, dropout can potentially hurt performance at large sizes.
        self.type_vocab_size = type_vocab_size
    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))
    
class PositionalEmbedding(nn.Module):
    def __init__(self, nb_in, dropout=0.0, max_length=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, nb_in)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nb_in, 2).float() * (-torch.log(torch.tensor(10000.0)) / nb_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class AlbertEmbeddings(nn.Module):
    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.embedding_to_hidden = nn.Linear(config.embedding_size, config.hidden_size)
        self.position_embeddings = PositionalEmbedding(config.hidden_size, max_length = config.max_position_embeddings, dropout = config.hidden_dropout_prob)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()

        if position_ids is None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape) # (S,) -> (B, S)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        
        # Factorized Embedding
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.embedding_to_hidden(words_embeddings)
        segment_embeddings = self.segment_embeddings(token_type_ids)

        embeddings = words_embeddings + segment_embeddings
        # Positional Embedding
        embeddings = self.position_embeddings(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class SoftmaxAttention(nn.Module):
    def __init__(self, embed_dim: int, num_attention_heads: int = 8, dropout_rate: float = 0.0, batch_first: bool = False):
        super().__init__()
        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_attention_heads, dropout=dropout_rate, batch_first=batch_first)

        # Linear layers for query, key, and value projections
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.qkv_projection.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.output_projection.weight, nonlinearity='linear')

    def forward(self, x: torch.FloatTensor, attention_mask: torch.BoolTensor = None):
        # Project input to query, key, and value tensors
        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)
        
        # Transpose for multi-head attention
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(q, k, v, key_padding_mask=~attention_mask)
        
        # Transpose back and project output
        attn_output = attn_output.transpose(0, 1)
        output = self.output_projection(attn_output)
        
        return output
class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention = SoftmaxAttention(config.hidden_size, config.num_attention_heads, dropout_rate=config.hidden_dropout_prob)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.max_position_embeddings, config.hidden_size], eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.LayerNorm(layer_output + attention_output)
        return layer_output

class AlbertModel(nn.Module):
    def __init__(self, config):
        super(AlbertModel, self).__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = nn.ModuleList([AlbertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, token_type_ids, key_padding_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        hidden_states = embedding_output
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states, key_padding_mask)
        return hidden_states
    
class AlbertMaskedWrapper(nn.Module):
    """
    Wrapper for ALBERT Pretraining: Masked Language Modeling (MLM) and Sentence Order Prediction (SOP)
    """
    def __init__(self, transformer):
        super(AlbertMaskedWrapper, self).__init__()
        self.transformer = transformer
        self.cls_mlp = nn.Linear(self.transformer.config.hidden_size, 2)
        
        # Decoder is shared with embedding layer
        self.ve_weight = self.transformer.embeddings.word_embeddings.weight
        self.eh_weight = self.transformer.embeddings.embedding_to_hidden.weight.t()
        self.eh_bias = self.transformer.embeddings.embedding_to_hidden.bias

    def forward(self, input_ids, token_type_ids=None, key_padding_mask=None, masked_pos=None):
        attn = self.transformer(input_ids, token_type_ids=token_type_ids, key_padding_mask=key_padding_mask)

        cls_attn = attn[:, 0]
        cls_logits = self.cls_mlp(cls_attn)

        # Ensure the dimensions match for the linear transformation
        attn = attn.view(-1, attn.size(-1))  # Flatten the attn tensor
        token_logits = F.linear(F.linear(attn + self.eh_bias, self.eh_weight), self.ve_weight)
        token_logits = token_logits.view(input_ids.size(0), input_ids.size(1), -1)  # Reshape back to original dimensions

        return cls_logits, token_logits
    
class AlbertMaskedTrainer:
    """
    Pretraining Helper Class for ALBERT: Masked LM and Sentence Order Prediction (SOP)
    """
    def __init__(self, config_model, config_train):
        self.config = config_model
        #self.device = torch.device(device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() and not (config_train and config_train.cpu) else 'cpu')
        self.device = torch.device('cpu')
        self.transformer = AlbertMaskedWrapper(AlbertModel(config_model).to(self.device)).to(self.device)
        if config_train.data_parallel: # use Data Parallelism with Multi-GPU
            self.transformer = nn.DataParallel(self.transformer)

        self.optim = torch.optim.Adam(self.transformer.parameters(), lr=config_train.learning_rate)
        self.optim.zero_grad()

        self.scaler = torch.amp.GradScaler(self.device.type)  # for automatic mixed-precision

        self.update_frequency = ceil(config_train.batch_size / config_train.mini_batch_size)
        self.train_steps = 0

    #@torch.autocast(device_type="cuda")
    def _calculate_loss(self, 
            input_ids: torch.LongTensor, 
            token_type_ids: torch.LongTensor, 
            key_padding_mask: torch.BoolTensor,
            token_labels: torch.LongTensor,
            sentence_order: torch.LongTensor,
        ):
        """
        Calculates Masked LM and SOP loss
        """
        cls_logits, token_logits = self.transformer(input_ids, token_type_ids, key_padding_mask=key_padding_mask)

        cls_loss = F.cross_entropy(cls_logits, sentence_order)

        cls_prediction = torch.argmax(cls_logits, dim=-1)  # for accuracy calculating, not training
        cls_correct = (cls_prediction == sentence_order).sum()
        cls_total = sentence_order.shape[0]
        cls_accuracy = cls_correct / cls_total

        masked_token_idx = (token_labels >= 0)
        masked_targets = token_labels[masked_token_idx]
        masked_token_logits = token_logits[masked_token_idx]

        # Check the range of masked_targets
        num_classes = token_logits.size(-1)
        if masked_targets.max() >= num_classes:
            raise ValueError(f"Target value {masked_targets.max()} is out of bounds for {num_classes} classes.")

        token_loss = F.cross_entropy(masked_token_logits, masked_targets)
        
        masked_token_prediction = torch.argmax(masked_token_logits, dim=-1)
        token_correct = (masked_token_prediction == masked_targets).sum()
        token_total = masked_targets.shape[0]
        token_accuracy = token_correct / token_total

        loss = cls_loss + token_loss

        return loss, (cls_loss, cls_accuracy), (token_loss, token_accuracy)

    def _unpack_batch(self, batch):
        """
        Unpack batch and cast to device
        """
        batch = [b.to(self.device) for b in batch]
        return batch
    
    def train_step(self, batch):
        """
        Perform one step of training, accumulate gradients and return metrics
        """
        X, seg, key_padding_mask, token_labels, sentence_order = self._unpack_batch(batch)
        self.transformer.train()
        with torch.autocast(device_type="cuda") if self.device.type == 'cuda' else nullcontext():
            loss, *metrics = self._calculate_loss(
                X, seg, key_padding_mask,
                token_labels, sentence_order,
                )
        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()
        return (loss, *metrics)

    @torch.no_grad()
    def eval_step(self, batch):
        """
        Perform one step of evaluation and return metrics
        """
        X, seg, key_padding_mask, token_labels, sentence_order = self._unpack_batch(batch)
        self.transformer.eval()
        loss, *metrics = self._calculate_loss(
            X, seg, key_padding_mask.bool(),
            token_labels, sentence_order,
        )

        return (loss, *metrics)