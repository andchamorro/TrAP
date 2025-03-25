import json
from typing import Optional
from math import ceil, sqrt
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
class AlbertConfig:
    """
    This is the configuration class to store the configuration of an ALBERT model. It is used to instantiate an ALBERT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the ALBERT architecture.

    Args:
        vocab_size (int, optional, defaults to 30522): 
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the 
            `input_ids` passed when calling the model.
        embedding_size (int, optional, defaults to 24): 
            Dimensionality of the token embeddings.
        hidden_size (int, optional, defaults to 64): 
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (int, optional, defaults to 6): 
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (int, optional, defaults to 12): 
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (int, optional, defaults to 768): 
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        max_position_embeddings (int, optional, defaults to 512): 
            The maximum sequence length that this model might ever be used with. Typically set this to something large 
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (int, optional, defaults to 2): 
            The vocabulary size of the `token_type_ids` passed when calling the model.
        attention_probs_dropout_prob (float, optional, defaults to 0.0): 
            The dropout ratio for the attention probabilities.
        hidden_dropout_prob (float, optional, defaults to 0.1): 
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        classifier_dropout_prob (float, optional, defaults to 0.1): 
            The dropout ratio for the classification head.
        layer_norm_eps (float, optional, defaults to 1e-12): 
            The epsilon used by the layer normalization layers.
        hidden_act (str, optional, defaults to "gelu_new"): 
            The activation function to use.
        net_structure_type (int, optional, defaults to 0): 
            The network structure type.
        pad_token_id (int, optional, defaults to 0): 
            The ID of the padding token.
        bos_token_id (int, optional, defaults to 2): 
            The ID of the beginning-of-sequence token.
        eos_token_id (int, optional, defaults to 3): 
            The ID of the end-of-sequence token.
    """
    def __init__(self,
                 architectures=[],
                 model_type="albert",
                 vocab_size=30522, 
                 embedding_size=24,
                 hidden_size=64,
                 num_hidden_layers=6,
                 num_attention_heads=12, 
                 intermediate_size=768,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 attention_probs_dropout_prob=0.0,
                 hidden_dropout_prob=0.1,
                 classifier_dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 hidden_act="gelu_new",
                 net_structure_type=0,
                 pad_token_id=0,
                 bos_token_id=2,
                 eos_token_id=3):
        
        self.architectures = architectures
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.net_structure_type = net_structure_type
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.embedding_to_hidden = nn.Linear(config.embedding_size, config.hidden_size)
        self.position_embeddings = PositionalEmbedding(config.hidden_size, max_length = config.max_position_embeddings, dropout = config.hidden_dropout_prob)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute") # default absolute

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()

        if position_ids is None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape) # (S,) -> (B, S)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)
        
        # Factorized Embedding
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.embedding_to_hidden(words_embeddings)
        segment_embeddings = self.segment_embeddings(token_type_ids)

        embeddings = words_embeddings + segment_embeddings
        # Positional Embedding
        if self.position_embedding_type == "absolute":
            embeddings = self.position_embeddings(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden_size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        attention_probs = attention_probs

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(2, 1).flatten(2)

        projected_context_layer = self.dense(context_layer)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        return layernormed_context_layer
class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention = SoftmaxAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.max_position_embeddings, config.hidden_size], eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions = False):
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)
        layernormed_context_layer = self.LayerNorm(attention_output + hidden_states)

        intermediate_output = self.intermediate(layernormed_context_layer)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.LayerNorm(layer_output + layernormed_context_layer)
        return (layer_output, attention_output) if output_attentions else layernormed_context_layer

class AlbertModel(nn.Module):
    def __init__(self, config):
        super(AlbertModel, self).__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertLayer(config)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []

        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                total_numel.append(param.numel())

        return sum(total_numel)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        # Albert Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        hidden_states = embedding_output
        # Albert Transformer
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(), device=input_ids.device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(attention_mask.dtype).min
        for _ in range(self.config.num_hidden_layers):
            hidden_states = self.encoder(hidden_states, extended_attention_mask)
        return hidden_states
    
class AlbertMLMHead(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores

class AlbertForMaskedLM(nn.Module):
    def __init__(self, transformer, loss_fct=None):
        super(AlbertForMaskedLM, self).__init__()
        self.config = transformer.config
        self.transformer = transformer
        self.predictions = AlbertMLMHead(transformer.config)
        self.loss_fct = loss_fct if loss_fct is not None else nn.CrossEntropyLoss()
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        return self.transformer.num_parameters(only_trainable, exclude_embeddings)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None, **kwargs):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        prediction_scores = self.predictions(outputs)
        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        return dict(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs)
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

    def forward(self, input_ids, token_type_ids=None, position_ids=None, key_padding_mask=None, masked_pos=None):
        attn = self.transformer(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, key_padding_mask=key_padding_mask)

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
        # TODO: position_ids
        cls_logits, token_logits = self.transformer(input_ids, token_type_ids, position_ids=None, key_padding_mask=key_padding_mask)

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