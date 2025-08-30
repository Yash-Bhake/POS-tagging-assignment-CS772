import torch
import torch.nn as nn
import gradio as gr
import json
import numpy as np
import re # Import the regular expressions module

# --- 1. Define the Model Architecture ---
# This must be the EXACT same architecture as the one you trained.
# We include LSTMCell since the main model depends on it.

class LSTMCell(nn.Module):
    """A from-scratch implementation of a single LSTM cell."""
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, states):
        h_prev, c_prev = states
        gates = self.linear_ih(x) + self.linear_hh(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)
        
        i_t = torch.sigmoid(i)
        f_t = torch.sigmoid(f)
        g_t = torch.tanh(g)
        o_t = torch.sigmoid(o)
        
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm_cells_fwd = nn.ModuleList([LSTMCell(embedding_dim if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
        self.lstm_cells_bwd = nn.ModuleList([LSTMCell(embedding_dim if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, src):
        batch_size, seq_len = src.shape
        embedded = self.dropout(self.embedding(src))
        input_seq = embedded.permute(1, 0, 2)

        fwd_outputs = []
        h_fwd = [torch.zeros(batch_size, self.hidden_dim).to(self.device) for _ in range(self.n_layers)]
        c_fwd = [torch.zeros(batch_size, self.hidden_dim).to(self.device) for _ in range(self.n_layers)]
        for t in range(seq_len):
            input_t = input_seq[t]
            for i, layer in enumerate(self.lstm_cells_fwd):
                h_fwd[i], c_fwd[i] = layer(input_t, (h_fwd[i], c_fwd[i]))
                input_t = self.dropout(h_fwd[i]) if i < self.n_layers - 1 else h_fwd[i]
            fwd_outputs.append(h_fwd[-1])
            
        bwd_outputs = []
        h_bwd = [torch.zeros(batch_size, self.hidden_dim).to(self.device) for _ in range(self.n_layers)]
        c_bwd = [torch.zeros(batch_size, self.hidden_dim).to(self.device) for _ in range(self.n_layers)]
        for t in range(seq_len - 1, -1, -1):
            input_t = input_seq[t]
            for i, layer in enumerate(self.lstm_cells_bwd):
                h_bwd[i], c_bwd[i] = layer(input_t, (h_bwd[i], c_bwd[i]))
                input_t = self.dropout(h_bwd[i]) if i < self.n_layers - 1 else h_bwd[i]
            bwd_outputs.append(h_bwd[-1])
        bwd_outputs.reverse()
        
        fwd_outputs_tensor = torch.stack(fwd_outputs)
        bwd_outputs_tensor = torch.stack(bwd_outputs)
        lstm_outputs = torch.cat((fwd_outputs_tensor, bwd_outputs_tensor), dim=2)
        
        predictions = self.fc_out(self.dropout(lstm_outputs))
        return predictions.permute(1, 0, 2)

# --- 2. Load Artifacts and Model ---
# Load preprocessing artifacts
with open('artifacts.json', 'r') as f:
    artifacts = json.load(f)

word_to_idx = artifacts['word_to_idx']
tag_to_idx = artifacts['tag_to_idx']
MAX_LEN = artifacts['MAX_LEN']

# Reverse mapping for converting predictions back to tags
# JSON loads integer keys as strings, so we must convert them back
idx_to_tag = {int(i): tag for i, tag in artifacts.get('idx_to_tag', {v: k for k, v in tag_to_idx.items()}).items()}


# Model Hyperparameters (should match your training script)
INPUT_DIM = len(word_to_idx)
OUTPUT_DIM = len(tag_to_idx)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 1
DROPOUT = 0.5
WORD_PAD_IDX = word_to_idx['<PAD>']

# Instantiate the model
model = BiLSTMPOSTagger(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, WORD_PAD_IDX)

# Load the trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('pos-tagger-model-fixed.pt', map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode

# --- 3. Prediction Function ---
def tag_sentence(sentence):
    """Takes a sentence string and returns a list of (word, tag) tuples."""
    # --- CHANGE START: Improved Tokenization ---
    # This regex finds words or punctuation marks.
    original_words = re.findall(r"[\w']+|[.,!?;]", sentence)
    tokens = [word.lower() for word in original_words]
    # --- CHANGE END ---
    
    # Truncate if sentence is longer than MAX_LEN
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
        original_words = original_words[:MAX_LEN]

    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    
    padded_indices = indices + [WORD_PAD_IDX] * (MAX_LEN - len(indices))
    src_tensor = torch.LongTensor(padded_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(src_tensor)
    
    pred_indices = output.argmax(2).squeeze(0).cpu().numpy()
    
    pred_tags = [idx_to_tag.get(i, 'UNK') for i in pred_indices[:len(tokens)]]
    
    # Format for Gradio's HighlightedText component
    return list(zip(original_words, pred_tags))

# --- 4. Create and Launch the Gradio Interface ---
title = "Part-of-Speech Tagger"
description = """
Enter an English sentence to see its Part-of-Speech tags based on the Universal Tagset. 
The model is a Bidirectional LSTM trained on the Brown Corpus.
Common Tags: NOUN (noun), VERB (verb), ADJ (adjective), ADP (adposition), ADV (adverb), CONJ (conjunction), DET (determiner), NUM (numeral), PRON (pronoun), PRT (particle), . (punctuation), X (other).
"""

iface = gr.Interface(
    fn=tag_sentence,
    inputs=gr.Textbox(lines=3, label="English Sentence"),
    outputs=gr.HighlightedText(label="Tagged Sentence"),
    title=title,
    description=description,
    examples=[
        ["The old man the boat."],
        ["Colorless green ideas sleep furiously"],
        ["The quick, brown fox jumps over the lazy dog!"]
    ]
)

if __name__ == "__main__":
    iface.launch()

