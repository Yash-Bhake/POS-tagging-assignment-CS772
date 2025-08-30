import gradio as gr
import torch
import torch.nn as nn
import json
import random
import numpy as np

# --- 1. SET UP DEVICE AND REPRODUCIBILITY ---
# This ensures that the model runs on the correct hardware (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# --- 2. DEFINE THE MODEL ARCHITECTURE ---
# The model class definitions MUST be present in the script to load the saved model.
# This code is identical to what was used in the training script.

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
        i_t, f_t, g_t, o_t = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm_cells = nn.ModuleList([LSTMCell(emb_dim if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])

    def forward(self, src):
        batch_size, seq_len = src.shape
        embedded = self.embedding(src)
        
        # Initialize hidden and cell states as lists of tensors for each layer
        h_states = [torch.zeros(batch_size, self.hidden_dim).to(device) for _ in range(self.n_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_dim).to(device) for _ in range(self.n_layers)]

        input_for_lstm = embedded.permute(1, 0, 2)
        for t in range(seq_len):
            input_t = input_for_lstm[t]
            for i, layer in enumerate(self.lstm_cells):
                # Update the states in the list (this is not an inplace operation)
                h_states[i], c_states[i] = layer(input_t, (h_states[i], c_states[i]))
                input_t = h_states[i]
        
        # Stack the lists to create the final context tensors
        return torch.stack(h_states), torch.stack(c_states)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm_cells = nn.ModuleList([LSTMCell(emb_dim if i == 0 else hidden_dim, hidden_dim) for i in range(n_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        # Unbind the layers into a list of tensors
        hidden_states = list(hidden.unbind(0))
        cell_states = list(cell.unbind(0))

        input = input.unsqueeze(0)
        embedded = self.embedding(input).squeeze(0)
        
        input_for_layer = embedded
        for i, layer in enumerate(self.lstm_cells):
            hidden_states[i], cell_states[i] = layer(input_for_layer, (hidden_states[i], cell_states[i]))
            input_for_layer = hidden_states[i]
        
        # Stack the updated states back into single tensors
        h_new = torch.stack(hidden_states)
        c_new = torch.stack(cell_states)
            
        prediction = self.fc_out(h_new[-1])
        return prediction, h_new, c_new

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs.permute(1, 0, 2)

# --- 3. LOAD ARTIFACTS AND MODEL ---

# Load vocabularies and configuration from the JSON file
with open('artifacts.json', 'r') as f:
    artifacts = json.load(f)

word_to_idx = artifacts['word_to_idx']
tag_to_idx = artifacts['tag_to_idx']
MAX_LEN = artifacts['MAX_LEN']

# Create the inverse mapping for tags to convert model output back to labels
idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}

# Define hyperparameters (these MUST match the trained model's parameters)
INPUT_DIM = len(word_to_idx)
OUTPUT_DIM = len(tag_to_idx)
ENC_EMB_DIM = 300
DEC_EMB_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2

# Instantiate the model architecture
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS)
model = EncoderDecoder(encoder, decoder, device).to(device)

# Load the trained model weights
model.load_state_dict(torch.load('pos-tagger-model.pt', map_location=device))
model.eval() # Set the model to evaluation mode

# --- 4. CREATE THE PREDICTION FUNCTION ---

def predict_tags(sentence: str):
    """
    Takes a raw sentence string, validates it, preprocesses it,
    runs it through the model, and formats the output for Gradio.
    """
    # --- Input Validation ---
    if not sentence or not sentence.strip():
        # Handle empty input gracefully
        return [] # Return an empty list for HighlightedText
    
    tokens = sentence.strip().split()
    if len(tokens) > MAX_LEN:
        # Handle sentences that are too long
        raise gr.Error(f"Input sentence is too long. Please use a sentence with no more than {MAX_LEN} words.")

    # --- Preprocessing ---
    # Convert tokens to lowercase and then to numerical indices
    indices = [word_to_idx.get(token.lower(), word_to_idx['<UNK>']) for token in tokens]
    
    # Pad the sequence to the MAX_LEN
    padded_indices = indices + [word_to_idx['<PAD>']] * (MAX_LEN - len(indices))
    src_tensor = torch.LongTensor(padded_indices).unsqueeze(0).to(device)
    
    # Create a dummy target tensor for the model's forward pass
    trg_tensor = torch.zeros_like(src_tensor).to(device)
    
    # --- Inference ---
    with torch.no_grad():
        # Run the model with teacher forcing off
        output = model(src_tensor, trg_tensor, 0)
    
    # Get the predicted tag indices by taking the argmax
    pred_indices = output.argmax(2).squeeze(0).cpu().numpy()
    
    # --- Post-processing and Formatting ---
    # Convert indices back to tag labels
    # FIX: Look up the integer index `i` directly, without converting to a string.
    pred_tags = [idx_to_tag.get(i, '???') for i in pred_indices[1:len(tokens)+1]]
    
    # Format for Gradio's HighlightedText component
    # It expects a list of (word, tag) tuples
    highlighted_output = list(zip(tokens, pred_tags))
    
    return highlighted_output

# --- 5. BUILD AND LAUNCH THE GRADIO INTERFACE ---

# Define a color map for better visualization of tags
color_map = {
    "NOUN": "#FFA07A",  # Light Salmon
    "VERB": "#20B2AA",  # Light Sea Green
    "ADJ": "#778899",   # Light Slate Gray
    "ADP": "#87CEFA",   # Light Sky Blue
    "ADV": "#90EE90",   # Light Green
    "CONJ": "#FFD700",  # Gold
    "DET": "#DDA0DD",   # Plum
    "NUM": "#B0C4DE",   # Light Steel Blue
    "PRON": "#FFB6C1",  # Light Pink
    "PRT": "#F0E68C",   # Khaki
    ".": "#D3D3D3",     # Light Grey
    "X": "#F4A460"       # Sandy Brown
}

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_tags,
    inputs=gr.Textbox(
        lines=3,
        label="Input Sentence",
        placeholder="Enter a sentence here..."
    ),
    outputs=gr.HighlightedText(
        label="Tagged Sentence",
        color_map=color_map,
        show_legend=True
    ),
    title="Part-of-Speech Tagger using a from-scratch LSTM",
    description="This demo uses an Encoder-Decoder model with a custom-built LSTM to predict Part-of-Speech tags for an input sentence. The model was trained on the Brown Corpus. Enter a sentence to see the model's predictions.",
    examples=[
        ["The old man the boat."],
        ["Colorless green ideas sleep furiously."],
        ["All the king's horses and all the king's men couldn't put Humpty together again."]
    ],
    allow_flagging="never" # Disable flagging for this demo
)

# Launch the app!
iface.launch()


