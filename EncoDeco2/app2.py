import torch
import torch.nn as nn
import gradio as gr
import json

# --- 1. Model Architecture Definition ---
# This must be IDENTICAL to the architecture used during training.
# We need to define it here so PyTorch knows how to load the weights.

# Hyperparameters (must match the trained model)
EMBEDDING_DIM = 64
HIDDEN_DIM = 64

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_g = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, states):
        h_prev, c_prev = states
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        g_t = torch.tanh(self.W_g(x) + self.U_g(h_prev))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = LSTMCell(embedding_dim, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        embedded = self.embedding(x)
        h = torch.zeros(1, self.hidden_size)
        c = torch.zeros(1, self.hidden_size)
        for i in range(embedded.shape[0]):
            h, c = self.lstm_cell(embedded[i].unsqueeze(0), (h, c))
        return h, c

class Decoder(nn.Module):
    def __init__(self, tag_size, embedding_dim, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tag_size, embedding_dim)
        self.lstm_cell = LSTMCell(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, x, states):
        embedded = self.embedding(x)
        h, c = self.lstm_cell(embedded, states)
        output = self.fc(h)
        return output, (h, c)

class Seq2SeqPOSTagger(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqPOSTagger, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


# --- 2. Load Artifacts and Instantiate Model ---

# Load the vocabularies
with open('word_to_ix.json', 'r') as f:
    word_to_ix = json.load(f)
with open('tag_to_ix.json', 'r') as f:
    tag_to_ix = json.load(f)

# Create the inverse mapping for tags
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

# Instantiate the model with the correct dimensions
encoder = Encoder(len(word_to_ix), EMBEDDING_DIM, HIDDEN_DIM)
decoder = Decoder(len(tag_to_ix), EMBEDDING_DIM, HIDDEN_DIM)
model = Seq2SeqPOSTagger(encoder, decoder)

# Load the saved weights
model.load_state_dict(torch.load('pos_tagger_model.pth'))
model.eval() # Set the model to evaluation mode

print("✅ Model and vocabularies loaded successfully.")


# --- 3. Prediction Function ---

def tag_sentence(sentence, model, word_to_ix, ix_to_tag):
    model.eval()
    words = sentence.lower().split()
    word_indices = [word_to_ix.get("<SOS>", 2)] + [word_to_ix.get(w, word_to_ix.get("<UNK>", 1)) for w in words] + [word_to_ix.get("<EOS>", 3)]
    sentence_tensor = torch.tensor(word_indices)

    predicted_tags = []
    max_len = len(word_indices) + 5 # Add a buffer

    with torch.no_grad():
        h, c = model.encoder(sentence_tensor)
        decoder_input = torch.tensor([tag_to_ix["<SOS>"]])
        
        for _ in range(max_len):
            output, (h, c) = model.decoder(decoder_input, (h, c))
            top_prediction = output.argmax(1)
            predicted_tag_ix = top_prediction.item()
            
            if predicted_tag_ix == tag_to_ix["<EOS>"]:
                break
            
            predicted_tags.append(ix_to_tag[predicted_tag_ix])
            decoder_input = top_prediction
            
    return predicted_tags

# --- 4. Gradio Interface Function ---

def predict_pos_tags(sentence):
    """
    Takes a sentence string and returns a formatted list for Gradio's HighlightedText.
    """
    if not sentence:
        return []
    words = sentence.split()
    tags = tag_sentence(sentence, model, word_to_ix, ix_to_tag)
    
    # Align words and tags, handling potential length mismatch
    output = []
    for i in range(min(len(words), len(tags))):
        output.append((words[i], tags[i]))
        
    return output

# --- 5. Launch the Gradio App ---

iface = gr.Interface(
    fn=predict_pos_tags,
    inputs=gr.Textbox(
        lines=3, 
        label="Input Sentence", 
        placeholder="Enter a sentence here..."
    ),
    outputs=gr.HighlightedText(
        label="Tagged Sentence",
        color_map={
            "NOUN": "red",
            "VERB": "green",
            "ADJ": "blue",
            "ADP": "purple",
            "ADV": "orange",
            "CONJ": "yellow",
            "DET": "teal",
            "NUM": "pink",
            "PRON": "brown",
            "PRT": "gray",
            ".": "black",
            "X": "silver"
        }
    ),
    title="📝 LSTM Part-of-Speech Tagger",
    description="This app uses a from-scratch LSTM Encoder-Decoder model to predict the part-of-speech (POS) tag for each word in a sentence. The model was trained on the Brown Corpus.",
    examples=[
        ["the old man is sitting on the bench"],
        ["a new program will run soon"],
        ["she quickly read the book"]
    ]
)

if __name__ == "__main__":
    iface.launch()
