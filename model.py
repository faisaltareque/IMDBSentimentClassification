import torch
   
class RNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.RNN(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text) # text = [batch size, sentence length] -> embedded = [batch size, sentence length, embedding dim]
        output, hidden = self.rnn(embedded) # output = [batch size, sent len, hid dim], hidden = [num_layer=1, batch size, hid dim]
        return self.fc(hidden[-1]) # hidden = [num_layer=2, batch size, hid dim] -> hidden[-1] = [batch size, hid dim] -> fc = [batch size, output dim] (taking the last layer hidden state)
    
class LSTM(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text) # text = [batch size, sentence length] -> embedded = [batch size, sentence length, embedding dim]
        output, (hidden, cell) = self.rnn(embedded) # output = [batch size, sent len, hid dim], hidden = [num_layer=1, batch size, hid dim], cell = [num_layer=1, batch size, hid dim]
        return self.fc(hidden[-1]) # hidden = [num_layer=1, batch size, hid dim] -> hidden.squeeze(0) = [batch size, hid dim]  -> fc = [batch size, output dim] (taking the last layer hidden state)