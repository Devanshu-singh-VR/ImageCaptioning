import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, hidden_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.vgg = models.vgg16(pretrained=False)
        self.vgg.classifier[6] = nn.Linear(4096, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.vgg(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, captions, hidden):
        captions = captions.unsqueeze(0)
        embeddings = self.dropout(self.embed(captions))
        output, hidden = self.gru(embeddings, hidden)
        outputs = self.linear(output)
        return outputs, hidden


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers, device):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(hidden_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, len(vocab.vocab), num_layers)
        self.vocab_size = len(vocab.vocab)
        self.vocab = vocab
        self.device = device

    def forward(self, images, captions):
        hidden = self.encoderCNN(images)
        hidden = hidden.unsqueeze(0)

        seq_len, batch_size = captions.shape
        outputs = torch.zeros((seq_len, batch_size, self.vocab_size)).to(self.device)

        x = captions[0]
        for t in range(1, seq_len):
            output, hidden = self.decoderRNN(x, hidden)

            outputs[t] = output
            x = captions[t]

        return outputs

    def image_to_caption(self, img, max_word=50):
        hidden = self.encoderCNN(img).unsqueeze(0)
        caption = []

        x = torch.tensor([0]).to(self.device)
        for i in range(max_word):
            output, hidden = self.decoderRNN(x, hidden)
            x = output.squeeze(0).argmax(1)
            caption.append(self.vocab.vocab.itos[int(x[0])])

            if self.vocab.vocab.itos[int(x[0])] == "<EOS>":
                break

        return caption



