import torch
from torch import nn


def load_network(device, architecture='cnn_feed_lstm', path=None, **cnn_lstm_params):

    num_features = cnn_lstm_params['num_features']
    bidirectional = cnn_lstm_params['bidirectional']

    # initial a cnn-lstm network
    if architecture == 'cnn_feed_lstm':
        model = CNN_feed_LSTM(bidirectional=bidirectional,
                              num_features=num_features)
        model = model.to(device)
    elif architecture == 'cnn_concat_lstm':
        model = CNN_concat_LSTM(bidirectional=bidirectional,
                                num_features=num_features)
        model = model.to(device)
    else:
        raise ValueError("Unsupported network architecture.")

    # if exists, load the pre-trained model
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    return model


def test_forward_pass():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn_lstm_params = dict()
    cnn_lstm_params['num_features'] = 32
    cnn_lstm_params['bidirectional'] = True
    model = load_network(device, architecture='cnn_concat_lstm', **cnn_lstm_params)

    signal_length = 1000
    no_channels = 1
    batch_size = 10
    example = torch.rand((batch_size, no_channels, signal_length)).to(device)

    features = model.get_features(example)
    print(features.size())

    result = model(example)
    print(result.size())


def run():
    test_forward_pass()


class BiLSTM(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, n_layers=2, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_directions = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.hidden_size,
                             num_layers=n_layers,
                             dropout=.2,
                             batch_first=True,  # (batch, seq, feature)
                             bidirectional=bidirectional)

    def forward(self, x):
        x, _ = self.lstm1(x)  # x: (batch, seq_len, num_directions * hidden_size)
        x = x[:, -1, :]
        return x


# extract features of ecg-signals
class CNN1d(nn.Module):
    def __init__(self, output_size):
        super(CNN1d, self).__init__()  # (batch, 1, 1000)
        self.output_size = output_size

        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 50, 5),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.Dropout(p=0.5))  # (batch, 16, 95)
        self.conv2 = nn.Sequential(nn.Conv1d(16, 64, 45, 5),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2),
                                   nn.Dropout(p=0.5))  # (batch, 64, 5)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5))  # (batch, 128, 3)
        self.conv4 = nn.Sequential(nn.Conv1d(128, self.output_size, 3, 1),
                                   nn.BatchNorm1d(self.output_size),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5))  # (batch, 256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view(-1, 1, self.output_size)


class CNN_feed_LSTM(nn.Module):  # input -> CNN1d -> BiLSTM -> output
    def __init__(self, bidirectional=True, num_features=32):
        super(CNN_feed_LSTM, self).__init__()
        self.input_size = 256  # extract 256 features from each sample using CNN1d
        self.hidden_size = 100
        self.output_size = 4  # corresponding to 4 classes
        self.n_layers = 2
        self.n_directions = 2 if bidirectional else 1
        self.num_features = num_features

        # feature extraction
        self.cnn = CNN1d(self.input_size)
        self.lstm = BiLSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers)
        self.feature_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_size * self.n_directions, self.num_features),
            nn.Sigmoid(),
        )

        # final prediction
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.num_features, self.output_size),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        output = x

        # extract features
        output = self.get_features(output)

        # final predict
        output = self.classifier(output)

        return output

    def get_features(self, x):
        output = x

        output = self.cnn(output)
        output = self.lstm(output)
        output = self.feature_layer(output)
        return output


class CNN_concat_LSTM(nn.Module):  # input -> CNN1d + BiLSTM -> output
    def __init__(self, bidirectional=True, num_features=32):
        super(CNN_concat_LSTM, self).__init__()
        self.input_size = 1000
        self.output_size = 256
        self.hidden_size = 128
        self.n_classes = 4  # corresponding to 4 classes
        self.n_layers = 2
        self.n_directions = 2 if bidirectional else 1
        self.num_features = num_features

        # feature extraction
        self.cnn = CNN1d(self.output_size)
        self.lstm = BiLSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers)

        self.feature_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.output_size + self.hidden_size * self.n_directions, self.num_features),
            nn.Sigmoid(),
        )

        # final prediction
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.num_features, self.n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        output = x

        # extract features
        output = self.get_features(output)

        # final predict
        output = self.classifier(output)

        return output

    def get_features(self, x):
        output = x

        output_cnn = self.cnn(output)
        output_cnn = output_cnn[:, -1, :]
        output_lstm = self.lstm(output)

        output = torch.cat((output_cnn, output_lstm), dim=1)

        output = self.feature_layer(output)
        return output


if __name__ == "__main__":
    run()



