import torch
import torch.nn as nn
import torch.nn.functional as F
from grammar import zinc_grammar as G
from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self, input_size=100, hidden_n=100, output_feature_size=12, max_seq_length=15):
        super(Decoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_n = hidden_n
        self.output_feature_size = output_feature_size
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc_input = nn.Linear(input_size, hidden_n)

        self.gru_1 = nn.GRU(input_size=input_size,
                            hidden_size=hidden_n, batch_first=True)
        self.gru_2 = nn.GRU(input_size=hidden_n,
                            hidden_size=hidden_n, batch_first=True)
        self.gru_3 = nn.GRU(input_size=hidden_n,
                            hidden_size=hidden_n, batch_first=True)
        self.fc_out = nn.Linear(hidden_n, output_feature_size)

    def forward(self, encoded, hidden_1, hidden_2, hidden_3, beta=0.3, target_seq=None):
        _batch_size = encoded.size()[0]

        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(_batch_size, 1, -1) \
            .repeat(1, self.max_seq_length, 1)
        out_1, hidden_1 = self.gru_1(embedded, hidden_1)
        out_2, hidden_2 = self.gru_2(out_1, hidden_2)

        if self.training and target_seq:
            out_2 = out_2 * (1 - beta) + target_seq * beta
        out_3, hidden_3 = self.gru_3(out_2, hidden_3)
        out = self.fc_out(out_3.contiguous().view(-1, self.hidden_n)).view(_batch_size, self.max_seq_length,
                                                                           self.output_feature_size)
        return F.relu(F.sigmoid(out)), hidden_1, hidden_2, hidden_3

    def init_hidden(self, batch_size):
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h2 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h3 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        return h1, h2, h3


class Encoder(nn.Module):
    def __init__(self, k1=2, k2=3, k3=4, hidden_n=100, channels=12, linear_mult=9):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=k1, groups=channels)
        self.bn_1 = nn.BatchNorm1d(channels)
        self.conv_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=k2, groups=channels)
        self.bn_2 = nn.BatchNorm1d(channels)
        self.conv_3 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=k3, groups=channels)
        self.bn_3 = nn.BatchNorm1d(channels)

        self.fc_0 = nn.Linear(channels * linear_mult, hidden_n)
        self.fc_mu = nn.Linear(hidden_n, hidden_n)
        self.fc_var = nn.Linear(hidden_n, hidden_n)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x_ = x.view(batch_size, -1)
        h = self.fc_0(x_)
        return self.fc_mu(h), self.fc_var(h)


class EncoderZinc(nn.Module):
    def __init__(self, hidden_n=100):
        super(EncoderZinc, self).__init__()

        self.conv_1 = nn.Conv1d(76,9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 10, kernel_size=9)
        self.conv_3 = nn.Conv1d(10, 11, kernel_size=11)

        # self.fc_0 = nn.Linear(12 * 9, hidden_n)
        self.fc_0 = nn.Linear(2761, hidden_n)
        self.fc_mu = nn.Linear(hidden_n, hidden_n)
        self.fc_var = nn.Linear(hidden_n, hidden_n)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x_ = x.view(batch_size, -1)
        h = self.fc_0(x_)

        return self.fc_mu(h), self.fc_var(h)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False

    def forward(self, x, mu, log_var, recon_x):
        """gives the batch normalized Variational Error."""

        # recon_x = self.mask(x, recon_x)
        batch_size = x.size()[0]
        BCE = self.bce_loss(recon_x, x)

        KLD_element = mu.pow(2).add_(
            log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return (BCE + KLD) / batch_size

    def mask(self, x, recon_x):
        most_likely = torch.argmax(x, axis=-1)
        most_likely = most_likely.view(-1)
        ix2 = torch.unsqueeze(torch.tensor(G.ind_of_ind)[most_likely], 1).int()
        M2 = gather_nd(torch.tensor(G.masks),ix2)
        M3 = torch.reshape(M2, (-1, 15, 12))  # reshape them -1,15,12
        # apply them to the exp-predictions
        P2 = torch.mul(torch.exp(recon_x), M3)
        # normalize predictions
        P2 = torch.div(P2, torch.sum(P2, keepdim=True))
        return P2

def gather_nd(params, indices):

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

class GrammarVariationalAutoEncoder(nn.Module):
    def __init__(self, model_name='eq_grammar', rules=None):
        super(GrammarVariationalAutoEncoder, self).__init__()
        if rules is None:
            rules = []
        if model_name == 'zinc_grammar':
            self.encoder = EncoderZinc(hidden_n=56)
            self.decoder = Decoder(input_size=56, hidden_n=56, output_feature_size=len(rules), max_seq_length=277)
        elif model_name == 'eq_grammar':
            self.encoder = Encoder()
            self.decoder = Decoder()
        else:
            self.encoder = Encoder(channels=54, linear_mult=204)
            self.decoder = Decoder(output_feature_size=54, max_seq_length=210)

    def forward(self, x):
        batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        return output, mu, log_var

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
