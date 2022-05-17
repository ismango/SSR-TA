import torch
import random
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, padding_idx):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        embedded = self.dropout(self.embedding(src))

        enc_output, enc_hidden = self.rnn(embedded)

        z = torch.tanh(self.fc(torch.cat((enc_hidden[-1, :, :], enc_hidden[-2, :, :]), dim=1)))

        return z, enc_output


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, dropout, padding_idx):
        super().__init__()
        self.output_dim = output_dim
        self.dec_hid_dim = dec_hid_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, z):
        dec_input = dec_input.unsqueeze(1)

        embedded = self.dropout(self.embedding(dec_input))

        dec_output, dec_hidden = self.rnn(embedded, z.unsqueeze(0))

        pred = self.fc_out(dec_hidden.squeeze(0))

        return pred, dec_hidden.squeeze(0)


class Model(nn.Module):
    def __init__(self, encoder, decoder, rec_pred_dim, dropout, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.attn = nn.Linear(rec_pred_dim + self.encoder.enc_hid_dim * 2, self.decoder.dec_hid_dim, bias=False)
        self.vv = nn.Linear(self.decoder.dec_hid_dim, 1, bias=False)
        self.fc = nn.Linear(self.encoder.enc_hid_dim * 2, self.decoder.dec_hid_dim)
        self.output = nn.Sequential(
            nn.Linear(self.decoder.dec_hid_dim, self.decoder.dec_hid_dim),
            nn.Dropout(dropout),
            torch.nn.ReLU(),
            nn.Linear(self.decoder.dec_hid_dim, rec_pred_dim)
        )

    def forward(self, src, trg, k, teacher_forcing_ratio=0.5):

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        pred = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # z = [batch_size, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim*2]
        z, enc_output = self.encoder(src)
        src_len = enc_output.shape[1]

        for i in range(k):
            if i == 0:
                # rec_pred = [batch_size, rec_pred_dim]
                rec_pred = self.output(z)
                rec_pred_all = rec_pred.unsqueeze(0)
            else:
                # rec_pred = [batch_size, src_len, pred_dim]
                rec_pred = rec_pred.unsqueeze(1).repeat(1, src_len, 1)
                attn = self.attn(torch.cat((rec_pred, enc_output), dim=2))
                # weights = [batch_size, src_len]
                a_weights = self.vv(attn).squeeze(2)

                # weights = [batch_size, 1, src_len]
                weights = a_weights.unsqueeze(1)
                # temp_z = [batch_size, enc_hid_dim*2]
                temp_z = torch.bmm(weights, enc_output).squeeze(1)
                # [batch_size, dec_hid_dim]
                z = self.fc(temp_z)

                rec_pred = self.output(z)
                rec_pred_all = torch.cat((rec_pred_all, rec_pred.unsqueeze(0)))

        dec_input = trg[:, 0]

        for t in range(1, trg_len):
            dec_output, z = self.decoder(dec_input, z)

            pred[t] = dec_output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = dec_output.argmax(1)

            dec_input = trg[:, t] if teacher_force else top1
        return rec_pred_all, pred