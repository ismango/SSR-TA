import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from epoch_time import epoch_time
from torch.optim import Adam
from ssrta.model import Encoder, Decoder, Model
from ssrta.load_data import load_data

# load data
train_data_path = ''
test_data_path = ''
train_loader, test_loader, dtext, rtext, expert = load_data(train_data_path, test_data_path)

# parameters
soft = nn.Softmax(dim=0)
cla_criterion = nn.CrossEntropyLoss()
seq_criterion = nn.CrossEntropyLoss(ignore_index=dtext.vocab.TRG_PAD_IDX)
alpha_1 = 0.3
alpha_2 = 0.5
alpha_3 = 1 - (alpha_1 + alpha_2)

input_dim = len(dtext.vocab)
output_dim = len(rtext.vocab)
rec_pred_dim = len(expert.vocab)
emb_dim = 256
enc_hid_dim = 128
dec_hid_dim = 128
dropout = 0.5
cuda = True
device = torch.device("cuda:0" if cuda else "cpu")
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# model
encoder = Encoder(input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, dtext.vocab.TRG_PAD_IDX)
decoder = Decoder(output_dim, emb_dim, dec_hid_dim, dropout, dtext.vocab.TRG_PAD_IDX)
model = Model(encoder=encoder, decoder=decoder, rec_pred_dim=rec_pred_dim, dropout=dropout, device=device).to(device)


# objective_function
def objective_function(cla_out, cla_label, trg_pred, trg):
    cla_label = cla_label.repeat(10, 1)
    cla_loss = cla_criterion(cla_out.view(cla_out.shape[0] * cla_out.shape[1], cla_out.shape[2]),
                             cla_label.view(cla_label.shape[0] * cla_label.shape[1]))

    cla_out = soft(cla_out).transpose(0, 1)
    pos_loss = 0
    for i in range(cla_out.shape[0]):
        pos_loss += cla_out.shape[1] - len(set(cla_out[i].argmax(1).cpu().numpy().tolist()))
    pos_loss = pos_loss / cla_out.shape[0]

    seq_loss = seq_criterion(trg_pred, trg)

    return alpha_1 * cla_loss + alpha_2 * pos_loss + alpha_3 * seq_loss


def train(model, train_loader, test_loader, optimizer):
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        src, trg, label = map(lambda x: x.to(device), data)

        cla_out, pred = model(src, trg, 10)

        pred_dim = pred.shape[-1]

        trg = trg.transpose(0, 1)[1:].contiguous().view(-1)
        pred = pred[1:].view(-1, pred_dim)

        loss = objective_function(cla_out, label.to(torch.int64), pred, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            src, trg, label = map(lambda x: x.to(device), data)

            cla_out, pred = model(src, trg, 10, 0)

            pred_dim = pred.shape[-1]

            trg = trg.transpose(0, 1)[1:].contiguous().view(-1)
            pred = pred[1:].view(-1, pred_dim)

            loss = objective_function(cla_out, label.to(torch.int64), pred, trg)
            test_loss += loss.item()

    return train_loss / len(train_loader), test_loss / len(test_loader)


def get_mrr(pred):
    return sum(1 / i if i != 0 else 0 for i in pred) / len(pred)


def get_topk(pred, k):
    idx = [i for i in range(len(pred))]
    dic = sorted(dict(zip(idx, pred)).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return [t[0] for t in dic[:k]]


def evaluate():
    model = torch.load('./model.pt').to(device)
    model.eval()

    ks = []
    mrrs = []
    mstrs = []
    rrs = []

    for k in range(1, 11):
        mrr_data = []
        mstr_data = []
        rr_data = []

        soft = nn.Softmax(dim=0)

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                src, trg, label = map(lambda x: x.to(device), data)
                cla_out, _ = model(src, trg, k, 0)
                for i in range(label.shape[0]):
                    true = int(label[i].cpu().numpy())
                    topk = [get_topk(soft(cla_out[j][i]).cpu().numpy().tolist(), 1)[0] for j in range(k)]
                    pos_of_true = topk.index(true) + 1 if true in topk else 0
                    mrr_data.append(pos_of_true)
                    if pos_of_true != 0: mstr_data.append(pos_of_true)
                    rr_data.append(1) if pos_of_true != 0 else rr_data.append(0)
        ks.append(k + 1)
        mrr = get_mrr(mrr_data)
        mrrs.append(mrr)
        mstr = sum(mstr_data) / len(mstr_data)
        mstrs.append(mstr)
        rr = np.count_nonzero(np.array(rr_data)) / len(rr_data)
        rrs.append(rr)

    res = {'k': ks, 'mrr': mrrs, 'mstrs': mstrs, 'rrs': rrs}
    res = pd.DataFrame(res)
    res.to_csv('ssrta.csv')


epochs = 600

optimizer = Adam(model.parameters(), lr=1e-3)

best_valid_loss = float('inf')
for epoch in range(epochs):

    start_time = time.time()

    train_loss, test_loss = train(model, train_loader, test_loader, optimizer)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if test_loss < best_valid_loss:
        best_valid_loss = test_loss
        torch.save(model, './model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {test_loss:.3f}')
print('finish! ')

evaluate()
