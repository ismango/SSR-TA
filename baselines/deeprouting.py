import torch.nn as nn
import torch
from ssrta.load_data import load_data

LETTER_GRAM_SIZE = 3
WINDOW_SIZE = 3
TOTAL_LETTER_GRAMS = int(3 * 1e4)
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS
K = 300
L = 128
J = 4
FILTER_LENGTH = 1
TICKET_SIZE = 0  # len(Ticket.vocab)
GROUP_SIZE = 0  # len(Group.vocab)
EMBEDDING_DIM = 300
alpha = 0.5

'''
the text-view input embedding as SSR-TA
the graph-view input was trained using pygcn https://github.com/tkipf/pygcn
'''

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class DeepRouting(nn.Module):
    def __init__(self):
        super(DeepRouting, self).__init__()
        self.ticket_em = nn.Embedding(TICKET_SIZE, EMBEDDING_DIM)
        self.groupt_em = nn.Embedding(GROUP_SIZE, EMBEDDING_DIM)
        # layers for ticket
        self.ticket_conv = nn.Conv1d(EMBEDDING_DIM, K, FILTER_LENGTH)
        self.ticket_sem = nn.Linear(K, L)
        # layers for groupt
        self.groupt_conv = nn.Conv1d(EMBEDDING_DIM, K, FILTER_LENGTH)
        self.groupt_sem = nn.Linear(K, L)
        # layers for groupg
        self.groupg_sem = nn.Linear(K, L)
        # sim for ticket and groupt
        self.text_sim = nn.CosineSimilarity(dim=1)
        # sim for ticket and groupg
        self.graph_sim = nn.CosineSimilarity(dim=1)

    def forward(self, ticket, groupt, groupg):
        # ticket: ticket description groupt: text view of group groupg: graph view of group
        batch = ticket.shape[0]
        ticket = self.ticket_em(ticket)
        ticket = ticket.transpose(1, 2)
        ticket_c = torch.tanh(self.ticket_conv(ticket))
        ticket_k = kmax_pooling(ticket_c, 2, 1)
        ticket_k = ticket_k.transpose(1, 2)
        ticket_s = torch.tanh(self.ticket_sem(ticket_k))
        ticket_s = ticket_s.reshape(batch, L)

        groupt = self.groupt_em(groupt)
        groupt = groupt.transpose(1, 2)
        groupt_c = torch.tanh(self.groupt_conv(groupt))
        groupt_k = kmax_pooling(groupt_c, 2, 1)
        groupt_k = groupt_k.transpose(1, 2)
        groupt_s = torch.tanh(self.groupt_sem(groupt_k))
        groupt_s = groupt_s.reshape(batch, L)

        groupg_s = self.groupg_sem(groupg)

        sim = alpha * self.text_sim(ticket_s, groupt_s) + (1 - alpha) * self.graph_sim(ticket_s, groupg_s)
        return sim