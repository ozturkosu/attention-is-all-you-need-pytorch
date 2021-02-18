''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, factorized_k, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        #self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # Factorized Weight Matrix for Q 
        self.W_A = nn.Parameter(torch.rand(d_model, factorized_k), requires_grad=True)
        self.W_B = nn.Parameter(torch.rand(factorized_k, d_model), requires_grad=True)

        # Factorized Weight Matrix for K
        self.W_A2 = nn.Parameter(torch.rand(d_model, factorized_k), requires_grad=True)
        self.W_B2 = nn.Parameter(torch.rand(factorized_k, d_model), requires_grad=True)

        
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        #q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        #k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        W_a = self.W_A.view(self.n_head, self.d_k,-1)
        W_b = self.W_A.view(self.n_head, -1, self.d_k)

        W_a2 = self.W_A2.view(self.n_head, self.d_k,-1)
        W_b2 = self.W_A2.view(self.n_head, -1, self.d_k)



        # Transpose for attention dot product: b x n x lq x dv
        #q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        v = v.transpose(1, 2)

        qt = torch.einsum("abc->acb", [q])
        qt = qt.view(sz_b, self.n_head, self.d_k, -1)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = q.view(sz_b, -1, self.n_head, self.d_k)

        out, attn = self.attention(q, W_a, W_b, W_a2, W_b2, qt, v, d_k, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        out = self.dropout(self.fc(out))
        out += residual

        out = self.layer_norm(out)

        return out, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    #Temperature is d 

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    #def forward(self, q, k, v, mask=None):
    def forward(self, q, W_A, W_B, W_At, W_Bt, qt, v, d_k, mask=None):

        print(" Q Matrix size")
        print(q.size())

        print("WA matrix size")
        print(W_A.size())

        #Calculate I * A
        IA = torch.einsum('kabc,bcd->kabd', [q, W_A] )

        print("IA Matrix size")
        print(IA.size())

        print("WB matrix size")
        print(W_B.size())


        #Calculate IA * B
        IAB = torch.einsum('kabj,bji->kabi', [IA, W_B] )

        print("IAB Matrix size")
        print(IAB.size())

        #Calculate IAB*Bt
        IABBt = torch.einsum('kabi,bim->kabm', [IAB, W_Bt])

        #print("IABBt Matrix size")
        #print(IABBt.size())

        #Calculate IABBt * At
        IABBtAt = torch.einsum('kabm,bmj->kabj' , [IABBt , W_At])

        #print("qt Matrix size")
        #print(qt.size())

        #k is batch size b is #heads
        # Score attention matrix
        attn = torch.einsum('kabj,kbjm->kbam' , [IABBtAt, qt])

        #attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # attn here is attention matrix
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
