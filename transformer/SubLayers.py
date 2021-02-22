''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        #self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.attention = ScaledDotProductAttention(math.sqrt(d_k))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        print("len q")
        print(len_q)
        print("d_k")
        print(d_k)

        print("len_k")
        print(len_k)

        print("len_v")
        print(len_v)
        print("d_v")
        print(d_v)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv

        #q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        #k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        #v = self.w_vs(q).view(sz_b, len_q, n_head, d_v)

        W_a = self.W_A.view(self.n_head, self.d_k,-1)
        W_b = self.W_A.view(self.n_head, -1, self.d_k)

        W_a2 = self.W_A2.view(self.n_head, -1 , self.d_k,)
        W_b2 = self.W_A2.view(self.n_head, self.d_k , -1)



        # Transpose for attention dot product: b x n x lq x dv
        #q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        v = v.transpose(1, 2)

        qt = torch.einsum("abc->acb", [q])
        qt = qt.view(sz_b, self.n_head, self.d_k, -1)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = q.view(sz_b, -1, self.n_head, self.d_k)
        #out, attn = self.attention(q, W_a, W_b, W_a2, W_b2, qt, v, d_k, mask=mask)
        q, attn = self.attention(q, W_a, W_b, W_a2, W_b2, qt, v, d_k, mask=mask)


        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        #out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        #out = out.transpose(1, 2).contiguous().view(sz_b, -1, len_q)

        #out = self.dropout(self.fc(out))
        q= self.dropout(self.fc(q))

        #out += residual
        q += residual

        #out = self.layer_norm(out)
        q = self.layer_norm(q)

        #return out, attn
        return q, attn


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

        print(" V Matrix size")
        print(v.size())

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

        print("W_Bt Matrix size")
        print(W_Bt.size())

        #Calculate IAB*Bt
        IABBt = torch.einsum('kabi,bim->kabm', [IAB, W_Bt])

        print("IABBt Matrix size")
        print(IABBt.size())

        print("W_At Matrix size")
        print(W_At.size())

        #Calculate IABBt * At
        IABBtAt = torch.einsum('kabm,bmj->kabj' , [IABBt , W_At])

        print("IABBtAt Matrix Size")
        print(IABBtAt.size())

        print("qt Matrix size")
        print(qt.size())

        print("V Matrix size")
        print(v.size())

        #k is batch size b is #heads
        # Score attention matrix
        attn = torch.einsum('kabj,kbjm->kbam' , [IABBtAt, qt])

        print("Atten Matrix size")
        print(attn.size())


        #attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = torch.div(attn, self.temperature )

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        print("Atten Matrix size After MAsk")
        print(attn.size())

        # attn here is attention matrix
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        print("Output Matrix")
        print(output.size())

        return output, attn
