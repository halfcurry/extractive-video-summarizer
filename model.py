import torch
import torch.nn as nn
import torch.nn.functional as F


class MA_module(nn.Module):

	def __init__(self):
		super(MA_module, self).__init__()

		self.input_size = 1024 # cnn features size
		self.emebedding_size = 512

		self.A_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))
		self.B_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))
		self.U_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))


		self.final= nn.Linear(self.emebedding_size, 1)



  
	def forward(self, x): #x is of size (b,300,1024) as 300 is supposed to be the largest possible for our memory module

		A_embedding_matrix=self.A_embedding_matrix

		B_embedding_matrix=self.B_embedding_matrix

		a=torch.matmul(x,A_embedding_matrix) # (b,300,1024) , (1024,512) -> (b,300,512)

		b=torch.matmul(x,B_embedding_matrix) 

		u=torch.matmul(x,self.U_embedding_matrix) 


		a_t=torch.transpose(a,len(a.size())-1,len(a.size())-2) #(b,512,300)

	
		temp=torch.bmm(u,a_t)#(b,300,300) 

		#for every ui there is its multiplication with every ak which then softmax
		p_mat=nn.functional.softmax(temp, dim=(len(temp.size())-1))#apply on the last dim #(b,300,300) 



		output_vec=torch.bmm(p_mat,b) #(b,300,512)

		u_mod=u*output_vec #(b,300,512)

		score=self.final(u_mod) #(b,300,1)
		return score


class MA_module_multiple(nn.Module):

	def __init__(self,n_hops=1):
#         print("nhops_model:",n_hops)
		super(MA_module_multiple, self).__init__()

		self.input_size = 1024 # cnn features size
		self.emebedding_size = 512

		self.nhops=n_hops

	  	#embedding matrix for the video segments
		self.U_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size).float())


		self.memory_embedding_matrix= nn.ParameterList([nn.Parameter(torch.randn(self.input_size,self.emebedding_size).float()) for i in range(self.nhops+1)])


		self.final= nn.Linear(self.emebedding_size, 1)

  
	def forward(self, x): #x is of size (b,300,1024) as 300 is supposed to be the largest possible for our memory module

		
		u=torch.matmul(x,self.U_embedding_matrix) 
		

		for i in range(self.nhops):

			A_embedding_matrix=self.memory_embedding_matrix[i]

			B_embedding_matrix=self.memory_embedding_matrix[i+1]
			


			a=torch.matmul(x,A_embedding_matrix) # (b,300,1024) , (1024,512) -> (b,300,512)

			b=torch.matmul(x,B_embedding_matrix) 



			a_t=torch.transpose(a,len(a.size())-1,len(a.size())-2) #(b,512,300)

		
			temp=torch.bmm(u,a_t)#(b,300,300) 

			#for every ui there is its multiplication with every ak which then softmax
			p_mat=nn.functional.softmax(temp, dim=(len(temp.size())-1))#apply on the last dim #(b,300,300) 



			output_vec=torch.bmm(p_mat,b) #(b,300,512)

			u_mod=u*output_vec #(b,300,512)

			u=u_mod



		score=self.final(u_mod) #(b,300,1)
		return score



class MA_module_with_LSTM_out(nn.Module):

    def __init__(self):
        super(MA_module_with_LSTM_out, self).__init__()

        self.input_size = 1024 # cnn features size
        self.emebedding_size = 512

        self.A_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))
        self.B_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))
        self.U_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))


        self.final_lstm=nn.LSTM(self.emebedding_size, self.emebedding_size, batch_first=True, bidirectional=True)


        self.final= nn.Linear(self.emebedding_size*2, 1)


  
    def forward(self, x): #x is of size (b,300,1024) as 300 is supposed to be the largest possible for our memory module

#         print(x.type())
        self.h0=torch.zeros(2, 1, self.emebedding_size).cuda()
        self.c0=torch.zeros(2, 1, self.emebedding_size).cuda()


        A_embedding_matrix=self.A_embedding_matrix

        B_embedding_matrix=self.B_embedding_matrix

        a=torch.matmul(x,A_embedding_matrix) # (b,300,1024) , (1024,512) -> (b,300,512)

        b=torch.matmul(x,B_embedding_matrix) 

        u=torch.matmul(x,self.U_embedding_matrix) 


        a_t=torch.transpose(a,len(a.size())-1,len(a.size())-2) #(b,512,300)

    
        temp=torch.bmm(u,a_t)#(b,300,300) 

        #for every ui there is its multiplication with every ak which then softmax
        p_mat=nn.functional.softmax(temp, dim=(len(temp.size())-1))#apply on the last dim #(b,300,300) 



        output_vec=torch.bmm(p_mat,b) #(b,300,512)

        u_mod=u*output_vec #(b,300,512)

        otp, (hn, cn) = self.final_lstm(u_mod, (self.h0, self.c0))
        u_mod=otp
    

        score=self.final(u_mod) #(b,300,1)
        return score



class MA_module_with_LSTM_out_1(nn.Module):

    def __init__(self):
        super(MA_module, self).__init__()

        self.input_size = 1024 # cnn features size
        self.emebedding_size = 512

        self.A_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))
        self.B_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))
        self.U_embedding_matrix=nn.Parameter(torch.randn(self.input_size,self.emebedding_size))


        self.final_lstm=nn.LSTM(self.emebedding_size, self.emebedding_size, batch_first=True, bidirectional=True)


        self.final= nn.Linear(self.emebedding_size*2, 1)
        self.h0=torch.zeros(2, 1, self.emebedding_size).cuda()
        self.c0=torch.zeros(2, 1, self.emebedding_size).cuda()



  
    def forward(self, x): #x is of size (b,300,1024) as 300 is supposed to be the largest possible for our memory module

        
       
        A_embedding_matrix=self.A_embedding_matrix

        B_embedding_matrix=self.B_embedding_matrix

        a=torch.matmul(x,A_embedding_matrix) # (b,300,1024) , (1024,512) -> (b,300,512)

        b=torch.matmul(x,B_embedding_matrix) 

        u=torch.matmul(x,self.U_embedding_matrix) 


        a_t=torch.transpose(a,len(a.size())-1,len(a.size())-2) #(b,512,300)

    
        temp=torch.bmm(u,a_t)#(b,300,300) 

        #for every ui there is its multiplication with every ak which then softmax
        p_mat=nn.functional.softmax(temp, dim=(len(temp.size())-1))#apply on the last dim #(b,300,300) 



        output_vec=torch.bmm(p_mat,b) #(b,300,512)

        u_mod=u*output_vec #(b,300,512)

        otp, (hn, cn) = self.final_lstm(u_mod, (self.h0, self.c0))
        u_mod=otp
        

        score=self.final(u_mod) #(b,300,1)
        return score



#Attention
MAX_LENGTH=500
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

