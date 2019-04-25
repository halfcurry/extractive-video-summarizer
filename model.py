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