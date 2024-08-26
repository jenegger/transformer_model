#!/usr/bin/env python3
import math
import torch
from torch import nn
import torch.nn.functional as F

class transformer_model(nn.Module):
	def __init__(self, k, heads=8, mask=False):
		super().__init__()
		assert k % heads == 0
		self.k, self.heads = k, heads
		self.tokeys    = nn.Linear(k, k, bias=False)
		self.toqueries = nn.Linear(k, k, bias=False)
		self.tovalues  = nn.Linear(k, k, bias=False)
		self.unifyheads = nn.Linear(k, k)
		self.linear_embedding = torch.nn.Linear(4,k)

	def forward(self, x,in_hitnr):
		x = self.linear_embedding(x)	
		b,t, k = x.size()
		h = self.heads
		queries = self.toqueries(x)
		keys    = self.tokeys(x)
		values  = self.tovalues(x)
		s = k // h
		keys    = keys.view(b,t, h, s)
		queries = queries.view(b,t, h, s)
		values  = values.view(b,t, h, s)
		# - fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b*h, t, s)
		queries = queries.transpose(1, 2).contiguous().view(b*h, t, s)
		values = values.transpose(1, 2).contiguous().view(b*h, t, s)
		# Get dot product of queries and keys, and scale
		dot = torch.bmm(queries, keys.transpose(1, 2))	
		dot = dot / (s ** (1/2))
		#dot = F.softmax(dot,dim=2)
		out = torch.bmm(dot, values).view(b,h, t, s) 
		out = out.transpose(1,2).contiguous().view(b,t, s * h)
		out = self.unifyheads(out)
		#use cosine similarity
		L2_dist = torch.cosine_similarity(out[:,None] , out[:,:,None],dim=-1)		
		L2_dist = 0.5*(L2_dist+1)
		upper_tri_mask = torch.triu(torch.ones((out.shape[1],out.shape[1])),diagonal=1).bool() #out[1] is max hit number in batch 
		return L2_dist[:,upper_tri_mask]




class transformer_model_extended(nn.Module):
	def __init__(self, features, heads=8, mask=False):
		super().__init__()
		self.features, self.heads = features, heads
		self.linear_embedding = torch.nn.Linear(4,self.features)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.features, nhead=self.heads)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
		self.activation = torch.nn.ReLU()
		self.additional_linear_layer = torch.nn.Linear(self.features,self.features)
		

	def forward(self, x,in_hitnr):
		x = self.linear_embedding(x)	
		#x = self.activation(x)  #put in some relu function to test....	
		#x = self.additional_linear_layer(x) #additional linear layer
		out = self.transformer_encoder(x)	
		#out = self.additional_linear_layer(out)
		#out = self.activation(out)
		#out = self.additional_linear_layer(out)
		#use cosine similarity
		L2_dist = torch.cosine_similarity(out[:,None] , out[:,:,None],dim=-1)		
		L2_dist = 0.5*(L2_dist+1)
		upper_tri_mask = torch.triu(torch.ones((out.shape[1],out.shape[1])),diagonal=1).bool() #out[1] is max hit number in batch 
		#return L2_dist[:,upper_tri_mask]
		ret_val = L2_dist[:,upper_tri_mask]
		#out_ret_val = torch.where(ret_val > 0.7, torch.tensor(1), torch.tensor(0)).float()
		out_ret_val = torch.where(ret_val > 0.7, torch.tensor(1,requires_grad=True), torch.tensor(0,requires_grad=True)).float()
		print(type(out_ret_val))
		print(out_ret_val.shape)
		return out_ret_val





















