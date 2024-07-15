import torch
import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_shaping import create_data
from transform_model import transformer_model
from matplotlib import pyplot as plt
#1) open raw data .txt file
#2) select 3 events fill them in some kind of vector and reshuffle them
#the tensor should have columns: E, theta, phi, time, clusternr
#3) create according to the number of hits a matrix with zeros and ones, zero, if they do not belong to the same cluster, one if yes
#4) make an upper triangle matrix out of it and write it in plain line to file
#5) write also the selected hits with E, theta, phi,time into one file


class CustomDataset(Dataset):
    def __init__(self):
        self.d_set, self.target,self.hit_nr = create_data()
        assert  len(self.d_set) == len(self.target)

    def __len__(self):
        return len(self.d_set)

    def __getitem__(self, idx):
        event = self.d_set[idx]
        mask = self.target[idx]
        hit_nr = self.hit_nr[idx]
        return event,mask,hit_nr

def dynamic_length_collate(batch):
	list_of_lists = list(map(list, batch))
	in_data = [sublist[0] for sublist in list_of_lists]
	in_target = [sublist[1] for sublist in list_of_lists]
	in_hitnr = [sublist[2] for sublist in list_of_lists]
	nr_max_hits = max(in_hitnr)
	out_data = []
	out_target = []
	for in_data,in_target,in_hitnr in batch:
		pad_nr = nr_max_hits - in_hitnr
		zero_array_data = np.zeros((pad_nr,in_data.shape[1]))
		result_data = np.concatenate((in_data, zero_array_data), axis=0)
		zero_array_target = np.zeros((nr_max_hits,nr_max_hits))
		result_target = np.pad(in_target, ((0, nr_max_hits - in_target.shape[0]), (0, nr_max_hits - in_target.shape[1])), mode='constant', constant_values=0)		
		out_data.append(result_data)
		out_target.append(result_target)

	t_out_data = torch.from_numpy(np.array(result_data)).float()
	t_out_target = torch.from_numpy(np.array(result_target)).float()
	return t_out_data,t_out_target,in_hitnr
		
		



dataset = CustomDataset()
dloader = DataLoader(dataset,batch_size=8,shuffle=False,collate_fn=dynamic_length_collate)

# Train the model
dtype = torch.float32
transformer_model = transformer_model(32)
n_epochs = 50
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/8)
print (total_samples,n_iterations)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(transformer_model.parameters(), lr=3e-4)
transformer_model.train()
l_loss = []
for epoch in range(n_epochs):
	#for X_batch,target,in_hitnr in dloader:
	for i,(X_batch,target,in_hitnr) in enumerate(dloader):
		if (i+1) % 100 == 0:
			print(f"epoch {epoch+1}/{n_epochs}, step {i+1}/{n_iterations}")
		y_pred = transformer_model(X_batch,in_hitnr)
		y_true = target[:in_hitnr,:in_hitnr]
		upper_tri_mask = torch.triu(torch.ones((in_hitnr,in_hitnr)),diagonal=1).bool()
		y_true = y_true[upper_tri_mask]
		loss  = loss_fn(y_pred,y_true)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		l_loss.append(loss.item())

plt.title("Loss functions")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(l_loss)
plt.show()

		
