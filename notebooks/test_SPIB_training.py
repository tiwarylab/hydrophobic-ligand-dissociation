"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import torch
import numpy as np
import time
import os
import constraints

# Data Processing
# ------------------------------------------------------------------------------

def data_init(t0, dt, traj_data, traj_label, Ut, traj_weights):
	assert len(traj_data)==len(traj_label)
	
	# skip the first t0 data
	past_data = traj_data[t0:(len(traj_data)-dt)]
	future_data = traj_data[(t0+dt):len(traj_data)]
	label = traj_label[(t0+dt):len(traj_data)]
	
	# data shape
	data_shape = past_data.shape[1:]
	
	n_data = len(past_data)
	
	# 90% random test/train split
	p = np.random.permutation(n_data)
	past_data = past_data[p]
	future_data = future_data[p]
	label = label[p]
	
	past_data_train = past_data[0: (9 * n_data) // 10]
	past_data_test = past_data[(9 * n_data) // 10:]
	
	future_data_train = future_data[0: (9 * n_data) // 10]
	future_data_test = future_data[(9 * n_data) // 10:]
	
	label_train = label[0: (9 * n_data) // 10]
	label_test = label[(9 * n_data) // 10:]

	if Ut == None:
		Ut = torch.zeros(len(traj_data))
		Ut_train = torch.zeros(len(past_data_train))
		Ut_test = torch.zeros(len(past_data_test))
	else:
		Ut_train = Ut[p[0: (9 * n_data) // 10]]
		Ut_test = Ut[p[(9 * n_data) // 10:]]
	
	if traj_weights != None:
		assert len(traj_data)==len(traj_weights)
		weights = traj_weights[t0:(len(traj_data)-dt)]
		weights = weights[p]
		weights_train = weights[0: (9 * n_data) // 10]
		weights_test = weights[(9 * n_data) // 10:]
	else:
		weights_train = None
		weights_test = None
	
	return data_shape, past_data_train, future_data_train, label_train, weights_train,\
		past_data_test, future_data_test, label_test, weights_test, Ut_train, Ut_test

# Loss function
# ------------------------------------------------------------------------------

def calculate_loss(IB, data_inputs, data_targets, data_weights, beta = 1.0, gamma = 1.0, Ut = None, penalty = None, b = False, output_thermo = False, bandwidth = 0.1):
	
	# pass through VAE
	data_inputs.requires_grad = True
	outputs, z_sample, z_mean, z_logvar = IB.forward(data_inputs)
	#print(z_mean.shape)

	if Ut == None:
		Ut = torch.zeros(len(data_inputs))
	
	# KL Divergence
    # sample from the encoder
	log_p = IB.log_p(z_sample)
    # Gaussian prior
	log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample-z_mean, 2)
							 /torch.exp(z_logvar), dim=1)
	
	if data_weights == None:
		# Reconstruction loss is cross-entropy
		reconstruction_error = torch.mean(torch.sum(-data_targets*outputs, dim=1))
		
		# KL Divergence
		kl_loss = torch.mean(log_q-log_p)
		
	else:
		# Reconstruction loss is cross-entropy
		# reweighed
		reconstruction_error = torch.mean(data_weights*torch.sum(-data_targets*outputs, dim=1))
		
		# KL Divergence
		kl_loss = torch.mean(data_weights*(log_q-log_p))
		
	# add an extra penalty for the `physical constraint' to the loss function
	if penalty == 'mean':
		aux_loss = constraints.mean(z_mean)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'var':
		aux_loss = constraints.var(z_mean)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'Hz':
		aux_loss = constraints.Hz(z_mean)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
#	elif penalty == 'dSz':
#		import util
#		Ut = util.temperature_switch_potential(data_inputs[:,0], data_inputs[:,1])
#		aux_loss = constraints.dSz(data_inputs, z_mean, Ut)
#		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	#elif penalty == 'dSz_path':
#		import util
		# hard-code for now
#		dummy = np.loadtxt('states.txt', dtype = int)
#		s1 = int(dummy[0])
#		s2 = int(dummy[1])
#		Ut = util.temperature_switch_potential(data_inputs[:,0], data_inputs[:,1])
#		if output_thermo:
#			aux_loss, G11, U11, S11 = constraints.dSz_path(data_inputs, z_mean, Ut, outputs, s1, s2, output_thermo = True)
#		else:
#			aux_loss = constraints.dSz_path(data_inputs, z_mean, Ut, outputs, s1, s2)
#		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'dSz_point':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.dSz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.dSz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.dSz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, weights = data_weights)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'dSz_point_diff':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.dSz_point_diff(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.dSz_point_diff(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.dSz_point_diff(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = False, weights = data_weights)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'dSz_point_both':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.dSz_point_both(data_inputs, z_mean, Ut, output_thermo = True)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.dSz_point_both(data_inputs, z_mean, Ut, output_thermo = True)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.dSz_point(data_inputs, z_mean, Ut)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
#	elif penalty == 'dUz':
#		import util
#		Ut = util.temperature_switch_potential(data_inputs[:,0], data_inputs[:,1])
#		aux_loss = constraints.dUz(data_inputs, z_mean, Ut)
#		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
#	elif penalty == 'dUz_path':
#		import util
		# hard-code for now
#		dummy = np.loadtxt('states.txt', dtype = int)
#		s1 = int(dummy[0])
#		s2 = int(dummy[1])
#		Ut = util.temperature_switch_potential(data_inputs[:,0], data_inputs[:,1])
#		if output_thermo:
#			aux_loss, G11, U11, S11 = constraints.dUz_path(data_inputs, z_mean, Ut, outputs, s1, s2, output_thermo = True)
#		else:
#			aux_loss = constraints.dSz_path(data_inputs, z_mean, Ut, outputs, s1, s2)
#
#		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'dUz_point':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.dUz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.dUz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.dUz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, weights = data_weights)
		#print(reconstruction_error, beta, kl_loss, gamma, aux_loss)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'dGz_point':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.dGz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.dGz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.dGz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, weights = data_weights)
		loss = reconstruction_error + beta*kl_loss - gamma * aux_loss
	elif penalty == 'variational':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
		else:
			aux_loss = constraints.variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, weights = data_weights)
		loss = reconstruction_error + beta*kl_loss + gamma * aux_loss
	elif penalty == 'abs_variational':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, weights = data_weights)
		loss = reconstruction_error + beta*kl_loss + gamma * aux_loss
	elif penalty == 'delta_variational':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.delta_variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
				#print(aux_loss)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.delta_variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
				#print(aux_loss)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.delta_variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, weights = data_weights)
			#print(data_inputs.shape, z_mean.shape, Ut.shape, data_weights)
		loss = reconstruction_error + beta*kl_loss + gamma * aux_loss
	elif penalty == 'logsumexp_variational':
		import util
		if output_thermo:
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.logsumexp_variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.logsumexp_variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				print("Not implemented yet")
		else:
			aux_loss = constraints.logsumexp_variational(data_inputs, z_mean, Ut, bandwidth = bandwidth, weights = data_weights)
		loss = reconstruction_error + beta*kl_loss + gamma * aux_loss
	else:
		if output_thermo:
			import util
			if z_mean.shape[1] == 1:
				aux_loss, G11, U11, S11 = constraints.dSz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			elif z_mean.shape[1] == 2:
				aux_loss, G11, U11, S11, G22, U22, S22 = constraints.dSz_point(data_inputs, z_mean, Ut, bandwidth = bandwidth, output_thermo = True, weights = data_weights)
			else:
				print("Not implemented yet")
		aux_loss = torch.tensor(0)
		loss = reconstruction_error + beta*kl_loss

	
	#print(loss,  reconstruction_error, beta*kl_loss, gamma * aux_loss)
	if output_thermo:
		if z_mean.shape[1] == 1:
			return loss, reconstruction_error.float(), kl_loss.float(), aux_loss.float(), G11, U11, S11
		else:
			return loss, reconstruction_error.float(), kl_loss.float(), aux_loss.float(), G11, U11, S11, G22, U22, S22
	else:
		#print(aux_loss.float)
		return loss, reconstruction_error.float(), kl_loss.float(), aux_loss.float()


# Train and test model
# ------------------------------------------------------------------------------

def sample_minibatch(past_data, data_labels, data_weights, indices, device):
	sample_past_data = past_data[indices].to(device)
	sample_data_labels = data_labels[indices].to(device)
	
	if data_weights == None:
		sample_data_weights = None
	else:
		sample_data_weights = data_weights[indices].to(device)
	
	
	return sample_past_data, sample_data_labels, sample_data_weights


def train(IB, beta, gamma, Ut_train, Ut_test, train_past_data, train_future_data, init_train_data_labels, train_data_weights, \
		  test_past_data, test_future_data, init_test_data_labels, test_data_weights, \
			  learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, batch_size, threshold, patience, refinements, output_path, log_interval, device, index, penalty, bandwidth = 0.1):
	IB.train()
	
	step = 0
	start = time.time()
	log_path = output_path + '_train.log'
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	IB_path = output_path + "cpt" + str(index) + "/IB"
	os.makedirs(os.path.dirname(IB_path), exist_ok=True)
	
	train_data_labels = init_train_data_labels
	test_data_labels = init_test_data_labels

	update_times = 0
	unchanged_epochs = 0
	epoch = 0

	# initial state population
	state_population0 = torch.sum(train_data_labels,dim=0).float()/train_data_labels.shape[0]

	# generate the optimizer and scheduler
	optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

	while True:
		
		train_permutation = torch.randperm(len(train_past_data))
		test_permutation = torch.randperm(len(test_past_data))
		
		
		for i in range(0, len(train_past_data), batch_size):
			step += 1
			
			if i+batch_size>len(train_past_data):
				break
			
			train_indices = train_permutation[i:i+batch_size]
			U_t = Ut_train[train_indices]

			batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, \
																	   train_data_weights, train_indices, device)
					
			loss, reconstruction_error, kl_loss, aux_loss = calculate_loss(IB, batch_inputs, \
																batch_outputs, batch_weights, beta, gamma = gamma, Ut = U_t, penalty = penalty, bandwidth = bandwidth) ; print(penalty, loss, reconstruction_error, kl_loss, aux_loss)
			
			# Stop if NaN is obtained
			if(torch.isnan(loss).any()):
				# re-calculate reconstruction error and KL loss explicitly
				outputs, z_sample, z_mean, z_logvar = IB.forward(batch_inputs)
				
				# KL Divergence
				log_p = IB.log_p(z_sample)
				log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample-z_mean, 2)
										 /torch.exp(z_logvar), dim=1)
				
				if batch_weights == None:
					# Reconstruction loss is cross-entropy
					reconstruction_error = torch.mean(torch.sum(-batch_outputs*outputs, dim=1))
					
					# KL Divergence
					kl_loss = torch.mean(log_q-log_p)
					
				else:
					# Reconstruction loss is cross-entropy
					# reweighed
					reconstruction_error = torch.mean(batch_weights*torch.sum(-batch_outputs*outputs, dim=1))
					
					# KL Divergence
					kl_loss = torch.mean(batch_weights*(log_q-log_p))
				print(loss, reconstruction_error, kl_loss, aux_loss)
				np.save(output_path + 'nans_batch_outputs.npy', batch_outputs.detach().numpy())
				np.save(output_path + 'nans_outputs.npy', outputs.detach().numpy())
				np.save(output_path + 'nans_z_sample.npy', z_sample.detach().numpy())
				np.save(output_path + 'nans_z_mean.npy', z_mean.detach().numpy())
				np.save(output_path + 'nans_z_logvar.npy', z_logvar.detach().numpy())
				return True
	
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
			
			if step % 500 == 0:
				#with torch.no_grad():
					
				batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, \
																		   train_data_weights, train_indices, device)
						
				loss, reconstruction_error, kl_loss, aux_loss = calculate_loss(IB, batch_inputs, \
																	batch_outputs, batch_weights, beta, gamma = gamma, Ut = U_t, penalty = penalty, bandwidth = bandwidth)
				train_time = time.time() - start
		
				print(
					"Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
					"Reconstruction loss (train) %f\t Auxillary loss (train) %f" % (
						step, train_time, loss, kl_loss, reconstruction_error, aux_loss))
				print(
				   "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
					"Reconstruction loss (train) %f\t Auxillary loss (train) %f" % (
						step, train_time, loss, kl_loss, reconstruction_error, aux_loss), file=open(log_path, 'a'))
				j=i%len(test_permutation)
				
				
				
				test_indices = test_permutation[j:j+batch_size]
				U_t = Ut_test[test_indices]
				
				batch_inputs, batch_outputs, batch_weights = sample_minibatch(test_past_data, test_data_labels, \
																		   test_data_weights, test_indices, device)
				
				loss, reconstruction_error, kl_loss, aux_loss = calculate_loss(IB, batch_inputs, \
																	 batch_outputs, batch_weights, beta, gamma = gamma, Ut = U_t, penalty = penalty, bandwidth = bandwidth)

				train_time = time.time() - start
				print(
				   "Loss (test) %f\tKL loss (test): %f\n"
				   "Reconstruction loss (test) %f\t Auxillary loss (test) %f" % (
					   loss, kl_loss, reconstruction_error, aux_loss))
				print(
				   "Loss (test) %f\tKL loss (test): %f\n"
				   "Reconstruction loss (test) %f\t Auxillary loss (test) %f" % (
					   loss, kl_loss, reconstruction_error, aux_loss), file=open(log_path, 'a'))
		
			if step % log_interval == 0:
				# save model
				torch.save({'step': step,
							'state_dict': IB.state_dict()},
						   IB_path+ '_%d_cpt.pt'%step)
				torch.save({'optimizer': optimizer.state_dict()},
						   IB_path+ '_%d_optim_cpt.pt'%step) 

		epoch+=1
		
		# check convergence
		if IB.UpdateLabel == True:
			new_train_data_labels = IB.update_labels(train_future_data, batch_size)
		else:
			new_train_data_labels = train_data_labels

		# save the state population
		state_population = torch.sum(new_train_data_labels,dim=0).float()/new_train_data_labels.shape[0]

		print(state_population)
		print(state_population, file=open(log_path, 'a'))

		# print the state population change
		state_population_change = torch.sqrt(torch.square(state_population-state_population0).sum())
		
		print('State population change=%f'%state_population_change)
		print('State population change=%f'%state_population_change, file=open(log_path, 'a'))

		# update state_population
		state_population0 = state_population

		scheduler.step()
		if scheduler.gamma < 1:
			print("Update lr to %f"%(optimizer.param_groups[0]['lr']))
			print("Update lr to %f"%(optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))

		# check whether the change of the state population is smaller than the threshold
		if state_population_change < threshold:
			unchanged_epochs += 1
			
			if unchanged_epochs > patience:

				# check whether only one state is found
				if torch.sum(state_population>0)<2:
					print("Only one metastable state is found!")
					break

				# Stop only if update_times >= refinements
				if IB.UpdateLabel and update_times < refinements:
					
					train_data_labels = new_train_data_labels
					test_data_labels = IB.update_labels(test_future_data, batch_size)
	
					update_times+=1
					print("Update %d\n"%(update_times))
					print("Update %d\n"%(update_times), file=open(log_path, 'a'))
					
					# reset epoch and unchanged_epochs
					epoch = 0
					unchanged_epochs = 0

					# reset the representative-inputs
					representative_inputs = IB.estimatate_representative_inputs(train_past_data, train_data_weights, batch_size)
					IB.reset_representative(representative_inputs.to(device))
	
					# reset the optimizer and scheduler
					optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

					scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
					
				else:
					break

		else:
			unchanged_epochs = 0

		print("Epoch: %d\n"%(epoch))
		print("Epoch: %d\n"%(epoch), file=open(log_path, 'a'))
		#print(loss, reconstruction_error, beta * kl_loss, gamma * aux_loss)

	# output the saving path
	total_training_time = time.time() - start
	print("Total training time: %f" % total_training_time)
	print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))
	# save model
	torch.save({'step': step,
				'state_dict': IB.state_dict()},
			   IB_path+ '_%d_cpt.pt'%step)
	torch.save({'optimizer': optimizer.state_dict()},
			   IB_path+ '_%d_optim_cpt.pt'%step)
	
	torch.save({'step': step,
				'state_dict': IB.state_dict()},
			   IB_path+ '_final_cpt.pt')
	torch.save({'optimizer': optimizer.state_dict()},
			   IB_path+ '_final_optim_cpt.pt')

	return False

#@torch.no_grad()
def output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
						test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, output_path, \
							path, dt, beta, gamma, Ut_train, Ut_test, learning_rate, penalty, index=0, output_thermo = False, bandwidth = 0.1):
	
	#with torch.no_grad():
	final_result_path = output_path + '_final_result' + str(index) + '.npy'
	os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
	if output_thermo:
		thermo_path = output_path + 'thermo/'
		os.makedirs(thermo_path, exist_ok = True)
	
	# label update
	if IB.UpdateLabel:
		train_data_labels = IB.update_labels(train_future_data, batch_size)
		test_data_labels = IB.update_labels(test_future_data, batch_size)
	
	final_result = []
	# output the result
	
	loss, reconstruction_error, kl_loss, aux_loss = [0 for i in range(4)]
	
	counter = 0
	for i in range(0, len(train_past_data), batch_size):
		indices = range(i,min(i+batch_size,len(train_past_data)))
		dummy = Ut_train[indices]
		batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, train_data_weights, \
																   indices, IB.device)
		if IB.z_dim == 1:
			if output_thermo:
					loss1, reconstruction_error1, kl_loss1, aux_loss1, G11, U11, S11 = calculate_loss(IB, batch_inputs, batch_outputs, \
																		batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																		output_thermo = output_thermo, bandwidth = bandwidth)
			else:
					loss1, reconstruction_error1, kl_loss1, aux_loss1 = calculate_loss(IB, batch_inputs, batch_outputs, \
																		batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																		output_thermo = output_thermo, bandwidth = bandwidth)
		else:
			if output_thermo:
					loss1, reconstruction_error1, kl_loss1, aux_loss1, G11, U11, S11, G22, U22, S22 = calculate_loss(IB, batch_inputs, batch_outputs, \
																	batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																	output_thermo = output_thermo, bandwidth = bandwidth)
			else:
				loss1, reconstruction_error1, kl_loss1, aux_loss1 = calculate_loss(IB, batch_inputs, batch_outputs, \
																	batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																	output_thermo = output_thermo, bandwidth = bandwidth)
		loss += loss1*len(batch_inputs)
		reconstruction_error += reconstruction_error1*len(batch_inputs)
		kl_loss += kl_loss1*len(batch_inputs)
		aux_loss += aux_loss1*len(batch_inputs)
		if output_thermo:
			if IB.z_dim == 1:
				np.save(thermo_path + 'batch' + str(counter) + '_G11.npy', G11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_U11.npy', U11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_S11.npy', S11.detach().numpy())
			else:
				np.save(thermo_path + 'batch' + str(counter) + '_G11.npy', G11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_U11.npy', U11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_S11.npy', S11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_G22.npy', G22.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_U22.npy', U22.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_S22.npy', S22.detach().numpy())
			counter += 1
	# output the result
	loss/=len(train_past_data)
	reconstruction_error/=len(train_past_data)
	kl_loss/=len(train_past_data)
	aux_loss /= len(train_past_data)
			
	final_result += [loss.data.cpu().numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy(), aux_loss.cpu().data.numpy()]
	print(
		"Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
				"Reconstruction loss (train) %f\tAuxillary loss (train) %f" % (
			index, loss, kl_loss, reconstruction_error, aux_loss))
	print(
		"Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
				"Reconstruction loss (train) %f\tAuxillary loss (train) %f" % (
			index, loss, kl_loss, reconstruction_error, aux_loss),
		file=open(path, 'a'))

	loss, reconstruction_error, kl_loss, aux_loss = [0 for i in range(4)]
	
	counter = 0
	for i in range(0, len(test_past_data), batch_size):
		indices = range(i,min(i+batch_size,len(test_past_data)))
		dummy = Ut_test[indices]
		batch_inputs, batch_outputs, batch_weights = sample_minibatch(test_past_data, test_data_labels, test_data_weights, \
																					 indices, IB.device)
		if IB.z_dim == 1:
			if output_thermo:
					loss1, reconstruction_error1, kl_loss1, aux_loss1, G11, U11, S11 = calculate_loss(IB, batch_inputs, batch_outputs, \
																		batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																		output_thermo = output_thermo, bandwidth = bandwidth)
			else:
					loss1, reconstruction_error1, kl_loss1, aux_loss1 = calculate_loss(IB, batch_inputs, batch_outputs, \
																		batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																		output_thermo = output_thermo, bandwidth = bandwidth)
		else:
			if output_thermo:
					loss1, reconstruction_error1, kl_loss1, aux_loss1, G11, U11, S11, G22, U22, S22 = calculate_loss(IB, batch_inputs, batch_outputs, \
																	batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																	output_thermo = output_thermo, bandwidth = bandwidth)
			else:
				loss1, reconstruction_error1, kl_loss1, aux_loss1 = calculate_loss(IB, batch_inputs, batch_outputs, \
																	batch_weights, beta, gamma = gamma, Ut = dummy, penalty = penalty, 
																	output_thermo = output_thermo, bandwidth = bandwidth)
		loss += loss1*len(batch_inputs)
		reconstruction_error += reconstruction_error1*len(batch_inputs)
		kl_loss += kl_loss1*len(batch_inputs)
		aux_loss += aux_loss1*len(batch_inputs)
		if output_thermo:
			if IB.z_dim == 1:
				np.save(thermo_path + 'batch' + str(counter) + '_G11_test.npy', G11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_U11_test.npy', U11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_S11_test.npy', S11.detach().numpy())
			else:
				np.save(thermo_path + 'batch' + str(counter) + '_G11_test.npy', G11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_U11_test.npy', U11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_S11_test.npy', S11.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_G22_test.npy', G22.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_U22_test.npy', U22.detach().numpy())
				np.save(thermo_path + 'batch' + str(counter) + '_S22_test.npy', S22.detach().numpy())
			counter += 1
		
	
	# output the result
	loss/=len(test_past_data)
	reconstruction_error/=len(test_past_data)
	kl_loss/=len(test_past_data)
	aux_loss /= len(test_past_data)
	
	final_result += [loss.cpu().data.numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy(), aux_loss.cpu().data.numpy()]
	print(
		"Loss (test) %f\tKL loss (test): %f\n"
		"Reconstruction loss (test) %f\tAuxillary loss (test) %f"
		% (loss, kl_loss, reconstruction_error, aux_loss))
	print( 
		"Loss (test) %f\tKL loss (test): %f\n"
		"Reconstruction loss (test) %f\tAuxillary loss (test) %f"
		% (loss, kl_loss, reconstruction_error, aux_loss), file=open(path, 'a'))
	
	print("dt: %d\t Beta: %f\t Learning_rate: %f" % (
		dt, beta, learning_rate))
	print("dt: %d\t Beta: %f\t Learning_rate: %f" % (
		dt, beta, learning_rate),
		  file=open(path, 'a'))    
	
	
	final_result = np.array(final_result)
	np.save(final_result_path, final_result)
