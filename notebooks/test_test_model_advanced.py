"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import numpy as np
import torch
import os
import sys
import configparser
import json

import SPIB
import test_SPIB_training as SPIB_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")


def test_model_advanced():
    # Settings
    # ------------------------------------------------------------------------------
    

    # If there is a configuration file, import the configuration file
    # Otherwise, an error will be reported
    if '-config' in sys.argv:
        config = configparser.ConfigParser(allow_no_value=True)

        config.read(sys.argv[sys.argv.index('-config') + 1])

        # By default, we save all the results in subdirectories of the following path.
        base_path = str(config.get("Path","base_path"))
        print(base_path)
        
        # Model parameters
        # Time delay delta t in terms of # of minimal time resolution of the trajectory data
        dt_list = json.loads(config.get("Model Parameters","dt"))
        
        
        # By default, we use all the all the data to train and test our model
        t0 = 0 
        
        # Dimension of RC or bottleneck
        RC_dim_list = json.loads(config.get("Model Parameters","d"))
        
        # Encoder type ('Linear' or 'Nonlinear')
        if config.get("Model Parameters","encoder_type")=='Nonlinear':
            encoder_type = 'Nonlinear'
        else:
            encoder_type = 'Linear'

        # Number of nodes in each hidden layer of the encoder
        neuron_num1_list = json.loads(config.get("Model Parameters","neuron_num1"))
        # Number of nodes in each hidden layer of the encoder
        neuron_num2_list = json.loads(config.get("Model Parameters","neuron_num2"))
        
        
        # Training parameters
        batch_size = int(config.get("Training Parameters","batch_size"))

        # Threshold in terms of the change of the predicted state population for measuring the convergence of training
        threshold = float(config.get("Training Parameters","threshold"))

        # Number of epochs with the change of the state population smaller than the threshold after which this iteration of training finishes
        patience = int(config.get("Training Parameters","patience"))

        # Minimum refinements
        refinements = int(config.get("Training Parameters","refinements"))
            
        # By default, we save the model every 10000 steps
        log_interval = int(config.get("Training Parameters","log_interval"))
        
        # Period of learning rate decay
        lr_scheduler_step_size = int(config.get("Training Parameters","lr_scheduler_step_size"))

        # Multiplicative factor of learning rate decay. Default: 1 (No learning rate decay)
        lr_scheduler_gamma = float(config.get("Training Parameters","lr_scheduler_gamma"))

        # learning rate of Adam optimizer
        learning_rate_list = json.loads(config.get("Training Parameters","learning_rate"))
        
        # Hyper-parameter beta
        beta_list = json.loads(config.get("Training Parameters","beta"))

        # Hyper-parameter gamma
        gamma_list = json.loads(config.get("Training Parameters","gamma"))

        # Bandwidth for KDE estimation of the free energy and energy
        try:
            bandwidth = float(config.get("Training Parameters","bandwidth"))
        except NoSectionError:
            print("Bandwidth for KDE not specified. Setting bandwidth = 0.1")
            bandwidth = 0.1
        print('bandwidth is ', bandwidth)

        ## trajectory of the potential energy for the system
        #energy_file = str(config.get("Training Parameters","energy_file"))
        #Ut = np.loadtxt(energy_file)

        # Type of physical constraint for the loss function
        try:
            penalty = str(config.get("Training Parameters","constraint"))
        except NoSectionError:
            penalty = None
        print(penalty)
        
        # Import data

        # Path to the trajectory data
        traj_data_path = config.get("Data","traj_data")
        traj_data_path = traj_data_path.replace('[','').replace(']','')
        traj_data_path = traj_data_path.split(',')

        # Load the data
        traj_data_list = [torch.from_numpy(np.load(file_path)).float().to(default_device) for file_path in traj_data_path]

        
        # Path to the initial state labels
        initial_labels_path = config.get("Data","initial_labels")
        initial_labels_path = initial_labels_path.replace('[','').replace(']','')
        initial_labels_path = initial_labels_path.split(',')
        
        traj_labels_list = [torch.from_numpy(np.load(file_path)).float().to(default_device) for file_path in initial_labels_path]
        
        output_dim = traj_labels_list[0].shape[1]
        
        assert len(traj_data_list)==len(traj_labels_list)

        # Path to the trajectory of the potential
        energy_path = config.get("Data","energy_file")
        energy_path = energy_path.replace('[','').replace(']','')
        energy_path = energy_path.split(',')

        # Load the data
        energy_list = [torch.from_numpy(np.loadtxt(file_path)).float().to(default_device) for file_path in energy_path]

        assert len(traj_data_list)==len(energy_list)

        # Path to the weights of the samples
        traj_weights_path = config.get("Data","traj_weights")
        if traj_weights_path == None:
            traj_weights_list = None
            IB_path = os.path.join(base_path, "Unweighted")
        else:
            traj_weights_path = traj_weights_path.replace('[','').replace(']','')
            traj_weights_path = traj_weights_path.split(',')
        
            traj_weights_list = [torch.from_numpy(np.load(file_path)).float().to(default_device) for file_path in traj_weights_path]
            IB_path = os.path.join(base_path, "Weighted")
            assert len(traj_weights_list)==len(traj_labels_list)

        
        # Other controls

        # Random seed
        seed_list = json.loads(config.get("Other Controls","seed"))

        # Whether to refine the labels during the training process
        if config.get("Other Controls","UpdateLabel") == 'True':
            UpdateLabel = True
        else:
            UpdateLabel = False
        
        # Whether save trajectory results
        if config.get("Other Controls","SaveTrajResults") == 'True':
            SaveTrajResults = True
        else:
            SaveTrajResults = False

    else:
        print("Please input the config file!")
        return

    
    
    # Train and Test our model
    # ------------------------------------------------------------------------------

    final_result_path = IB_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    print("Final Result", file=open(final_result_path, 'w'))
    
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)

        for dt in dt_list:
            data_init_list = [] 
            if traj_weights_list == None:
                for i in range(len(traj_data_list)):
                    data_init_list+=[SPIB_training.data_init(t0, dt, traj_data_list[i], traj_labels_list[i], energy_list[i], None)]
                train_data_weights = None
                test_data_weights = None
            else:
                for i in range(len(traj_data_list)):
                    data_init_list+=[SPIB_training.data_init(t0, dt, traj_data_list[i], traj_labels_list[i], energy_list[i], traj_weights_list[i])]

                train_data_weights = torch.cat([data_init_list[i][4] for i in range(len(traj_data_list))], dim=0)
                test_data_weights = torch.cat([data_init_list[i][8] for i in range(len(traj_data_list))], dim=0)

            data_shape = data_init_list[0][0]
            train_past_data = torch.cat([data_init_list[i][1] for i in range(len(traj_data_list))], dim=0)
            train_future_data = torch.cat([data_init_list[i][2] for i in range(len(traj_data_list))], dim=0)
            train_data_labels = torch.cat([data_init_list[i][3] for i in range(len(traj_data_list))], dim=0)

            test_past_data = torch.cat([data_init_list[i][5] for i in range(len(traj_data_list))], dim=0)
            test_future_data = torch.cat([data_init_list[i][6] for i in range(len(traj_data_list))], dim=0)
            test_data_labels = torch.cat([data_init_list[i][7] for i in range(len(traj_data_list))], dim=0)
            Ut_train =  torch.cat([data_init_list[i][9] for i in range(len(traj_data_list))], dim=0)
            Ut_test =  torch.cat([data_init_list[i][10] for i in range(len(traj_data_list))], dim=0)

            for RC_dim in RC_dim_list:
                for neuron_num1 in neuron_num1_list:
                    for neuron_num2 in neuron_num2_list:
                        for beta in beta_list:
                            for gamma in gamma_list:
                                for learning_rate in learning_rate_list:

                                    output_path = IB_path + "_d=%d_t=%d_b=%.9f_gamma=%.9f_learn=%f" \
                                        % (RC_dim, dt, beta, gamma, learning_rate)

                                    IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device, \
                                                UpdateLabel, neuron_num1, neuron_num2)
                                    
                                    IB.to(device)
                                    
                                    # use the training set to initialize the pseudo-inputs
                                    IB.init_representative_inputs(train_past_data, train_data_labels)

                                    train_result = False
                                    
                                    optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

                                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

                                    print('Training...')
                                    train_result = SPIB_training.train(IB, beta, gamma, Ut_train, Ut_test, train_past_data, train_future_data, \
                                                                    train_data_labels, train_data_weights, test_past_data, test_future_data, \
                                                                        test_data_labels, test_data_weights, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, \
                                                                            batch_size, threshold, patience, refinements, output_path, \
                                                                                log_interval, device, seed, penalty = penalty, bandwidth = bandwidth)
                                    
                                    if train_result:
                                        print(train_result)
                                        return
                                    
                                    print('Ouputting results...')
                                    SPIB_training.output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                                                    test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, \
                                                                        output_path, final_result_path, dt, beta, gamma, Ut_train, Ut_test, learning_rate, penalty, seed, output_thermo = False, bandwidth = bandwidth)

                                    for i in range(len(traj_data_list)):
                                        IB.save_traj_results(traj_data_list[i], batch_size, output_path, SaveTrajResults, i, seed)
                                    
                                    IB.save_representative_parameters(output_path, seed)


if __name__ == '__main__':
    
    test_model_advanced()
    
