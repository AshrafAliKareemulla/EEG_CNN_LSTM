import torch.nn as nn 
import torch.optim
import argparse
from Train import train_model
from Architecture import CNN, LSTM, Hybrid_CNN_LSTM
from sklearn.model_selection import KFold
import h5py





# Argument Parser
parser = argparse.ArgumentParser(description = "CNN/LST for EEG")


# Device Settings
parser.add_argument('--device', default = 'cpu', type = str, help = 'CPU/GPU for training')
parser.add_argument('--architecture', default = 'cnn', type = str, help = 'CNN/LSTM/Hybrid')


# Signal Preprocessing
parser.add_argument('--freq_domain_sampling_rate', default = 3, type = int, help = 'Freq Domain Sampling Rate')
parser.add_argument('--start_freq', default = 3, type = int, help = 'Start Freqency')
parser.add_argument('--end_freq', default = 3, type = int, help = 'End Frequency')
parser.add_argument('--window_length', default = 3, type = int, help = 'Window length of each sample point')
parser.add_argument('--original_freq', default = 3, type = int, help = 'Original Frequency')


# Utils
parser.add_argument('--num_classes', default = 3, type = int, help = 'Number of classes')
parser.add_argument('--batch_size', default = 64, type = int, help = 'Training Batch Size')
parser.add_argument('--num_workers', default = 4, type = int, help = 'Number of data loading workers')


# Load, Save, Resume checkpoints
parser.add_argument('--start_epoch', default = 0, type = int, help = 'Starting epoch for training')
parser.add_argument('--end_epoch', default = 10, type = int, help = 'Ending epoch for training')








delta_params = {'stftn' : 128, 'fStart' : 1, 'fEnd' : 3, 'fs' : 512, 'window' : 1}
theta_params = {'stftn' : 128, 'fStart' : 4, 'fEnd' : 7, 'fs' : 512, 'window' : 1}
alpha_params = {'stftn' : 128, 'fStart' : 8, 'fEnd' : 13, 'fs' : 512, 'window' : 1}
beta_params = {'stftn' : 128, 'fStart' : 14, 'fEnd' : 30, 'fs' : 512, 'window' : 1}
gamma_params = {'stftn' : 128, 'fStart' : 31, 'fEnd' : 50, 'fs' : 512, 'window' : 1}














if __name__ == '__main__':

    args = parser.parse_args()
    
    if ((args.device == 'gpu') and (torch.cuda.is_available())):
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    print(f"---> Using {args.device} for training the model")


    if (args.architecture == 'cnn'):
        model = CNN().to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 9e-5)

    elif (args.architecture == 'lstm'):
        model = LSTM().to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)

    elif (args.architecture == 'hybrid'):
        model = Hybrid_CNN_LSTM().to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    lossfun = nn.CrossEntropyLoss()
    
    print(f"Trainable parameters for {args.architecture} : { sum(n.numel() for n in model.parameters() if n.requires_grad) }")


    # Load the data and labels

    with h5py.File('SEED_DE.h5', 'r') as f:
        data = f['key1'][:]
        labels = f['key2'][:]

    # print(data.shape)
    # print(len(labels))
    

    # Train the model

    train_acc, train_loss, test_acc = train_model(args, model, lossfun, optimizer, data, labels)