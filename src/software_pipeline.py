from __future__ import annotations
from BrisbaneVPRDataset import BrisbaneVPRDataset
from VPRNetwork import VPRNetwork
from plotting import plot_confusion_matrix

import os, sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import signal

import numpy as np

import lava.lib.dl.slayer as slayer
from lava.lib.dl.slayer.classifier import Rate

# ------------------- Settings -----------------------#

# General settings
epochs = 50

# Network settings
input_size = 34
threshold = 1.00

# Data settings
num_places = 30
start_time = 100
place_gap = 2 #164/num_places # The streams run for approximately 164 seconds 
samples_per_sec = 1000
place_duration = 2
max_spikes = None

# Sequencer settings
sequence_length = 3

# Plot settings
redo_plot = False
vmin = 0
vmax = 0.05


#----------------- Helper Functions -----------------# 

def transpose( matrix):
    if len(matrix) == 0:
        return []
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]



# Make a folder for the trained network
trained_folder = 'Trained'
os.makedirs(trained_folder, exist_ok=True)

# Use GPU
print(torch.cuda.is_available())
device = torch.device('cuda')

#---------------- Create the Network -----------------#

# Create the network
net = VPRNetwork(input_size, num_places, threshold=threshold).to(device)

#---------------- Load the Dataset -------------------#

# Load the data
training_set = BrisbaneVPRDataset(train=True, place_duration = place_duration, place_gap=place_gap, num_places=num_places, start_time=start_time,samples_per_sec=samples_per_sec, max_spikes=max_spikes, subselect_num=input_size)
testing_set  = BrisbaneVPRDataset(train=False, place_duration = place_duration, training_locations=training_set.training_locations, place_gap=place_gap, num_places=num_places, start_time=start_time, samples_per_sec=samples_per_sec, max_spikes=max_spikes, subselect_num=input_size)
            
train_loader = DataLoader(dataset=training_set, batch_size=5, shuffle=True)
test_loader  = DataLoader(dataset=testing_set , batch_size=5, shuffle=True)

print(training_set[0])


#------------- Training the Network -----------------#

# Define an optimiser 
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training the network
error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)

# Create a training assistant object
stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)


if redo_plot:
    for i, (input, label) in enumerate(train_loader): # training loop
        output = assistant.train(input, label)

else: 
    for epoch in range(epochs):

        for i, (input, label) in enumerate(train_loader): # training loop
            output = assistant.train(input, label)
        print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')

        for i, (input, label) in enumerate(test_loader): # training loop
            output = assistant.test(input, label)
        print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')

        if epoch%20 == 19: # cleanup display
            print('\r', ' '*len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
            stats_str = str(stats).replace("| ", "\n")
            print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')

        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), trained_folder + '/network.pt')
        stats.update()
        stats.save(trained_folder + '/')
        net.grad_flow(trained_folder + '/')


# ----------------- Load best network -------------------#

# import the best network during training 
net.load_state_dict(torch.load(trained_folder + '/network.pt'))
net.export_hdf5(trained_folder + '/network.net')

# Get the output for the input to each place
test_loader2  = DataLoader(dataset=testing_set , batch_size=5, shuffle=False)
rate = []
labels = []
for i, (input, label) in enumerate(test_loader2):
    output = net(input.to(device)) # Get network output
    #guesses = assistant.classifier(output).cpu().data.numpy() # get the predictions 
    rate.extend(Rate.rate(output).cpu().data.numpy()) # Get the firing rates for each place 
    labels.extend(label.cpu().data.numpy()) # Get place labels


# Get rates in percentages of total
for i in range(num_places):
    sum = np.sum(rate[i])
    if sum != 0:
        rate[i] = np.divide(rate[i],sum)

# Make confusion matrix annotations
accuracy = 0
annotations = [['' for i in range(num_places)] for j in range(num_places)]
for qryIndex in range(num_places):
    max_idx = np.argmax(rate[qryIndex])
    annotations[max_idx][qryIndex] = 'x'
    if max_idx ==qryIndex:
        accuracy += 1


#--------------- Apply a sequencer ------------------#
I = np.identity(sequence_length)
conv = signal.convolve2d(rate, I, mode='same')
print(np.shape(conv))

# Make confusion matrix annotations
accuracy_s = 0
annotations_s = [['' for i in range(num_places)] for j in range(num_places)]
for qryIndex in range(num_places):
    max_idx = np.argmax(conv[qryIndex])
    annotations_s[max_idx][qryIndex] = 'x'
    if max_idx == qryIndex:
        accuracy_s += 1

#--------------- Save Results ------------------#

# Transpose the matricies 
rate = transpose(rate)
conv = transpose(conv)

output_path = "./results/confusion_matrix_" 
output_path_s = "./results/confusion_matrix_seq_" 
plot_confusion_matrix(rate, labels, annotations, output_path, vmin, vmax)
plot_confusion_matrix(conv, labels, annotations_s, output_path_s, vmin, vmax)

# Print accuracy!!!
accuracy = accuracy/num_places
accuracy_s = accuracy_s/num_places
print("The accuracy of the network is: " + str(accuracy))
print("The accuracy with a sequencer is: " + str(accuracy_s))

# Log the results
# line = str(threshold) + " - " + str(accuracy) + " - " + str(accuracy_s) + "\n"
# f.write(line)