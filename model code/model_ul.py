# Created for NEUR490 Project
# Author: Katja Brand
# Description: A simple model of memory consolidation during sleep using a Hopfield network,
#              based off a previous model by Walker and Russo (2004). This version also contains
#              an implementation of unlearning.
# How to run: This program takes four arguments in the following order:
#             - Number of sleep cycles (integer
#             - Number of unlearning cycles (integer)
#             - Frequency of noise bursts (integer)
#             - Amplitude of noise bursts (float)

import numpy as np
import matplotlib.pyplot as plt
import sys
import random

# Function to plot the images after the testing phase
def plot_images(images, title, no_i_x, no_i_y=3):
    fig = plt.figure(figsize=(10, 15))
    fig.canvas.set_window_title(title)
    images = np.array(images).reshape(-1, 10, 15)
    images = np.pad(images, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=-1)
    for i in range(no_i_x):
        for j in range(no_i_y):
            ax = fig.add_subplot(no_i_x, no_i_y, no_i_x * j + (i + 1))
            ax.matshow(images[no_i_x * j + i], cmap="gray")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
                            
            if j == 0 and i == 0:
                ax.set_title("Real")
            elif j == 0 and i == 1:
                ax.set_title("Distorted")
            elif j == 0 and i == 2:
                ax.set_title("Reconstructed")

# Function for training the network using WR learning rule
def train(neu, training_patterns, w, cycles=600):
    prev_o = np.zeros(len(training_patterns[0])) #array to hold previous output of network
    for pat in training_patterns:
        output = np.array(pat)
        for _ in range(cycles):
            w = update_weights(w, prev_o, output)
            prev_o = np.array(output)
            output = update_output(prev_o, w)#update output
            if np.array_equal(prev_o, output):
                break #move onto next pattern when network reaches stable state
    for diag in range(neu):
        w[0][diag][diag] = 0 # self-connection weight set to zero
        w[1][diag][diag] = 0
    return w, prev_o

# Function for updating weights of the network using learning rule from Walker and Russo (2004)
# To modify STM or LTM learning coefficients, change a_s or a_l respectively
def update_weights(w, prev_o, o, a_s=0.01, a_l=0.001):
    for i in range(n_neurons):
        for j in range(n_neurons):
            w[0][i][j] = (1.0 - a_s)*(w[0][i][j]) + a_s*(2*o[i]-1)*(2*o[j]-1) #STM weight update
            w[1][i][j] = (1.0 - a_l)*(w[1][i][j]) + a_l*(2*o[i]-1)*(2*o[j]-1) #LTM weight update
    return w

# Function for updating network output
def update_output(o, w):
    indices = list(range(n_neurons))
    indices = random.sample(indices, len(indices)) #puts indices in random order
    for n in range(n_neurons):
        i = indices[n]
        x = 0.0
        for j in range(n_neurons):
            x += (w[0][i][j]*o[j] + w[1][i][j]*o[j])/2
        if x > 0:
            o[i] = 1
        else:
            o[i] = 0
    return o

# Function to simulate sleep
def sleep(f, a, w, o, cycles=4800):
    #set initial state of network
    for c in range(0,cycles):
        prev_o = o #save previous network state
        if c % f == 0: #check for noise burst
            n_invert = int(a*len(o)) #number of neurons to invert
            for n in range(n_invert):
                r = np.random.randint(0, len(o))
                if o[r] == 1:
                    o[r] = 0
                else:
                    o[r] = 1
        w = update_weights(w, prev_o, o, a_s=0.0)#update LTM weights
        o = update_output(o,w) #calculate new output
    return w

# Function to simulate unlearning during REM sleep
# To change STM learning coefficient during unlearning, change a_s passed to update_weights().
def rem(f, a, w, o, cycles=1200):
    #set initial state of network
    for c in range(0,cycles):
        prev_o = o #save previous network state
        if c % f == 0: #check for noise burst
            n_invert = int(a*len(o)) #number of neurons to invert
            for n in range(n_invert):
                r = np.random.randint(0, len(o))
                if o[r] == 1:
                    o[r] = 0
                else:
                    o[r] = 1
        w = update_weights(w, prev_o, o, a_s=-0.0001, a_l=0.0) #unlearn STM weights
        o = update_output(o,w) #calculate new output
    return w

# Function for testing the network
def test(weights, testing_data):
    success = 0.0
    h_dist = np.zeros(len(testing_data))
    count = 0
    
    output_data = []
    
    for data in testing_data:
        true_data = data[0]
        noisy_data = data[1]
        predicted_data = retrieve_pattern(weights, noisy_data)
        if np.array_equal(true_data, predicted_data):
            success += 1.0
            h_dist[count] = 0
        else:
            for i in range(len(true_data)):
                if true_data[i] != predicted_data[i]:
                    h_dist[count] += 1
        output_data.append([true_data, noisy_data, predicted_data])
        count += 1
    
    return (success / len(testing_data)), output_data, h_dist

# Function to retrieve a learnt pattern
def retrieve_pattern(weights, data, steps=100):
    res = np.array(data) # the distorted pattern
    
    for _ in range(steps):
        change = False
        for i in range(len(res)):
            raw_v = (np.dot(weights[0][i], res) + np.dot(weights[1][i], res))/2
            #update network values with weighted sum of inputs
            prev = res[i] #store previous value of neuron
            if raw_v > 0: #set values to 0/1 according to activation rule
                res[i] = 1
            else:
                res[i] = 0
            if prev != res[i]:
                change = True
        if change == False:
            break  #keep going until stable state is reached (no value was changed)
        
    return res

"""
    Determine network parameters.
"""

n_train = 10 # number of training images
n_neurons = 150 # number of units in the network

"""
    Generate training, interference, and testing patterns for the network.
"""
def generate_patterns(n_train, n_neurons):
    # Generate training patterns
    
    train_data = []
    
    for i in range(n_train):
        train_data.append(np.random.randint(0, 2, n_neurons))
    
    # Generate interference patterns
    n_intf = 10
    intf_data = []
    
    for i in range(n_intf):
        intf_data.append(np.random.randint(0, 2, n_neurons))
    
    # Generate testing data by adding noise to training data
    test_data = []

    distort = 0.1 # amount of distortion
    n_invert = int(distort*n_neurons) # number of neurons that are inverted in each pattern

    for i in range(n_train):
        noisy_data = np.array(train_data[i])
        for n in range(n_invert):
            r = np.random.randint(0, n_neurons) #generate random number between 0 and n_neurons
            if noisy_data[r] == 1:
                noisy_data[r] = 0
            else:
                noisy_data[r] = 1
        test_data.append((train_data[i], noisy_data)) #add noised pattern to test_data

    return train_data, intf_data, test_data


"""
   Training and testing of the network.
"""
accuracy = 0.0

while accuracy != 1.0:
    # Generate patterns
    train_data, intf_data, test_data = generate_patterns(n_train, n_neurons)
    
    # Train the network
    weights = np.zeros([2, n_neurons, n_neurons]) # weights array holding short- and long-term weights
    weights, output = train(n_neurons, train_data, weights)
    
    # Test the network
    accuracy, op_imgs, h_dist1 = test(weights, test_data)

# Store initial accuracy and hamming distance
acc1 = accuracy * 100
hd1 = np.sum(h_dist1)/len(h_dist1)

# Get parameters from shell script input
sleep_cycles = int(sys.argv[1])
rem_cycles = int(sys.argv[2])
frequency = int(sys.argv[3])
amplitude = float(sys.argv[4])

# Simulate sleep
# If running unlearning phase between multiple cycles of sleep, uncomment last line.
if sleep_cycles != 0:
    weights = sleep(frequency, amplitude, weights, output, cycles=sleep_cycles)
    if rem_cycles != 0:
        weights = rem(frequency, amplitude, weights, output, cycles=rem_cycles)
    #weights = sleep(frequency, amplitude, weights, output, cycles=sleep_cycles)

# Train on interference patterns
weights, output = train(n_neurons, intf_data, weights, cycles=150)

# Test the network again
accuracy, op_imgs, h_dist2 = test(weights, test_data)

# Store accuracy and hamming distance post-interference
acc2 = accuracy * 100
hd2 = np.sum(h_dist2)/len(h_dist2)

# Output results
print("Cycles: %d    Frequency: %d    Amplitude: %f" % (sleep_cycles, frequency, amplitude))
print("Initial accuracy: %f     Final accuracy: %f" % (acc1, acc2))
print("Initial HD: %f           Final HD: %f" % (hd1, hd2))

"""
    Plot initial, distorted, and reconstructed patterns for each pattern in training set.
    Uncomment to see plots. Not recommended when running program multiple times from shell script.
"""
#plot_images(op_imgs, "Reconstructed Data", n_train)
#plt.show()
