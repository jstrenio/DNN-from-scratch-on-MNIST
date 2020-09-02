# John Strenio
# PSU ID: 970018744
# CS_545
# Professor: Paul Doliotis
# Assignment_2

# import libraries
import numpy as np
import pandas as pd 
import seaborn as sn
import matplotlib.pyplot as plt
import time

# CONSTANTS
DIGITS = 10
MAX_EPOCHS = 50
NUM_EXP = 3
FULL_DATA_LEN = 60000

# Experiment Choice 1, 2, 3 else all
choice = input("select experiment (1-3) or 0 for all: ")
choices = ['1', '2', '3']
if choice not in choices:
    choice = '0'

# start timer
program_start_time = time.time()

def main(DIGITS, MAX_EPOCHS, NUM_EXP, FULL_DATA_LEN, exp):
    # vars (some pre-declared for references in different experiments)
    train = "mnist_train.csv"
    valid = "mnist_validation.csv"
    expected_vals_train = images = TRAIN_DATA_LEN = PIXEL_LEN = None
    expected_vals_valid = v_images = VALID_DATA_LEN = None
    weights_in = weights_out = old_delta_in = old_delta_out = None
    data = None

    # learning rates + set learning rate
    learning_rates = [0.1]
    lr = learning_rates[0]

    # setup experiment specific variables
    hidden_sizes = [20, 50, 100]
    training_data_sizes = [1.0, 0.5, 0.25]
    momentum_rates = [0.25, 0.5, 0.95]
        
    # accuracy scores
    train_score = np.zeros(shape=(NUM_EXP, MAX_EPOCHS))
    valid_score = np.zeros(shape=(NUM_EXP, MAX_EPOCHS))

    # confusion matrices
    cfn_mx = [np.zeros(shape=(DIGITS, DIGITS)) for i in range(NUM_EXP)]

    # functions
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def scale_data_size(data, h, DATA_LEN, FULL_DATA_LEN):
        # get new data length
        new_data_len = int(FULL_DATA_LEN * training_data_sizes[h])

        # sort the data by the digit it is
        data = data.sort_values(0)

        # mark the first occurences of each digit
        l = []
        for i in range(DIGITS):
            l.append(data[0].searchsorted(i))
        l.append(FULL_DATA_LEN)

        # split the digits data into seperate lists
        l_mod = [0] + l + [max(l)+1]
        data_list_split = [data.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]

        # trim the arrays to the specified length
        new_data_list = []
        for i in data_list_split:
            new_data_list.append(i.drop(i.index[int(new_data_len/DIGITS):], axis=0))

        # put the new shortened data set back together
        new_data = pd.concat(new_data_list)
        new_data.drop(new_data.tail(1).index, inplace=True)

        return new_data, new_data_len

    def import_data(csv_file, h, exp, is_training_data, data, DATA_LEN, expected_vals, images, PIXEL_LEN, FULL_DATA_LEN):
        
        # read csv into pandas
        if h == 0 or exp == '2':
            data = pd.read_csv(csv_file, header=None)

            # save sizes for later use
            DATA_LEN = data[0].count() # 60,000 images
        
        # save the original size of the data length for rescaling use
        if h == 0:
            FULL_DATA_LEN = DATA_LEN
            
        # evenly scale data size if running experiment 2 and not the first round
        if exp == '2' and h != 0 and is_training_data:
            data, DATA_LEN = scale_data_size(data, h, DATA_LEN, FULL_DATA_LEN)

        # randomly shuffle data if its any of the exp 2 cycles or if its just the first cycle
        if exp == '2' or h == 0:
            data = data.sample(frac=1).reset_index(drop=True)
            if is_training_data:
                print("training ", end='')
            else:
                print("validation ", end='')
            print("data length: " + str(DATA_LEN))
            
            # save training data labels
            expected_vals = data[0]
            images = data.drop(columns=0) # images -= expected values

            # scale the pixel values
            images = images.apply(lambda x: x/255)

            # add bias's to every image
            PIXEL_LEN = len(images.columns) + 1 # 784 pixels + 1 bias (added below)
            images.insert(0, "bias", np.ones(DATA_LEN, dtype=int))
            images = images.to_numpy()

        return (expected_vals, images, DATA_LEN, PIXEL_LEN, data)

    def train_weights(weights_in, weights_out, expected_vals_train, images, lr, old_delta_in, old_delta_out):
        # select example
        pixels = images[i]
        pixels_trans = pixels.reshape(-1, 1)
        pixels = pixels[np.newaxis,...] # add's 1 dimension for pixels shape * originally (785,) -> (785,1) *
        
        # setup t values for comparisons
        expected = expected_vals_train[i]
        t = np.full((DIGITS,1), 0.1)
        t[expected] = 0.9

        # ----- forward propagate activations ------------------
        # generate hidden activations 
        hidden_layer = sigmoid(np.dot(weights_in, pixels_trans))

        # attach bias onto newly generated hidden layer
        hidden_layer_wbias = np.vstack(([1], hidden_layer))

        # use those to generate output activations
        y = sigmoid(np.dot(weights_out, hidden_layer_wbias))

        # ----- back propagate errors & update weights *stochastic gradient descent = updating / example* ------
        # calculate output error = (output_vector)(1-output_vector)(t_vector - output_vector)
        output_error_vector = np.multiply.reduce((y, np.subtract(1, y), np.subtract(t, y)))

        #transpose this for delta_W_out dot product
        output_error_vector_trans = output_error_vector.reshape(1, -1)

        # calculate hidden error  hidden * (1 - hidden) * (weights_out dot output_error)
        #     (20x1)           <-               (20x1)  *   (20x1)
        hidden_error_vector = np.multiply(hidden_layer, np.subtract(1, hidden_layer))
        
        #    (21x10)     <-     (10x21)
        weights_out_trans = weights_out.T
        
        # must remove bias weight for hidden_error_vector cross product
        #    (20x10)         <-                 (21x10)
        weights_out_no_bias = np.delete(weights_out_trans, (0), axis=0)

        #    (20x1)         <-                    (20x1) *  (20x1)       <-      (20x10) DOT (10x1)
        hidden_error_vector = np.multiply(hidden_error_vector, np.dot(weights_out_no_bias, output_error_vector))

        # weights_out delta  
        #         (21x10)        <-                         (21x1) DOT (1x10)
        delta_weights_out_matrix = lr * (np.dot(hidden_layer_wbias, output_error_vector_trans))

        # weights_in delta
        #      (20x785)         <-                        (20x1) DOT (1x785)
        delta_weights_in_matrix = lr * (np.dot(hidden_error_vector, pixels))
        
        # calculate and add momentum term for exp 3
        if exp == '3':
            # no momentum term for first loop
            if j != 0:
                delta_weights_in_matrix = np.add(delta_weights_in_matrix, np.multiply(momentum_rates[h], old_delta_in))
                delta_weights_out_matrix = np.add(delta_weights_out_matrix, np.multiply(momentum_rates[h], old_delta_out))
            
            # save delta_weights for next loop
            old_delta_in = delta_weights_in_matrix
            old_delta_out = delta_weights_out_matrix

        # return the updated weights and the delta's for the momentum term
        #               (20x785) + (20x785)                           (10x21) + (10x21)
        return np.add(weights_in, delta_weights_in_matrix), np.add(weights_out, delta_weights_out_matrix.T), old_delta_in, old_delta_out

    def test_weights(weights_in, weights_out, DATA_LEN, expected_vals, images, cfn_mx, fill_matrix):
        correct = 0

        # for every image
        for i in range(DATA_LEN):
            expected = expected_vals[i]

            # forward propagate throug NN (see training_weights fn for detailed breakdown)
            pixels = images[i].reshape(-1, 1)
            hidden_layer = sigmoid(np.dot(weights_in, pixels))
            hidden_layer = np.vstack(([1], hidden_layer))
            y = sigmoid(np.dot(weights_out, hidden_layer))
                
            # select max value from outputs as answer
            prediction = np.argmax(y)

            # if its correct + 1 correct update confusin matrix
            if prediction == expected:
                correct += 1
            if fill_matrix:
                cfn_mx[h][expected][prediction] += 1

        return correct, cfn_mx

    # import data before loop when data size is static
    if exp != '2':
        print("reading in data...")
        expected_vals_train, images, TRAIN_DATA_LEN, PIXEL_LEN, data = import_data(train, 0, exp, True, data, TRAIN_DATA_LEN, expected_vals_train, images, PIXEL_LEN, FULL_DATA_LEN)
        expected_vals_valid, v_images, VALID_DATA_LEN, PIXEL_LEN, data = import_data(valid, 0, exp, False, data, VALID_DATA_LEN, expected_vals_valid, v_images, PIXEL_LEN, FULL_DATA_LEN)
        print("done")

    # ------ main experiment loop ---------------------
    for h in range(NUM_EXP):
        #loop start time
        loop_start_time = time.time()

        # process data and scale if necessary for exp 2
        if exp == '2':
            print("preping data...")
            expected_vals_train, images, TRAIN_DATA_LEN, PIXEL_LEN, data = import_data(train, h, exp, True, data, TRAIN_DATA_LEN, expected_vals_train, images, PIXEL_LEN, FULL_DATA_LEN)
            if h == 0:
                expected_vals_valid, v_images, VALID_DATA_LEN, PIXEL_LEN, data = import_data(valid, h, exp, False, data, VALID_DATA_LEN, expected_vals_valid, v_images, PIXEL_LEN, FULL_DATA_LEN)
            print("done")

        # set hidden layer size
        if exp == '1':
            N = hidden_sizes[h]
        # lock hidden layer size to 100 for exp 2 and 3
        else:
            N = hidden_sizes[2]

        # initialize weights_in -> n hidden units x 784+1 pixel weights
        weights_in = pd.DataFrame(np.random.uniform(-0.05, 0.05, size=(N, PIXEL_LEN)))
        weights_in = weights_in.to_numpy()

        # initialize weights_out -> 10 digits x n+1 hidden units
        weights_out = pd.DataFrame(np.random.uniform(-0.05, 0.05, size=(DIGITS, N+1)))
        weights_out = weights_out.to_numpy()

        # print experiment pertinent data
        if exp == '1':
            print("Beginning training and learning of hidden layer size: " + str(N) + "...")
        elif exp == '2':
            print("Beginning training and learning of data size: " + str(TRAIN_DATA_LEN) + "...")
        else:
            print("Beginning training and learning of momentum rate: " + str(momentum_rates[h]) + "...")

        # ------ main training and testing loop -----------------------------
        for j in range(MAX_EPOCHS):
            # test weights: compute accuracy on training and validation set & save for plot
            print("testing weights on training and validation data...")
            correct_train = test_weights(weights_in, weights_out, TRAIN_DATA_LEN, expected_vals_train, images, cfn_mx, False)[0] # <- don't need confusion matrix until after training
            train_score[h][j] = correct_train / TRAIN_DATA_LEN * 100
            correct_valid = test_weights(weights_in, weights_out, VALID_DATA_LEN, expected_vals_valid, v_images, cfn_mx, False)[0]
            valid_score[h][j] = correct_valid / VALID_DATA_LEN * 100
            print("done")

            # train the weights for number of training data (1 epoch)
            print("training epoch " + str(j) + "...") 
            for i in range(TRAIN_DATA_LEN):
                weights_in, weights_out, old_delta_in, old_delta_out = train_weights(weights_in, weights_out, expected_vals_train, images, lr, old_delta_in, old_delta_out)   
            print("finished epoch " + str(j) + " accuracy(%): training_data: " + str(round(train_score[h][j], 2)) + " validation data: " + str(round(valid_score[h][j], 2)))

        # training done fill confusion matrix
        cfn_mx = test_weights(weights_in, weights_out, VALID_DATA_LEN, expected_vals_valid, v_images, cfn_mx, True)[1] # <- now only storing the confusion matrix output
        if exp == '1':
            print("Training and Testing of hidden layer size: " + str(N) + " complete.")
        elif exp == '2':
            print("Training and Testing of data size: " + str(TRAIN_DATA_LEN) + " complete.")
        else:
            print("Training and Testing of momentum rate: " + str(momentum_rates[h]) + " complete.")
        print("50 epoch time (mins): " + str(round((time.time() - loop_start_time) / 60, 2)))
        print("-------------------------------------------------------------------------")

    # graphing
    for i in range(NUM_EXP):
        # plot
        plt.figure()
        if exp == '1':
            plt.title("Hidden Layer Size " + str(hidden_sizes[i]))
        elif exp == '2':
            plt.title("Data Size " + str(training_data_sizes[i] * 60000))
        if exp == '3':
            plt.title("Momentum Rate " + str(momentum_rates[i]))
        plt.xlabel("Epoch")
        plt.ylabel("% Accuracy")
        plt.plot(train_score[i], label='training data accuracy')
        plt.plot(valid_score[i], label='validation data accuracy')
        plt.legend()
        plt.grid()

        # confusion matrix
        plt.figure()
        ax = plt.subplot()
        df_cfn_mx = pd.DataFrame(cfn_mx[i], range(DIGITS), range(DIGITS))
        sn.set(font_scale=1.2)
        sn.heatmap(df_cfn_mx, annot=True, annot_kws={"size": 8}, ax = ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        if exp == '1':
            ax.set_title("Hidden Layer Size: " + str(hidden_sizes[i]) + ' Confusion Matrix')
        if exp == '2':
            ax.set_title("Data Size: " + str(training_data_sizes[i]) + ' Confusion Matrix')
        if exp == '3':
            ax.set_title("Momentum Rate: " + str(momentum_rates[i]) + ' Confusion Matrix')
    
    # wait to show plots incase of running multiple experiments consecutively 
    plt.draw()
    print("Total Runtime: " + str(round((time.time() - program_start_time) / 60, 2)))
    print("Experiment " + exp + " complete.")

# run selected expirement(s)
if choice == '0':
    for m in choices:
        exp = m
        main(DIGITS, MAX_EPOCHS, NUM_EXP, FULL_DATA_LEN, exp)
else:
    exp = choice
    main(DIGITS, MAX_EPOCHS, NUM_EXP, FULL_DATA_LEN, exp)

# print plots
plt.show()
