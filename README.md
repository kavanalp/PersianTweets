# PersianTweets

1 Introduction
In the NLP world of predicting the next word in a sentence, it is necessary to build a stream data model that can add knowledge to the sequence model by learning from the current data. This approach leads to generating a block called Recurrent Neural Network(RNN). These amazing blocks come in two different types, LSTM and GRU.
2. Dataset
Persian tweets contain 249981 records. Each type consists of between 1 and 280 characters.
Unfortunately, no more information is added to the dataset
3. Purpose
Training a Convolutional Neural Network pipeline using Keras package with different hidden layers using Rnn blocks(GRU, LSTM ) with a different number of units, different learning rate, and ...
4. Preprocessing
The dataset has one column, which is a string, so we will drop all nan values. We Also tokenize the data by removing worthless characters extra, spaces enters, and, ....
We sort values based on their length, and we use the first 1000 values to build a strong model that can be reused for the rest of our dataset and to accelerate our rough training.
Because we don't want to work with strings, we assign every word to its own index so we can use vectors of numbers instead of each record.
With n_gram architecture, we make an n sample for each vector, and it divides each sentence into a sequence of different numbers as inputs, so all words in a sentence can be predicted no matter where they occur. The last step is to make all inputs the same size. The length of every vector is the length of the longest tweet, and the rest spots are zeroes
5. Model
I built 2 models that both are in RNN set First is LSTM(Long short term memory) And GRU(gated recurrent unit). These models are mostly used to predict sequential data.
5.1 LSTM
During back propagation, recurrent neural networks suffer from the vanishing gradient problem. Gradients are used to update the weights of a neural network. The vanishing gradient problem is when the gradient shrinks as it back propagates through time. If a gradient value becomes extremely small, it doesn’t contribute too much learning.
Our purpose is to tune the hyper parameter of our LSTM model which are the number of units, learning rate, optimizer, batch size, and, number of epochs. The embedding matrix(which reduces the input matrix and we are able to overcome memory-intensive problems) is constant.
As you can see, different hyper parameters resulted in different accuracy and loss.
             This heatmap shows a correlation between different hyper-parameters and accuracy, as you can see, there is a 78 percent correlation between batch_size and accuracy, which shows that in one hidden LSTM, larger batch size is more effective. The optimizer does not significantly change the result; ADAM like always wins the competition against RMS. Different units behave differently during different training sessions.
5.2 GRU
The GRU is the newer generation of Recurrent Neural networks and is pretty similar to an LSTM. GRUs got rid of the cell state and used the hidden state to transfer information. It also only has two gates, a reset gate, and an update gate. It has an update gate that acts exactly like forget gate in LSTM and a Reset gate which let how much information pass to the update gate.
Our purpose is to tune the hyper parameter of our GRU model which are the number of units, learning rate, optimizer, batch size, and, number of epochs.
 The embedding matrix(which reduces the input matrix and we are able to overcome memory-intensive problems) is constant.
        The number of units decreased through training and as you can see there is a minus correlation between accuracy and unit number, the higher the number, the lower the accuracy.
6 Evaluation
6.1 Performance
In our problem, we can see that the LSTM has performed more accurately. The best accuracy was 95 percent for the LSTM with 150 hidden units in one layer LSTM 128 batch_size, which means that for every iteration, 128 data points were trained using the ADAM optimizer.
GRU's best performance was 71 percent accuracy containing 50 interior units, 128 batch sizes, and ADAM as well.
6.2 Timing
LSTM has more gates than GRU, The Timing difference is theoretically obvious. Unfortunately, we did not go through the details of the training time for each model separately. Because we only stopped on 20 - 30 epochs. This can be
overcome by having a large number of epochs and checking for the same accuracy over a large number of epochs.
But overall views in my code showed LSTM worked. I count this as some miss-configurations
