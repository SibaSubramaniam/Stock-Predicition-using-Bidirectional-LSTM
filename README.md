# Stock-Predicition-using-Bidirectional-LSTM
The goal of this model is to apply Bidirectional LSTM for stock market prediction. 

Input to the system is the prices of a particular stock for last 'n' days and output are 3 labels: (-1, 0, 1) where -1 denotes a drop in stock price, 0 denotes no change, and 1 denotes an increase in stock price for the next day.

# Bidrectional LSTM
Bidirectional LSTMs train two instead of one LSTMs on the input sequence. The first on
the input sequence as-is and the second on a reversed copy of the input sequence. This can provide
additional context to the network and result in faster and even fuller learning on the problem. So it
was intuitive to use it. This is then followed by a Dense layer with softmax activation.
