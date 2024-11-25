# mmWave_ML_Interview

## Question (a)
This question makes me think of ``cocktail party problem'', however, the difference is also large. Looking at the signatures, it could be seen clearly that they are all spectrograms that are delivered by computing the STFT of a signal. An intuitive thought is to first extract some kind of prior-knowledge feature related to this specific circumstances and then use it together with K-means to perform clustering. However, the prior-knowledge is unknown, and hence dimension reduction should be performed without prior-knowledge. PCA and 2DPCA seems like an easy and straight forward method, their results are as below:


However, PCA's effectiveness of directly performing it on the values of a spectrogram is doubted. Since I am currently working with VAE, I started looking around with prior work, and indeed found one which they first transformed the spectrograms into figures, and then used VAE on the figures. I did the same, and the results are as below:

In that specific paper, they also claimed that using VAE would yield a much higher accuracy, hence my answer to the first question is 4, which the details of each sample as below.

For the specific implementation and the chain of thought, please refer to github repo:, file.

## Question (b)

## Question (c)

## Question (d)
### (i)

### (ii)

### (iii)