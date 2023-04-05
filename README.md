# Part of Speech Tagging
Our goal here is to implement Part of Speech Tagging using various implementations of Bayesian Networks.
Here, we have tried 3 different techniques to associate a Part of Speech (POS) tag to a given word:
1) **Simple model - Using Naive Bayes Algorithm**
2) **Hidden Markov Model - Using Viterbi Algorithm**
3) **Gibb's Sampling on a Monte Carlo Markov Chain**
 
## 1. Simple Algorithm:
 - We created a dictionary of posterior probabilities of every word and its associated tag by taking into consideration the count of a word and the POS tag it bears throughout the training set.
 - We then divide this count by the total count of the occurrence of the tag throughout the training set.
 - For all the words in the test sequences, we predict the tag by taking into consideration the maximum probability of the word with all the 12 possible tags it can be associated with. The above can be realized with the given formula:

*tags.argmax(Posterior\_Probability[(word, associated\_pos\_tag)] -> P(tag\_given\_word))*
 
 - At the same time, we return the log of that probability of that word to the total sum of the logarithm of the probability of that sentence.

## 2. Viterbi Algorithm:
We have used the previously calculated posterior probabilities as an emission table of the form:

*Emission\_Probability[(word, associated\_pos\_tag)] -> P(tag\_given\_word)*
 
- This is coupled with a transition probability table of the form:

*Transition\_Probability[(current\_tag, previous\_tag)] -> P(current\_tag given previous\_tag)*

- For every (word, tag) pair, we have multiplied the emission probability of that pair with the transition probability of the said tag and its previous tag.
- We finally select the tag with the maximum product of probabilities.
- So the overall formula becomes:

*tags.argmax(Emission\_Probability[(word, associated\_pos\_tag)] * Transition\_Probability[(current\_tag, previous\_tag)])*

- We also add this max probability product to the posterior sum of the Viterbi algorithm and return it to the posterior function.

## 3. Complex Algorithm:
- We first calculate initial and second-level transition probabilities before preparing the Markov chain for the sampling algorithm
- We then use the simple algorithm's prediction as the initial state distribution and for the next 50 epochs (except for the 10 burn-in epochs), we keep on updating the states.
-  This procedure repeats until the state of the tags doesn't change (indicating we have reached the intended distribution) or we have reached the end of the epochs.
- For every word, we now consider the transition probabilities of its tag and the previous two tags that have been predicted as well its emission probability.
- We also keep track of the logarithm of the probability of the tags, later passed on to the posterior function, and keep updating the probability values as they reach zero and finally consider the tags with max probability.

- Following results are obtained by running the code on bc.test file
 
- So far scored 2000 sentences with 29442 words.
 
|                   | Words correct | Sentences correct |
|:-----------------:|:-------------:|:-----------------:|
| 0. Ground truth:  |    100.00%    |      100.00%      |
| 1. Simple         |     90.72%    |       34.95%      |
| 2. HMM            |     92.19%    |       40.55%      |
| 3. Complex        |     18.60%    |       0.00%       |
