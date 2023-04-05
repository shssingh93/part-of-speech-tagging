class Solver:
    
    def __init__(self):
        # Initializing some class attributes
        self.initial_pos_count = {}
        self.initial_pos_proba = {}
        self.second_level_transition_counts = {}
        self.second_level_transition_proba = {}

    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.simplified_posterior_score
            # return -999
        elif model == "HMM":
            return self.hmm_posterior
        elif model == "Complex":
            # return self.calculate_mcmc_score(sentence, self.tags)
            # return (self.simplified_posterior_score + self.hmm_posterior) / 2
            return -999
        else:
            print("Unknown algo!")

    # compute  Transition Probability
    def tag2_given_tag1(self, train_words_tagged, t2, t1):
        # Checking for all the tags present in train_words_tagged
        tags = [pair[1] for pair in train_words_tagged]
        # Calculating the count for tag1 in train_words_tagged
        count_t1 = len([t for t in tags if t==t1])
        count_t2_t1 = 0
        for index in range(len(tags)-1):
            # count_t2_t1 checks the occurence of tag t2 after tag t1 and calculates
            # the probability of tag t2 given that tag t1 has just occurred
            if tags[index]==t1 and tags[index+1] == t2:
                count_t2_t1 += 1
        return (count_t2_t1, count_t1)
    
    # Compute Transition Probability 
    def create_transition_prob_table(self, tags, train_words_tagged):
        # creating transition probability dictionary of pair of tags
        # Key appears as a tuple of a pair of tags (t1, t2) and value is the
        # probability of occurence of tag t2 after tag t1 has occurred
        self.tags_dict = {}
        for i, t1 in enumerate(list(tags)):
            for j, t2 in enumerate(list(tags)):
                t2_t1 = self.tag2_given_tag1(train_words_tagged, t2, t1)
                num = t2_t1[0]
                den = t2_t1[1]
                self.tags_dict[(t1,t2)] = num/den
        return self.tags_dict

    # Compute Emission Probability 
    def create_emission_prob_table(self, train_words_tagged):
        # For this function, we have already ran the except part
        # of the code and stored the emission_probability_dictionary 
        # as a json file. This is done to save time for execution
        # as there are more than 45000 unique words (numbers and punctuation)
        # and 12 tags, resulting in more than half a million possibilities
        try:
            f = open('emission_dictionary.json')
            temp_emission_proba_table = json.load(f)
            # Keys are loaded as a string of tuples, so a simple implementation of eval
            # function helps us to convert the string back to tuple
            temp_emission_proba_table = {eval(key):item for key,item in temp_emission_proba_table.items()}
            return temp_emission_proba_table
        except:
            # The table key is of form (word, associated_pos_tag)
            # The value is probability that the word's part of speech is the associated tag
            tag_lists = {}
            emission_proba_table = {}
            tags = self.tags
            tag_lens = {tag:0 for tag in self.tags}
            train_vocabulary = list({word for word, tag in self.train_words_tagged})
            
            # Calculating count of each tag across all the words
            for tag in tag_lens.keys():
                tag_list = [x for x in train_words_tagged if x[1]==tag]
                # Appending all the occurences of the tag in a tag_list
                tag_lists[tag] = tag_list
                # Appending the tag's count in a list
                tag_lens[tag] = len(tag_list)
            # Parsing through the words in the train_vocabulary
            for word in train_vocabulary:
                count_word_given_tag = 0
                # For every tag as a key in tag_list
                for tag in tag_lists.keys():
                    # Check if the (word,tag) pair is in the table or not
                    if (word,tag) not in emission_proba_table.keys():
                        # If it is not in table, check for every occurence of the pair
                        word_given_tag_list = [x[0] for x in tag_lists[tag] if x[0]==word]
                        # calculate the total no. of times the word has occurred as the passed pos tag.
                        count_word_given_tag = len(word_given_tag_list)
                        # Calculate the emission probability for the pair and insert in the table
                        emission_proba_table[(word,tag)] = count_word_given_tag/tag_lens[tag]
            # Converting the tuple key as string to saving a temp copy as a json file for future access
            temp_emission_proba_table = {str(key):item for key,item in emission_proba_table.items()}
            with open('emission_dictionary.json', 'w') as f:
                json.dump(temp_emission_proba_table, f)
            # Returning the original table
            return emission_proba_table

    def get_posterior_prob(self, sentence, train_words_tagged):
        state_list = []
        post_score = 0
        tags_list = self.tags
        self.emission_prob_list = []
        self.simplified_posterior_score = 0
        # Parsing the test sentence
        for key, word in enumerate(sentence):
            # Initializing a list to store probabilities for (word,tag) pair
            temp_emission_p_list = [] 
            # For each tag, do:
            for tag in tags_list:
                # Retrieve the emission probability of word given tag from the table                    
                emission_p = self.emission_proba_table[(word,tag)] if (word,tag) in self.emission_proba_table.keys() else 0
                # Append the probability to a temp list
                temp_emission_p_list.append(emission_p)
                # Append the temp list to list of emission probabilities
                self.emission_prob_list.append(temp_emission_p_list)
            # This step directly helps with the simplified algorithm by giving us the 
            # max probability, instead of searching later on.
            # State list checks the tag from a table of predicted tags (based on probabilities)
            # and chooses the max probability and its corresponding tag
            state_list.append(tags_list[temp_emission_p_list.index(max(temp_emission_p_list))])
            # Also calculating the log of posterior score for simple algorithm
            # This will be directly referred by the posterior function as well
            # as the viterbi function for futher calculation 
            self.simplified_posterior_score += math.log(max(temp_emission_p_list)) if max(temp_emission_p_list)!=0 else 0
        return state_list
    
    # Do the training!
    def train(self, data):
        # train_words_tagged will create a list of tuples of form (word, associated_pos_tag)
        # for every word in the train set
        self.train_words_tagged = []
        for sentence, sentence_tags in data:
            for word, word_tag in zip(sentence, sentence_tags):
                self.train_words_tagged.append((word, word_tag))

        # Initializing tags set as a list
        self.tags = ['pron', 'x', 'num', 'adp', 'verb', 'adv', 'det', 'noun', 'prt', 'adj', 'conj', '.']
        
        # Preparing a table of transition probabilities between pos tags
        self.transition_proba_table = self.create_transition_prob_table(self.tags, self.train_words_tagged)
        # Preparing a table of emission probabilities between every word and 12 pos tags
        self.emission_proba_table = self.create_emission_prob_table( self.train_words_tagged)
        # Calculating inital distribution of pos tags based on data in train file
        for tag in self.tags:
            self.create_initial_count_for_pos_tag(tag)
        for tag in self.tags:
            self.get_initial_pos_tag_proba(tag)
    
    def simplified(self, sentence, train_words_tagged):
        # The function calculates posterior probabilities and retuns the 
        # sequence of tags for every word in sentence
        state = self.get_posterior_prob(sentence, train_words_tagged)
        return state

    def hmm_viterbi(self, sentence):
        # Transition and Emission tables are already prepared in the train step 
        # let's call the viterbi algorithm
        self.hmm_posterior, updated_state = self.viterbi(sentence, self.train_words_tagged, self.transition_proba_table)
        return updated_state

    def viterbi(self, sentence, train_words_tagged, transition_proba_table):
        tags_list = self.tags
        post = 0
        updated_state = []
        # Parsing the sentence
        for key, word in enumerate(sentence):
            p = [] 
            # Parsing each tag
            for tag in tags_list:
                # If we are encountering the first word, consider it as a blank space
                # denoted by punctuation, this step takes care of the intitial probability
                # distribution and we don't have to check for all the initial states
                if key == 0:
                    transition_p = transition_proba_table[('.', tag)]
                else:
                    # Else, simply consiter the last tag and calculate the 
                    # transition probability between the current and last tag
                    transition_p = transition_proba_table[(updated_state[-1], tag)] 
                # fetch emission probability and calculate state probabilities
                emission_p = self.emission_proba_table[(word,tag)] if (word,tag) in self.emission_proba_table.keys() else 0
                # State probability as product of emission and transition probabilities
                state_probability = emission_p * transition_p 
                p.append(state_probability)
            # Adding the post probable value to the posterior score
            post += math.log(max(p)) if max(p)!=0 else 0
            # getting state for which probability is maximum
            updated_state.append(tags_list[p.index(max(p))])
        # Returning the posterior score and the tag sequence
        return (post, updated_state)

    # Creating a intitial count table
    # when the part of speech is at the beginning of the sentence
    # For MCMC Gibbs Sampling Complex stage
    def create_initial_count_for_pos_tag(self, tag):
        if tag in self.initial_pos_count:
            self.initial_pos_count[tag] = self.initial_pos_count[tag] + 1
        else:
            self.initial_pos_count[tag] = 1
    
    # Creating a intitial probability distribution table using the counts
    # when the part of speech is at the beginning of the sentence
    # For MCMC Gibbs Sampling Complex stage
    def get_initial_pos_tag_proba(self, tag):
        if tag in self.initial_pos_count:
            return self.initial_pos_count[tag] / sum(self.initial_pos_count.values())
        return 0.00000001

    # Computing the transition counts from Sn -> Sn+1 -> Sn+2
    # For MCMC Gibbs Sampling Complex stage
    def second_level_transition_pos_counts(self, tag1, tag2, tag3):
        if tag1 in self.second_level_transition_counts:
            if tag2 in self.second_level_transition_counts[tag1]:
                if tag3 in self.second_level_transition_counts[tag1][tag2]:
                    self.second_level_transition_counts[tag1][tag2][tag3] = self.second_level_transition_counts[tag1][tag2][tag3] + 1
                else:
                    self.second_level_transition_counts[tag1][tag2][tag3]= 1
            else:
                self.second_level_transition_counts[tag1][tag2] = {tag3 : 1}
        else:
            self.second_level_transition_counts[tag1]= {tag2:{tag3 : 1}}
    
    # Creating second level transition probability table from Sn -> Sn+1 -> Sn+2
    # For MCMC Gibbs Sampling Complex stage
    def fetch_second_level_transition_pos_proba(self, tag1, tag2, tag3):
        if tag1 in self.second_level_transition_proba and tag2 in self.second_level_transition_proba[tag1] and tag3 in  self.second_level_transition_proba[tag1][tag2]:
            return self.second_level_transition_proba[tag1][tag2][tag3]
        if tag1 in self.second_level_transition_counts and tag2 in self.second_level_transition_counts[tag1] and tag3 in self.second_level_transition_counts[tag1][tag2]:
            value = self.second_level_transition_counts[tag1][tag2][tag3]/sum(self.second_level_transition_counts[tag1][tag2].values())
            self.second_level_transition_counts[tag1]  = { tag2 : {tag3 : value}  }
            return value
        return 0.00000001
    
    def complex_mcmc(self, sentence):
        states = []
        tag_counter_list = []
        # We use the simplified algorithm for the initial state
        # Gives a better start to the algorithm
        state = self.simplified(sentence, self.train_words_tagged)
        # Keeping iterations to 50 to reduce execution time
        epochs = 50
        # Keeping the burn in iterations to 10
        # These initial iterations won't be considered when we walk
        # through the MCMC
        burn_in_epochs = 10
        for i in range(epochs):
            # Updating the state
            state = self.prepare_state(sentence, state)
            # If burn in iterations are crossed, update the states
            if i>=burn_in_epochs:
                states.append(state)
        
        # Maintaining a tag counter
        for j in range(len(sentence)):
            tag_counter = {}
            for state in states:
                if state[j] in tag_counter.keys():
                    tag_counter[state[j]] += 1
                else:
                    tag_counter[state[j]] = 1
            tag_counter_list.append(tag_counter)
        
        # Final sequence consists of the max probable tags found throughout the runs for the given sentence
        final_state_tags =  [max(tag_counter_list[i], key=tag_counter_list[i].get) for i in range(len(sentence))]
        return final_state_tags

    def prepare_state(self, sentence, state):
        tags = self.tags
        for i in range(len(sentence)):
            probas = [0] * len(tags)
            log_probas = [0] * len(tags)
            for j in range(len(tags)):
                state[i] = tags[j]
                # Calculating the log of probability 
                # that will be used further and by the posterior function
                log_probas[j] = self.calculate_mcmc_score(state, sentence)
            
            # We reduce the log of probabilities by the minimum value as they tend to 0
            min_proba = min(log_probas)
            for k in range(len(log_probas)):
                log_probas[k] -= min_proba
                # Updating the probability for kth word
                probas[k] = math.pow(10, log_probas[k])
            # Calculating new probabilities that will be used for random selection later
            proba_sum = sum(probas)
            probas = [element / proba_sum for element in probas]
            rand_num = random.random()
            prob = 0
            # For every tag, if the probability sum is greater than the random value
            # Assign the sum to the current tag
            for k in range(len(probas)):
                prob += probas[k]
                if rand_num < prob:
                    state[i] = tags[k]
                    break
        return state
    
    def calculate_mcmc_score(self, state, sentence):
        # Consider the first tag out of the generated sequence of tags
        tag1 = state[0]
        # Calculate the log of initial probability of the tag
        if tag1 in self.initial_pos_count.keys():
            prob_tag1 = math.log(self.initial_pos_count[tag1] / sum(self.initial_pos_count.values()), 10)
        else:
            prob_tag1 = 0.00000001  
        a,b,c=0,0,0
        for i in range(len(state)):
            if i<len(sentence) or i<len(state):
                # Fetch and add the emission probability of the given (word,tag) pair
                if (sentence[i], state[i]) in self.emission_proba_table.keys():
                    b+=math.log(self.emission_proba_table[(sentence[i], state[i])]) if self.emission_proba_table[(sentence[i], state[i])] !=0 else 0
            else:
                b+=0
            if i!=0:
                # If we are parsing after the first index, add the first level transition probability
                # Between the current and previous predicted tag
                if (state[i-1], state[i]) in self.tags_dict:
                    a+=math.log(self.tags_dict[(state[i-1], state[i])]) if self.tags_dict[(state[i-1], state[i])] !=0 else 0
                else:
                    a+=0
            # If we are parsing after the first two tags, add the second level transition probabilities
            # Between the ith, (i-1)th, and (i-2)nd tags
            if i!=0 and i!=1:
                c+=math.log(self.fetch_second_level_transition_pos_proba(state[i-2], state[i-1], state[i]))
        # Return the total sum of the probabilites
        return prob_tag1+a+b+c
    
    # This solve() method is called by label.py, so you should keep the interface the
    # same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    # part of speech per word.
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence, self.train_words_tagged)
            # return [ "noun" ] * len(sentence)
        elif model == "HMM":
            # return [ "noun" ] * len(sentence)
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return [ "noun" ] * len(sentence)
            # return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

