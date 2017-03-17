"""
Author: Olivia Winn
Project: NSF-NLPVision
Date: 01/21/17

Given an existing set of word clusters and word synonyms, provides negative
examples for given adjectives.

Lemmatizer: handles 'ish' as a comparative modifier, recognizes 'grey' and
'gray' as the same word. Has it's own equality function that can handle
different understandings of compound words (either by comparing as an entire
word or comparing the parts of the word). Can lemmatize by only returning the
first word in a compound word, or lemmatizing both components and returning them

Clusters: Class to hold the given clusters and synsets, and contains
functions to return negative examples as well as synonyms for words. Uses the
lemmatizer to lemmatize and compare words. Given different parameters,
handles compound words differently - can get a negative example for just the
first word, or a new compound word that is made of negative exampels of both
words.
"""

from nltk.stem.wordnet import WordNetLemmatizer
import random
from numpy import setdiff1d
import pickle


class OurLemmatizer():
    """
    Uses the WordNet lemmatizer, and handles 'ish' as if it is a comparative
    modifier. Depending on the input, returns lemmatized version of just the
    first word, just the last word, or both words.
    *** Manually handles understanding 'gray' and 'grey' as the same word
    """

    def __init__(self):
        self.orig_lem = WordNetLemmatizer()

    def lemmatize(self, word, first_word=True, both=False):
        '''
        Lemmatizes the given word
        :param word: word to lemmatize
        :param first_word: boolean of whether to return the first or the last
        word of a compound word
        :param both: return the first and last word of a compound word
        :return: lemmatized word
        '''
        if '-' in word:
            if first_word:
                word = word.split('-')[0]
            else:
                if both:
                    word1 = word.split('-')[0]
                    word2 = word.split('-')[-1]
                    lem1 = self.orig_lem.lemmatize(word1, 'a')
                    lem2 = self.orig_lem.lemmatize(word2, 'a')
                    return self.handle_ish(lem1) + '-' + self.handle_ish(lem2)
                else:
                    word = word.split('-')[-1]
        lem_word = self.orig_lem.lemmatize(word, 'a')
        # Represent 'grey' and 'gray' as the same word
        if 'grey' in lem_word:
            lem_word.replace('grey', 'gray')
        return self.handle_ish(lem_word)

    def handle_ish(self, word):
        '''
        Not perfect handling of 'ish', but it is at least consistent
        :param word: adjective to lemmatize
        :return: word w/o 'ish' if it was present
        '''
        if word.endswith('ish') and len(word) > 4:
            new_word = word.replace('ish', '')
            # If the word ends in a vowel that isn't e, assume e was removed
            # when 'ish' was added to the end
            if new_word[-1] in 'aiou':
                return new_word + 'e'
            elif new_word[-1] not in 'aeiou' and new_word[-2] == new_word[-1]:
                return new_word[:-1]
            return new_word
        return word

    def equals(self, word1, word2, part=True):
        '''
        Compares whether two words are equal w.r.t. the lemmatization of this class
        Ex: word1='blueish-green', word2='red-blue'
        If part=True, returns True as 'blueish' will match 'blue'. If
        part=False, will return False.
        :param word1: first word being compared
        :param word2: second word being compared
        :param part: if either word is compound, check whether any part of
        one word matches another, rather than the entire word
        :return: equality of words
        '''
        if part:
            if '-' in word1:
                w11 = word1.split('-')[0]
                w12 = word1.split('-')[-1]
                if '-' in word2:
                    w21 = word2.split('-')[0]
                    w22 = word2.split('-')[-1]
                    return (self.lemmatize(w11) == self.lemmatize(w21) or
                            self.lemmatize(w11) == self.lemmatize(w22) or
                            self.lemmatize(w12) == self.lemmatize(w21) or
                            self.lemmatize(w12) == self.lemmatize(w22))
                else:
                    return (self.lemmatize(w11) == self.lemmatize(word2) or
                            self.lemmatize(w12) == self.lemmatize(word2))
            elif '-' in word2:
                w21 = word2.split('-')[0]
                w22 = word2.split('-')[-1]
                return (self.lemmatize(word1) == self.lemmatize(w21) or
                        self.lemmatize(word1) == self.lemmatize(w22))
        return self.lemmatize(word1) == self.lemmatize(word2)


class Clusters:
    """
    Loads the clusters of words from the given file, and performs operations
    on them
    """
    def __init__(self, info_file):
        """
        Loads the file and stores it
        :param info_file: File containing cluster and synset info
        """
        self.lem = OurLemmatizer()
        [self.clusters, self.synsets] = pickle.load(open(info_file, "r"))
        self.c_dict = self.make_dict(self.clusters)
        self.s_dict = self.make_dict(self.synsets)
        return

    def make_dict(self, emb_list):
        """
        Given a list of word sets, creates a dictionary whose values are the
        index of the set the word (key) belongs in
        :param emb_list: list of word clusters
        :return: dictionary with words as keys and cluster indices as values
        """
        new_dict = {}
        for li in emb_list:
            items = {word: emb_list.index(li) for word in li}
            new_dict.update(items)
        return new_dict

    def find_cluster(self, adj):
        """
        Returns the index of the cluster the given word is in
        :param adj: word to search for
        :return: index of word's cluster; None if not found
        """
        return self.get_group(adj, is_cluster=True)

    def get_synset(self, adj):
        """
        Returns all words in the given adj's synset
        :param adj: adjective to search for
        :return: list of adj's synonyms (just [adj] if none found)
        """
        synset = self.get_group(adj, is_cluster=False)
        return self.synsets[synset] if synset else [adj]

    def get_group(self, adj, is_cluster, both=False):
        """
        Given the list of word groups, returns index of given word's location
        :param adj: word to search for
        :param is_cluster: whether searching through word clusters or synsets
        :param both: if word is compound, search for indices for both parts of
        the word. Otherwise only search for the index of the first part
        :return: Index of word cluster (or [ind, ind] for compound words and
        both=True), None if not found (or None in [ind, ind] for
        corresponding word not found)
        """
        if is_cluster:
            dic = self.c_dict
        else:
            dic = self.s_dict
        if not both:
            if adj in dic:
                return dic[adj]
            elif self.lem.lemmatize(adj) in dic:
                return dic[self.lem.lemmatize(adj)]
            return None
        adj1 = adj.split('-')[0]
        adj2 = adj.split('-')[-1]
        indices = [None, None]
        if self.lem.lemmatize(adj1) in dic:
            indices[0] = dic[self.lem.lemmatize(adj1)]
        if self.lem.lemmatize(adj2) in dic:
            indices[1] = dic[self.lem.lemmatize(adj2)]
        return indices

    def get_example(self, cluster, adj=None):
        '''
        Picks an example word from the given cluster
        :param cluster: Index of the cluster to generate an example from
        :param adj: Optional word to avoid returning a version of (given
        'grey', will not return 'greyish' or 'grey-blue')
        :return: new word from cluster
        '''
        if cluster is not None:
            synset = self.get_synset(adj)
            exclusive = setdiff1d(self.clusters[cluster], synset).tolist()
            if exclusive:
                new_word = random.choice(exclusive)
                if adj:
                    while self.lem.equals(adj, new_word):
                        new_word = random.choice(exclusive)
                return new_word
        return None

    def get_negative(self, adj, both=False):
        '''
        Given an adjective, return a negative example from its cluster
        :param adj: Word to generate a negative example for
        :param both: If word is compound, whether to get a negative example
        for just the first word or to generate negative examples for both
        :return: New example (None if word not found)
        '''
        cluster = self.find_cluster(adj)
        new_word = self.get_example(cluster, adj)
        if not new_word:
            return None
        # Handle compound words
        if '-' in adj or both:
            clusters = self.get_group(adj, True, True)
            # If the words are part of different clusters, handle separately
            if clusters[0] is not clusters[1]:
                # If the first word didn't have a negative, just return a
                # negative example for the second word
                if clusters[0] is None:
                    lem_word = self.lem.lemmatize(adj, False)
                    return self.get_example(clusters[1], lem_word)
                else:
                    lem_word = self.lem.lemmatize(adj, False)
                    if both and clusters[1] is not None:
                        new_word2 = self.get_example(clusters[1], lem_word)
                        while self.lem.equals(lem_word, new_word2):
                            new_word2 = self.get_example(clusters[1], lem_word)
                        return new_word + '-' + new_word2
                    # If both words should be replaced but the second word
                    # didn't have negative examples, just return a negative
                    # example of the first word
                    elif both:
                        return new_word
                    else:
                        return new_word + '-' + lem_word

            else:  # Both words in same cluster
                lem2 = self.lem.lemmatize(adj, False)
                while self.lem.equals(new_word, lem2):
                    new_word = self.get_example(cluster, adj)
                # Ensure the negative examples do not match either original word
                if both:
                    if '-' in new_word:
                        return new_word
                    new_word2 = self.get_example(cluster, adj)
                    while self.lem.equals(new_word2, lem2) or \
                            self.lem.equals(new_word2, new_word):
                        new_word2 = self.get_example(cluster, adj)
                    if '-' in new_word2:
                        return new_word2
                    return new_word + '-' + new_word2
        return new_word

    def get_synonym(self, adj):
        '''
        Returns a word from the synset of the given adjective
        :param adj: Word to get synonym for
        :return: A synonym of the given adjective (None if not found)
        '''
        synset = self.get_synset(adj)
        exclusive = setdiff1d(synset, unicode(adj)).tolist()
        if len(exclusive) > 1:
            new_choice = random.choice(exclusive)
            return new_choice
        return None


## Test cases to examine output
#clusters = Clusters('cluster_info.pk')
#print clusters.get_negative('simple')
# print clusters.get_negative('pearlescent')
# print clusters.get_negative('blue-grey', True)
# print clusters.get_negative('purple-red', False)
# print clusters.get_negative('blue-striped', True)
# print clusters.get_negative('blue-striped', False)
# print clusters.get_synonym('bright')
# print clusters.get_synonym('curved')