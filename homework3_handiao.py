############################################################
Classification
############################################################
import collections
import  os


############################################################
# Imports
############################################################
# Include your imports here, if any are used.
import email
from email.iterators import body_line_iterator
import math

############################################################
# Section 1: Spam Filter
############################################################

def load_tokens(email_path):
    with open(email_path,encoding="utf8") as infile:
        content = email.message_from_file(infile)
    totallist = []
    for line in email.iterators.body_line_iterator(content):
        word = line.split()
        totallist = totallist + word
    return  totallist


def log_probs(email_paths, smoothing):
    word_counter = collections.Counter()
    for item in email_paths:
        word = load_tokens(item)
        word_counter.update(word)

    prob_dict ={}
    alpha = smoothing
    sun_w = sum(word_counter.values())
    v_plus1 = len(word_counter) + 1
    bottom = sun_w + alpha * (v_plus1)
    for key in word_counter:
        count_w = word_counter[key]
        top = count_w + alpha

        prob_dict[key] =  math.log(top/bottom)
    prob_dict["<UNK>"] = math.log(alpha/ bottom)
    return  prob_dict



class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        path_spam = [os.path.join(spam_dir, i) for i in os.listdir(spam_dir)]
        num_spam_files = len(path_spam)
        path_ham = [os.path.join(ham_dir, i) for i in os.listdir(ham_dir)]
        num_ham_files = len(path_ham)
        total = num_spam_files + num_ham_files


        self.ham_probs = log_probs(path_ham, smoothing)
        self.spam_probs = log_probs(path_spam, smoothing)
        self.log_spam = math.log(num_spam_files / total)
        self.log_ham = math.log(num_ham_files / total)


    def is_spam(self, email_path):
        tokens = load_tokens(email_path)
        email_counter = collections.Counter()
        email_counter.update(tokens)

        spampr = self.log_spam
        hampr = self.log_ham
        for key in email_counter:
            wordcount = email_counter[key]
            if key in self.spam_probs:
                spamwordpr = self.spam_probs[key] * wordcount
            else:
                spamwordpr = self.spam_probs["<UNK>"] * wordcount

            if key in self.ham_probs:
                hamwordpr = self.ham_probs[key] * wordcount
            else:
                hamwordpr = self.ham_probs["<UNK>"] * wordcount
            spampr = spampr + spamwordpr
            hampr =  hampr + hamwordpr
        return spampr > hampr


    def most_indicative_spam(self, n):
        scoredict = {}
        intersection = set(self.ham_probs.keys()) & set(self.spam_probs.keys())
        for item in intersection:
            score = self.spam_probs[item] - self.ham_probs[item]
            scoredict[item] = score

        sorted_words = sorted(scoredict.items(), key=lambda x: x[1], reverse=True)
        result = []
        for item in range(n):
            result.append(sorted_words[item][0])
        return  result


    def most_indicative_ham(self, n):
        scoredict = {}
        intersection = set(self.ham_probs.keys()) & set(self.spam_probs.keys())
        for item in intersection:
            score =  self.ham_probs[item] - self.spam_probs[item]
            scoredict[item] = score

        sorted_words = sorted(scoredict.items(), key=lambda x: x[1], reverse=True)
        result = []
        for item in range(n):
            result.append(sorted_words[item][0])
        return result


