import os
from random import random
import numpy as np


def load_ner_dataset(filename, path="data"):
    """ 
    loads the dataset with named entity recognition labels, 
    seperated by sentences not documents
    """
    file = open(os.path.join(path, filename), "r")
    X = list()
    Y = list()
    x = list()
    y = list()
    for row in file:
        row = row.strip()
        if row and row != "-DOCSTART- -X- -X- O":
            splitted_row = row.split("\t")
            x.append(splitted_row[0])
            y.append(splitted_row[-1])
        elif x:
            X.append(x)
            Y.append(y)
            x = list()
            y = list()
    return X, Y


def load_embeddings(filename, path="embeddings"):
    """ 
    loads glove 50d embeddings and returns a word 
    to index dict, index to word list and the embedding matrix
    """
    file = open(os.path.join(path, filename), "r")
    word_to_index = {"<PAD>": 0, "<OOV>": 1}
    index_to_word = ["<PAD>", "<OOV>"]
    embedding_matrix = [
        [0 for _ in range(50)],
        [random() * 2 - 1 for _ in range(50)]
    ]
    for i, row in enumerate(file, 2):
        row = row.strip().split()
        word = row[0]
        embedding = [float(emb) for emb in row[1:]]
        word_to_index[word] = i
        index_to_word.append(word)
        embedding_matrix.append(embedding)
    return word_to_index, index_to_word, embedding_matrix


def label_to_index_generator(label_list):
    """ returns a lookup for labels """
    label_vocab = set(
        [label for sub_label_list in label_list for label in sub_label_list])
    label_to_index = dict()
    index_to_label = list()
    for i, label in enumerate(label_vocab):
        label_to_index[label] = i
        index_to_label.append(label)
    return label_to_index, index_to_label, len(index_to_label)


def tokenize(X, word_to_index, lower=True, OOV_token=1):
    """ tokenizes the input X by the lookup word_to_index """
    tokenized_X = list()
    for sentence in X:
        tokenized_sentence = list()
        for word in sentence:
            if lower:
                word = word.lower()
            tokenized_sentence.append(word_to_index.get(word, OOV_token))
        tokenized_X.append(tokenized_sentence)
    return tokenized_X


def create_data_buckets(X_test_tokenized, Y_test_tokenized, index_to_label, label_to_index):
    """ 
    returns multiple buckets of data
    3 buckets for the length of the sentence
    2 buckets for sentences with and without OOV token
    3 buckets for entity density in a sentence
    3 buckets for token-label consistency
    2 buckets for max entity length
    """
    buckets = dict()
    X = np.array(X_test_tokenized, dtype=object)
    Y = np.array(Y_test_tokenized, dtype=object)
    
    # sentence_length
    sentence_length = np.array([len(s) for s in Y])
    index_len_under_10 = sentence_length < 10
    index_len_between_10_40 = (sentence_length >= 10) & (sentence_length < 40)
    index_len_ge_40 = sentence_length >= 40
    X_len_under_10 = X[index_len_under_10]
    Y_len_under_10 = Y[index_len_under_10]
    X_len_between_10_40 = X[index_len_between_10_40]
    Y_len_between_10_40 = Y[index_len_between_10_40]
    X_index_len_ge_40 = X[index_len_ge_40]
    Y_index_len_ge_40 = Y[index_len_ge_40]
    buckets["Sentence Length"] = [(X_len_under_10, Y_len_under_10, f"length < 10 (n={len(X_len_under_10)})"), 
                                  (X_len_between_10_40, Y_len_between_10_40, f"10 <= length < 40 (n={len(X_len_between_10_40)})"), 
                                  (X_index_len_ge_40, Y_index_len_ge_40, f"length > 40 (n={len(X_index_len_ge_40)})")]
    
    # max entity length
    I_entity_tokens = [i for i, token in enumerate(index_to_label) if token[0] == "I"]
    max_entitiy_length = [0 for _ in X]
    for i, x_tokens in enumerate(Y):
        longest = 1
        curr_len = 1
        for token in x_tokens:
            if token not in I_entity_tokens:
                if curr_len > longest:
                    longest = curr_len
                curr_len = 1
            else:
                curr_len += 1
        if curr_len > longest:
            longest = curr_len
        max_entitiy_length[i] = longest  
    max_entitiy_length = np.array(max_entitiy_length)
    
    index_max_entitiy_length_1 = max_entitiy_length == 1
    index_max_entitiy_length_2 = (max_entitiy_length > 1) & (max_entitiy_length < 4)
    index_max_entitiy_length_3 = (max_entitiy_length > 3) & (max_entitiy_length < 5)
    index_max_entitiy_length_4 = max_entitiy_length > 4
    X_max_entitiy_length_1 = X[index_max_entitiy_length_1]
    Y_max_entitiy_length_1 = Y[index_max_entitiy_length_1]
    X_max_entitiy_length_2 = X[index_max_entitiy_length_2]
    Y_max_entitiy_length_2 = Y[index_max_entitiy_length_2]
    X_max_entitiy_length_3 = X[index_max_entitiy_length_3]
    Y_max_entitiy_length_3 = Y[index_max_entitiy_length_3]
    X_max_entitiy_length_4 = X[index_max_entitiy_length_4]
    Y_max_entitiy_length_4 = Y[index_max_entitiy_length_4]
    buckets["max Entitiy Length"] = [(X_max_entitiy_length_1, Y_max_entitiy_length_1, f"max entity length = 1 (n={len(X_max_entitiy_length_1)})"), 
                                     (X_max_entitiy_length_2, Y_max_entitiy_length_2, f"1 < max entity length < 4 (n={len(X_max_entitiy_length_2)})"), 
                                     (X_max_entitiy_length_3, Y_max_entitiy_length_3, f"3 < max entity length < 5 (n={len(X_max_entitiy_length_3)})"), 
                                     (X_max_entitiy_length_4, Y_max_entitiy_length_4, f"max entity length > 4 (n={len(X_max_entitiy_length_4)})")]
    
    # OOV count
    OOV_token = 1
    OOV_token_amount = np.array([0 for _ in X])
    for i, x in enumerate(X):
        for token in x:
            if token == OOV_token:
                OOV_token_amount[i] += 1
    index_without_OOV = OOV_token_amount == 0
    index_with_OOV = OOV_token_amount > 0         
    X_without_OOV = X[index_without_OOV]
    Y_without_OOV = Y[index_without_OOV]
    X_with_OOV = X[index_with_OOV]
    Y_with_OOV = Y[index_with_OOV]
    buckets["presence of OOV"] = [(X_without_OOV, Y_without_OOV, f"without OOV (n={len(X_without_OOV)})"), 
                                  (X_with_OOV, Y_with_OOV, f"with OOV (n={len(X_with_OOV)})")]
       
    # entity density
    O_token = label_to_index["O"]
    entity_density = [0 for _ in Y]
    for i, y in enumerate(Y):
        for token in y:
            if token != O_token:
                entity_density[i] += 1
        entity_density[i] /= len(y)
    entity_density = np.array(entity_density)
    index_density_1 = entity_density < 0.1
    index_density_2 = (entity_density >= 0.1) & (entity_density <= 0.2)
    index_density_3 = (entity_density >= 0.2) & (entity_density < 0.5)
    index_density_4 = entity_density >= 0.5
    X_density_1 = X[index_density_1]
    Y_density_1 = Y[index_density_1]
    X_density_2 = X[index_density_2]
    Y_density_2 = Y[index_density_2]
    X_density_3 = X[index_density_3]
    Y_density_3 = Y[index_density_3]
    X_density_4 = X[index_density_4]
    Y_density_4 = Y[index_density_4]
    buckets["Entity Density"] = [(X_density_1, Y_density_1, f"entity density < 0.1 (n={len(X_density_1)})"), 
                                   (X_density_2, Y_density_2, f"0.1 <= entity density < 0.2 (n={len(X_density_2)})"), 
                                   (X_density_3, Y_density_3, f"0.2 <= entity density < 0.5 (n={len(X_density_3)})"), 
                                   (X_density_4, Y_density_4, f"entity density > 0.5 (n={len(X_density_4)})")]
    
    # average token label inconsistency
    token_label_consitency_dict = dict()
    for x_tokens, y_tokens in zip(X, Y):
        for x, y in zip(x_tokens, y_tokens):
            if x in token_label_consitency_dict:
                if y not in token_label_consitency_dict[x]:
                    token_label_consitency_dict[x].append(y)
            else:
                token_label_consitency_dict[x] = [y]
    for key, value in token_label_consitency_dict.items():
        token_label_consitency_dict[key] = len(token_label_consitency_dict[key]) 
    avg_token_label_consitency = [0 for _ in X]
    for i, x_tokens in enumerate(X):
        for x in x_tokens:
            avg_token_label_consitency[i] += token_label_consitency_dict[x]
        avg_token_label_consitency[i] /= len(x_tokens)
    avg_token_label_consitency = np.array(avg_token_label_consitency)
    index_consitency_1 = avg_token_label_consitency == 1
    index_consitency_2 = (avg_token_label_consitency > 1) & (avg_token_label_consitency <= 2)
    index_consitency_3 = (avg_token_label_consitency > 2) & (avg_token_label_consitency <= 3)
    index_consitency_4 = (avg_token_label_consitency > 3) & (avg_token_label_consitency <= 4)
    index_consitency_5 = avg_token_label_consitency > 5
    X_consitency_1 = X[index_consitency_1]
    Y_consitency_1 = Y[index_consitency_1]
    X_consitency_2 = X[index_consitency_2]
    Y_consitency_2 = Y[index_consitency_2]
    X_consitency_3 = X[index_consitency_3]
    Y_consitency_3 = Y[index_consitency_3]
    X_consitency_4 = X[index_consitency_4]
    Y_consitency_4 = Y[index_consitency_4]
    X_consitency_5 = X[index_consitency_5]
    Y_consitency_5 = Y[index_consitency_5]
    buckets["avg Token Label Inconsistency"] = [(X_consitency_1, Y_consitency_1, f"avg inconsistency = 1 (n={len(X_consitency_1)})"), 
                                              (X_consitency_2, Y_consitency_2, f"1 < avg inconsistency <= 2 (n={len(X_consitency_2)})"), 
                                              (X_consitency_3, Y_consitency_3, f"2 < avg inconsistency <= 3 (n={len(X_consitency_3)})"), 
                                              (X_consitency_4, Y_consitency_4, f"3 < avg inconsistency <= 4 (n={len(X_consitency_4)})"), 
                                              (X_consitency_5, Y_consitency_5, f"avg inconsistency > 4 (n={len(X_consitency_5)})")]    

    return buckets


def evaluate_on_buckets(buckets, model):
    for bucket_name, bucket in buckets.items():
        subbucket = list()
        for x, y, subbucket_name in bucket:
            subbucket.append([model.evaluate(x, y)[-1], subbucket_name])
        buckets[bucket_name] = subbucket
    return buckets