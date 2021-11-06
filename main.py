
from utils import load_ner_dataset, load_embeddings, label_to_index_generator, tokenize, create_data_buckets, evaluate_on_buckets
from plotting import plot_confusion, plot_samples_per_class, plot_f1_scores, plot_evaluated_buckets
from BiLSTM import BiLSTM

from torch.nn import CrossEntropyLoss
from torch.optim import Adam


EPOCHS = 20
HIDDEN_SIZE = 100

# loading data, embeddings and lookups
print("Loading data, embeddings and lookups")
X_train, Y_train = load_ner_dataset("train.conll")
X_dev, Y_dev = load_ner_dataset("dev.conll")
X_test, Y_test = load_ner_dataset("test.conll")
word_to_index, index_to_word, embedding_matrix = load_embeddings(
    "glove.6B.50d.txt")
label_to_index, index_to_label, n_labels = label_to_index_generator(Y_train)

# tokenizing
print("Tokenizing")
X_train_tokenized = tokenize(X_train, word_to_index)
X_dev_tokenized = tokenize(X_dev, word_to_index)
X_test_tokenized = tokenize(X_test, word_to_index)
Y_dev_tokenized = tokenize(Y_dev, label_to_index, lower=False)
Y_train_tokenized = tokenize(Y_train, label_to_index, lower=False)
Y_test_tokenized = tokenize(Y_test, label_to_index, lower=False)

print("Starting training\n")
model = BiLSTM(embedding_matrix, HIDDEN_SIZE, n_labels)
model.cuda()
optimizer = Adam(model.parameters())
loss = CrossEntropyLoss().cuda()
train_history = model.train(EPOCHS,
                            X_train_tokenized,
                            Y_train_tokenized,
                            X_dev_tokenized,
                            Y_dev_tokenized,
                            X_test_tokenized,
                            Y_test_tokenized,
                            optimizer,
                            loss)

# create data buckets for evaluation
print("Starting evaluation")
buckets = create_data_buckets(
    X_test_tokenized, Y_test_tokenized, index_to_label, label_to_index)
evaluated_buckets = evaluate_on_buckets(buckets, model)

# plotting evaluation
plot_evaluated_buckets(evaluated_buckets)
plot_f1_scores(train_history)
plot_samples_per_class(Y_test_tokenized, index_to_label)
plot_confusion(train_history[-1][1], index_to_label)

print("All done.")
