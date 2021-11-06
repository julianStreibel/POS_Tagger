from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.functional import softmax
from torch import tensor, logical_not


class BiLSTM(Module):
    def __init__(self,
                 embedding_matrix,
                 hidden_size,
                 output_size):
        super(BiLSTM, self).__init__()
        self.output_size = output_size
        embedding_tensor = tensor(embedding_matrix)
        self.embedding_layer = Embedding.from_pretrained(
            embedding_tensor, freeze=True)
        self.bilstm_layer = LSTM(
            embedding_tensor.shape[1], hidden_size, bidirectional=True,)
        self.linear = Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        x, _ = self.bilstm_layer(x.view(len(x), 1, -1))
        x = self.linear(x.view(len(x), -1))
        return x

    def train(self, epochs, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, opt, loss):
        len_X_train = len(X_train)
        train_history = []
        for e_i in range(epochs):
            epoch_loss = 0
            for x, y in zip(X_train, Y_train):
                self.zero_grad()
                logits = self(tensor(x).cuda())
                batch_loss = loss(logits, tensor(y).cuda())
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss
            acc, confusion_matrix, tp, tn, fp, fn, f1, micro_f1, macro_f1 = self.evaluate(
                X_dev, Y_dev)
            print(f"Epoch: {e_i+1:02}")
            print(f"Train Loss: {epoch_loss / len_X_train:.5f}")
            print(
                f"Dev acc: {acc:f}, dev micro F1: {micro_f1:f}, dev macro F1: {macro_f1:f}\n")
            train_history.append(
                [acc, confusion_matrix, tp, tn, fp, fn, f1, micro_f1, macro_f1])

        acc, confusion_matrix, tp, tn, fp, fn, f1, micro_f1, macro_f1 = self.evaluate(
            X_test, Y_test)
        train_history.append(
            [acc, confusion_matrix, tp, tn, fp, fn, f1, micro_f1, macro_f1])
        print(
            f"Test acc: {acc:f}, test micro F1: {micro_f1:f}, test macro F1: {macro_f1:f}\n")
        return train_history

    def evaluate(self, X, Y):
        Y = tensor([item for sublist in Y for item in sublist]).cuda()
        list_of_logits = [self(tensor(x).cuda()) for x in X]
        list_of_labels = [softmax(logits, dim=1).argmax(dim=1)
                          for logits in list_of_logits]
        Y_hat = tensor(
            [item for sublist in list_of_labels for item in sublist]).cuda()
        confusion_matrix = [
            [0 for _ in range(self.output_size)] for _ in range(self.output_size)]
        for y, y_hat in zip(Y, Y_hat):
            confusion_matrix[y][y_hat] += 1
        acc = sum(Y_hat.eq(Y)) / Y.shape[0]
        tp = tensor([sum((Y == i) & (Y_hat == i))
                    for i in range(self.output_size)])
        tn = tensor([sum(logical_not(Y == i) & logical_not(Y_hat == i))
                    for i in range(self.output_size)])
        fp = tensor([sum(logical_not(Y == i) & (Y_hat == i))
                    for i in range(self.output_size)])
        fn = tensor([sum((Y == i) & logical_not(Y_hat.view(-1) == i))
                    for i in range(self.output_size)])
        pr = tp.div(fp + tp + 1e-7)
        re = tp.div(fn + tp + 1e-7)
        f1 = (pr * re).div(pr + re + 1e-7) * 2
        global_tp = tp.sum()
        global_fp = fp.sum()
        global_fn = fn.sum()
        global_pr = global_tp.div(global_fp + global_tp + 1e-7)
        global_re = global_tp.div(global_fn + global_tp + 1e-7)
        micro_f1 = (global_pr * global_re).div(global_pr +
                                               global_re + 1e-7) * 2
        pr_mean = pr.mean()
        re_mean = re.mean()
        macro_f1 = (pr_mean * re_mean).div(pr_mean + re_mean + 1e-7) * 2
        return acc, confusion_matrix, tp, tn, fp, fn, f1, micro_f1, macro_f1
