# -*- coding: utf-8 -*-
"""10_CNN_Master_Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SMRhwB_Xaf9EC31gZh9HpVpE2iW5LfYJ
"""
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch

torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0))

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# Setting this to True uses a CNN archetecture with 2 fully connected layers.
# If False there are 300 features (len(filter_sizes) * n_filters)
DUAL_FC_LAYERS = True
# The features are extracted from the output of the first layer (150 features per model)
NUM_FEATURES = 150

import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, 
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # self.fc = Identity()
        # self.dropout = Identity()

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

import torch.nn as nn
import torch.nn.functional as F
class CNN_2FC(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, 
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes
                                    ])
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, NUM_FEATURES)
        self.fc = nn.Linear(NUM_FEATURES, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # self.fc = Identity()
        # self.dropout = Identity()

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(self.fc1(cat))

class SUPER_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, dropout, pad_idx, use_base=False, num_features=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, 
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes
                                    ])

        # self.sentiment_model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes,
        #          output_dim, dropout, pad_idx)
        # self.sentiment_model.load_state_dict(torch.load('CNN-sentiment.pt'))
        self.use_base = use_base
        
        if self.use_base:
            self.sarcasm_base_model = torch.load('CNN-sarcasm-base.pt')
            for param in self.sarcasm_base_model.parameters():
                param.requires_grad = False
            self.sarcasm_base_model.embedding = Identity()
            self.sarcasm_base_model.fc = Identity()
            self.sarcasm_base_model.dropout = Identity()
        

        self.sentiment_model = torch.load('CNN-sentiment.pt', map_location=torch.device('cpu'))
        for param in self.sentiment_model.parameters():
          param.requires_grad = False
        self.sentiment_model.embedding = Identity()
        self.sentiment_model.fc = Identity()
        self.sentiment_model.dropout = Identity()

        # self.emotion_model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes,
        #          5, dropout, pad_idx)
        # self.emotion_model.load_state_dict(torch.load('CNN-emotion.pt'))

        self.emotion_model = torch.load('CNN-emotion.pt', map_location=torch.device('cpu'))
        for param in self.emotion_model.parameters():
          param.requires_grad = False
        self.emotion_model.embedding = Identity()
        self.emotion_model.fc = Identity()
        self.emotion_model.dropout = Identity()

        # self.formality_model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes,
        #          output_dim, dropout, pad_idx)
        # self.formality_model.load_state_dict(torch.load('CNN-formality-binary.pt'))

        self.formality_model = torch.load('CNN-formality-binary.pt', map_location=torch.device('cpu'))
        for param in self.formality_model.parameters():
          param.requires_grad = False
        self.formality_model.embedding = Identity()
        self.formality_model.fc = Identity()
        self.formality_model.dropout = Identity()

        # self.informativeness_model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes,
        #          output_dim, dropout, pad_idx)
        # self.informativeness_model.load_state_dict(torch.load('CNN-informativeness-binary.pt'))

        self.informativeness_model = torch.load('CNN-informativeness-binary.pt', map_location=torch.device('cpu'))
        for param in self.informativeness_model.parameters():
          param.requires_grad = False
        self.informativeness_model.embedding = Identity()
        self.informativeness_model.fc = Identity()
        self.informativeness_model.dropout = Identity()

        if DUAL_FC_LAYERS:
            self.fc1 = nn.Linear(len(filter_sizes)*n_filters, NUM_FEATURES)
            #self.fc = nn.Linear(5 * NUM_FEATURES, output_dim)
            self.fc = nn.Linear(5 * NUM_FEATURES, output_dim)
        else:
            #self.fc = nn.Linear(5 * len(filter_sizes)*n_filters, output_dim)
            self.fc = nn.Linear(3 * len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]
        # embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]
        
        # print("TESTING")
        o1 = self.sentiment_model(embedded)
        o2 = self.emotion_model(embedded)
        o3 = self.formality_model(embedded)
        o4 = self.informativeness_model(embedded)
        if self.use_base:
            o5 = self.sarcasm_base_model(embedded)
            cat = self.dropout(torch.cat((o1, o2, o3, o4, o5), dim=1))
            #cat = self.dropout(torch.cat((o2, o5), dim=1))
            #cat = self.dropout(o5)
        else:
            conved = [F.relu(conv(embedded.unsqueeze(1))).squeeze(3) for conv in self.convs]
            # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            pooled = torch.cat(pooled,dim=1)
            if DUAL_FC_LAYERS:
                pooled = self.fc1(pooled)
            cat = self.dropout(torch.cat((pooled, o1, o2, o3, o4), dim=1))
            #cat = self.dropout(pooled)
        # print("TESTING 2")
        # print(pooled.type)
        # print(o1.type)
        # print(o2.type)
        
        # print(cat.shape)

        return self.fc(cat)

"""## Sarcasm Model Definition"""

from torchtext import data
from torchtext import datasets
import torch.optim as optim
import random
import numpy as np
import time
from sklearn.metrics import f1_score
SEED = 1234

LR = 0.001
class ModelTrainer():

    def __init__(self, model_name, model_type=None, text_field_name=None,
                 label_field_name=None, IMDB=False, train_file_name=None,
                 test_file_name=None, embedding_dim=100, criterion_str="CE",
                 max_vocab_size=25000, num_filters=100, filter_sizes=[3, 4, 5],
                 dropout=0.5, output_dim=1, batch_size=64, dtype=torch.long, use_base=False, num_features=300):
        self.model_name = model_name
        self.model_type = model_type
        self.text_field_name = text_field_name
        self.label_field_name = label_field_name
        self.IMDB = IMDB
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.criterion_str = criterion_str
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.batch_size = batch_size
        self.dtype = dtype
        self.use_base = use_base

    def load_dataset(self):
        """ Loads the dataset using torchtext
        """
        r_seed = random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        self.TEXT = data.Field(
            tokenize='spacy',
            tokenizer_language='en_core_web_sm',
            batch_first=True)
        self.LABEL = data.LabelField(dtype=self.dtype) # dtype=torch.float)
        # LOAD DATASET
        if self.IMDB:
            self.dataset, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)
            train_data, valid_data = self.dataset.split(random_state=r_seed)
        else:
            if self.text_field_name == "Sentence":
                # Formality and Informativeness don't work with dict fields
                # tuples representative of tabular format
                fields = [
                          (self.label_field_name, self.LABEL),
                          (self.text_field_name, self.TEXT)
                ]
                skip_header = True
            else:
                fields = {
                    self.label_field_name: (self.label_field_name, self.LABEL),
                    self.text_field_name: (self.text_field_name, self.TEXT)
                }
                skip_header = False
            format = self.train_file_name.split('.')[-1]
            if self.test_file_name:
                
                self.dataset, test_data = data.TabularDataset.splits(
                    #path='/content',
                    path='./content',
                    train=self.train_file_name,
                    test=self.test_file_name,
                    format=format,
                    fields=fields,
                    skip_header=skip_header
                )
                train_data, valid_data = self.dataset.split(random_state=r_seed)
            else:
                self.dataset = data.TabularDataset.splits(
                    #path='/content',
                    path='./content',
                    train=self.train_file_name,
                    format=format,
                    fields=fields,
                    skip_header=skip_header
                )[0]
                train_data, valid_data, test_data = self.dataset.split(
                    split_ratio=[0.8, 0.1, 0.1], random_state=r_seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
          print(torch.cuda.get_device_name(0))

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.batch_size,
            device=self.device,
            sort=False
            )

    def build_vocab(self):
        """ Builds the vocabulary of the dataset using torchtext
        """
        self.TEXT.build_vocab(
            self.dataset,
            max_size=self.max_vocab_size,
            vectors="glove.6B.100d",
            unk_init=torch.Tensor.normal_
        )
        self.LABEL.build_vocab(self.dataset)

    def init_model(self):
        """ Create CNN model with the supplied/derived parameters.
            Loads "glove.6B.100d" weights into the model's embedding layer
        """
        # Get the size of the vocabulary
        vocab_size = len(self.TEXT.vocab)
        if self.criterion_str == "CE":
          output_dim = len(self.LABEL.vocab)
        else:
          output_dim = 1
        pad_idx = self.TEXT.vocab.stoi[self.TEXT.pad_token]
        # Initialize model
        if self.model_type == 'deep':
          self.model = VDCNN(output_dim, vocab_size, pad_idx,
                             self.embedding_dim, shortcut=True)
        elif self.model_type == 'master':
          self.model = SUPER_CNN(vocab_size, self.embedding_dim, 
                                 self.num_filters, self.filter_sizes, 
                                 output_dim, self.dropout, pad_idx, use_base=self.use_base)
        else:
          if DUAL_FC_LAYERS:
            self.model = CNN_2FC(vocab_size, self.embedding_dim, self.num_filters,
                                 self.filter_sizes, output_dim, self.dropout,
                                 pad_idx)
          else:
            self.model = CNN(vocab_size, self.embedding_dim, self.num_filters,
                             self.filter_sizes, output_dim, self.dropout,
                             pad_idx)
        # Get pretrained weights from vocab
        pretrained_embeddings = self.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(pretrained_embeddings)

        # zero initial weights of <unk> and <pad>
        unk_index = self.TEXT.vocab.stoi[self.TEXT.unk_token]

        self.model.embedding.weight.data[unk_index] = torch.zeros(self.embedding_dim)
        self.model.embedding.weight.data[pad_idx] = torch.zeros(self.embedding_dim)

    def init_optimizer(self):
        """ Initializes the optimizor (Adam) and the criterion (BCEWithLogitsLoss)
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.model = self.model.to(self.device)
        if self.criterion_str == "CE":
            self.criterion = nn.CrossEntropyLoss()
        if self.criterion_str == "WCE":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10]))
        elif self.criterion_str == "BCE":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.criterion_str == "Reg":
            self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(self.device)

    def train_model(self, num_epochs):
        """ Trains and validates the model (and prints the results) for the
            given number of epochs. Saves the model from the epoch which
            yeilded the lowest validation loss.

        Args:
            num_epochs (int): number of epochs to train the model
        """
        print(self.model_name)
        best_valid_loss = float('inf')

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc, _, _ = self.evaluate_epoch(self.val_iter)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # torch.save(self.model.state_dict(), f'{self.model_name}.pt')
                torch.save(self.model, f'{self.model_name}.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    def count_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'The model has {num_params:,} trainable parameters')

    def train_epoch(self):
        """ Trains the model for 1 epoch of the train iterator

        Returns:
            tuple: (loss, accuracy)
        """
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in self.train_iter:
            # print(batch.__dict__)
            self.optimizer.zero_grad()
            predictions = self.model(getattr(batch, batch.input_fields[0]))
            # Reduce dimentionality if using binary cross entropy loss
            if self.criterion_str == "BCE":
                predictions = predictions.squeeze(1)
                acc_formula = binary_accuracy
            elif self.criterion_str == "CE":
                acc_formula = categorical_accuracy 
            loss = self.criterion(predictions, getattr(batch, batch.target_fields[0]))
            acc = acc_formula(predictions, getattr(batch, batch.target_fields[0]))
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.train_iter), epoch_acc / len(self.train_iter)

    def evaluate_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        logits_list = []
        labels_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                predictions = self.model(getattr(batch, batch.input_fields[0]))
                logits_list.append(predictions)
                labels_list.append(getattr(batch, batch.target_fields[0]))
                # Reduce dimentionality if using binary cross entropy loss
                if self.criterion_str == "BCE":
                    predictions = predictions.squeeze(1)
                    acc_formula = binary_accuracy
                elif self.criterion_str == "CE":
                    acc_formula = categorical_accuracy
                loss = self.criterion(predictions, getattr(batch, batch.target_fields[0]))
                acc = acc_formula(predictions, getattr(batch, batch.target_fields[0]))

                epoch_loss += loss.item()
                epoch_acc += acc.item()
        logits_list = torch.cat(logits_list)
        labels_list = torch.cat(labels_list)
        # softmaxes = F.softmax(logits_list, dim=1)
        # _, predictions_list = torch.max(softmaxes, dim=1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if logits_list[0].shape[0] == 1:
            predictions_list = torch.round(torch.sigmoid(logits_list))
            ece_criterion = ECE_binary().to(device)

        else:
            softmaxes = F.softmax(logits_list, dim=1)
            _, predictions_list = torch.max(softmaxes, dim=1)
            #create ece class object
            ece_criterion = ECE_multiclass().to(device)

        # # first input is raw logits tensor (not softmaxed) and the second is a tensor of correct labels 
        ece_test = ece_criterion(logits_list, labels_list).item()
        
        f1 = f1_score(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy(), average='macro')
        return epoch_loss / len(iterator), epoch_acc / len(iterator) , f1, ece_test

    def test_model(self):
        """ Tests the model with the highest validation loss on the test iterator
        """
        # self.model.load_state_dict(torch.load(f'{self.model_name}.pt'))
        torch.load(f'{self.model_name}.pt')
        test_loss, test_acc, f1, ece = self.evaluate_epoch(self.test_iter)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%\n | F1 Score: {f1:.3f} | Calibration Error: {ece:.3f} |')
        #print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}')
        with open('model_results.txt', "a") as f:
            # f.write(f'{self.model_name}\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%\n | F1 Score: {f1:.3f} | Calibration Error: {ece:.3f} |'  )
            f.write(f'{self.model_name}\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f} |'  )

    def do_what_we_want(self, num_epochs=10):
        self.load_dataset()
        self.build_vocab()
        self.init_model()
        self.init_optimizer()
        self.train_model(num_epochs)
        self.test_model()


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
  

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class ECE_binary(nn.Module):

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE_binary, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        predictions = torch.round(torch.sigmoid(logits))
        confidences = []
        for x in logits:
          confidences.append(max(1-x,x))
        confidences = torch.Tensor(confidences)
        accuracies = predictions.eq(labels.unsqueeze(1))
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class ECE_multiclass(nn.Module):

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE_multiclass, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

# clear contents of file
with open("model_results.txt", "w") as f:
    pass

'''
# ======================== FORMALITY BINARY MODEL ========================= #
trainer = ModelTrainer(
    model_name="CNN-formality-binary", 
    text_field_name='Sentence', 
    label_field_name='Formality', 
    train_file_name='mturk_news_formality_BINARY.csv', 
    criterion_str='BCE',
    dtype=torch.float)
trainer.do_what_we_want(num_epochs=8)


# ======================== INFORMATIVENESS BINARY MODEL ========================= #
trainer = ModelTrainer(
    model_name="CNN-informativeness-binary", 
    text_field_name='Sentence', 
    label_field_name='Informativeness', 
    train_file_name='mturk_news_informativeness_reg.csv', 
    criterion_str='BCE',
    dtype=torch.float)
trainer.do_what_we_want(num_epochs=8)


# ======================== EMOTION MODEL ========================= #
trainer = ModelTrainer(
    model_name="CNN-emotion", 
    text_field_name='Text', 
    label_field_name='Emotion', 
    train_file_name='emotion_train.csv', 
    test_file_name='emotion_test.csv',
    criterion_str='CE'
)
trainer.do_what_we_want(num_epochs=8)

# ======================== SENTIMENT MODEL ========================= #
trainer = ModelTrainer("CNN-sentiment", IMDB=True, criterion_str="BCE", dtype=torch.float)
trainer.do_what_we_want(num_epochs=8)

# ======================== SARCASM MASTER MODEL ========================= #
trainer = ModelTrainer(
    model_name="CNN-sarcasm-master-without-base",
    model_type="master",
    text_field_name='headline', 
    label_field_name='is_sarcastic', 
    train_file_name='Sarcasm_Headlines_Dataset_v2.csv', 
    criterion_str='BCE',
    dtype=torch.float,
    use_base=False)
trainer.do_what_we_want(num_epochs=8)
'''

# after 1 iteration of the loop the seed gets set, so to get random combinations you need to only run it once then rerun the code. It's annoying.
num_random_iterations = 1
for _ in range(num_random_iterations): 

    batch_size = random.choice([32, 64])
    embed = 100
    drop = random.uniform(0.1, 0.5)
    filters = random.choice([100,200,300,400,500])
    filter_list = [[3, 4, 5], [2, 3, 4, 5], [3, 4, 5, 7], [2, 3, 4, 5, 7], [3, 5]]
    filter_comb = random.choice(filter_list)
    LR = random.uniform(1e-4, 1e-2)   
    print("start tuning")
    print('lr: {}, filters: {}, batch size: {}, dropout: {}, embedding dim: {}, NUM_FEATURES: {}, filter sizes: {}'.format(LR, filters, batch_size, drop, embed, NUM_FEATURES, filter_comb))   
    epochs = 8    

    # ======================== FORMALITY BINARY MODEL ========================= #
    trainer = ModelTrainer(
        model_name="CNN-formality-binary", 
        text_field_name='Sentence', 
        label_field_name='Formality', 
        train_file_name='mturk_news_formality_BINARY.csv', 
        embedding_dim=embed,
        criterion_str='BCE',
        num_filters=filters,
        filter_sizes=filter_comb,
        dropout=drop,
        dtype=torch.float,
        batch_size=batch_size
        )
        
    trainer.do_what_we_want(num_epochs=epochs)


    # ======================== INFORMATIVENESS BINARY MODEL ========================= #
    trainer = ModelTrainer(
        model_name="CNN-informativeness-binary", 
        text_field_name='Sentence', 
        label_field_name='Informativeness', 
        train_file_name='mturk_news_informativeness_BINARY.csv', 
        embedding_dim=embed,
        criterion_str='BCE',
        num_filters=filters,
        filter_sizes=filter_comb,
        dropout=drop,      
        dtype=torch.float,
        batch_size=batch_size)
    trainer.do_what_we_want(num_epochs=epochs)

    # ======================== EMOTION MODEL ========================= #
    trainer = ModelTrainer(
        model_name="CNN-emotion", 
        text_field_name='Text', 
        label_field_name='Emotion', 
        train_file_name='emotion_train.csv', 
        test_file_name='emotion_test.csv',
        embedding_dim=embed,
        criterion_str='CE',
        num_filters=filters,
        filter_sizes=filter_comb,
        dropout=drop,
        batch_size=batch_size
    )
    trainer.do_what_we_want(num_epochs=epochs)

    # ======================== SENTIMENT MODEL ========================= #
    trainer = ModelTrainer("CNN-sentiment", IMDB=True, embedding_dim=embed, criterion_str="BCE",num_filters=filters,filter_sizes=filter_comb, dropout=drop, dtype=torch.float, batch_size=batch_size)
    trainer.do_what_we_want(num_epochs=epochs)

    # ======================== SARCASM MASTER MODEL ========================= #
    trainer = ModelTrainer(
        model_name="CNN-sarcasm-master-without-base",
        model_type="master",
        text_field_name='headline', 
        label_field_name='is_sarcastic', 
        train_file_name='Sarcasm_Headlines_Dataset_v2.csv', 
        embedding_dim=embed,       
        criterion_str='BCE',
        num_filters=filters,
        filter_sizes=filter_comb,
        dropout=drop,
        dtype=torch.float,
        use_base=False,
        batch_size=batch_size)
    trainer.do_what_we_want(num_epochs=epochs)