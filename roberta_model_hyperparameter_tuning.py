#Get rid of annoying tensorflow warnings
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import AdamW, RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import os
import sys
from torch import nn, optim
from torch.nn import functional as F
import json
import random

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
        
# template of model we will make. Unfinished!!!!
class RoBERTa_BiLSTM(nn.Module):
    def __init__(self, num_classes, num_layers, hidden_dim, seq_length, drop=0.2):
        super(RoBERTa_BiLSTM, self).__init__()
        self.lstm_size = hidden_dim
        self.embedding_dim = 768 # weird, but it's the size of the RoBERTa hidden state/vectors, so we're fine
        self.num_layers = num_layers # weird, maybe?
        self.num_classes = num_classes
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=drop, batch_first=True, bidirectional=True
        )
        self.downsample = nn.Linear(2*self.lstm_size + self.embedding_dim , self.lstm_size)
        self.pooling = nn.MaxPool1d(seq_length-1)
        self.fc = nn.Linear(self.lstm_size , self.num_classes)

    def forward(self, init_state, input_ids, attention_mask):
        embed = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = embed[0] # batch_size, seq_length, hidden_size
        embeddings = hidden_state[:, 1:]
        
        output, h_state = self.lstm(embeddings, init_state)
        
        cat = torch.cat((output,embeddings),2)
        state = self.downsample(cat) # y.size() = (batch_size, num_sequences, hidden_size)
        
        state = state.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
        
        state = self.pooling(state) # y.size() = (batch_size, hidden_size, 1)
        
        state = state.squeeze(2)
        
        logits = self.fc(state)
        return logits

    def init_state(self, batch_size=1):
        return (torch.zeros(2*self.num_layers, batch_size, self.lstm_size),
            torch.zeros(2*self.num_layers, batch_size, self.lstm_size))

class ECE(nn.Module):

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
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
        
        
def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)


def Extract_Headlines():
    #Open Sarcasm JSON File
    #f = open ('Sarcasm_Headlines_Dataset_v2_Copy.json', "r")
    f = open ('Sarcasm_Headlines_Dataset_v2.json', "r")
    data = json.loads(f.read())
    f.close()
    labels = [] # list of correct labels
    headlines = [] #list of headlines

    max_length = 0
    #get data from file
    for item in data:
        label = int(item['is_sarcastic'])
        headline = item['headline']
        labels.append(label)
        headlines.append(headline)
        if len(headline.split()) > max_length:
            max_length = len(headline.split())


    #convert to numpy array before use
    return labels, headlines, max_length

#pick batch size and number of epochs
def Run_Model(device, batch_size, num_epochs, learningrate=2e-5):

    gold_labels, sentences, MAX_LEN  = Extract_Headlines()

    #number of classes
    num_classes = 2
    MAX_LEN = 64

    # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_classes, output_hidden_states=True)

    tokenizer =  RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

    #tokenize
    encoded_inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=MAX_LEN)

    input_ids = encoded_inputs["input_ids"]
    attention_masks = encoded_inputs["attention_mask"]


    #Split data into train/test/validation
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, gold_labels, random_state=2020, test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2020, test_size=0.2)

    temp_val = validation_inputs
    validation_inputs, test_inputs, validation_labels, test_labels = train_test_split(temp_val, validation_labels, random_state=2020, test_size=0.5)
    validation_masks,test_masks, _, _ = train_test_split(validation_masks, temp_val, random_state=2020, test_size=0.5)

    #Package data into dataloaders
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_masks)


    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size*4)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size*4)
    
    best_valid_acc = 0 
    
    # Run for however many times you want 
    num_random_iterations = 5
    for _ in range(num_random_iterations):
    
        hidden_size = random.choice([64,128,256, 768])
        drop = random.uniform(0.1, 0.4)
        layers = random.choice([1,2,3])

        model = RoBERTa_BiLSTM(num_classes=2, num_layers=layers, hidden_dim=hidden_size, seq_length=MAX_LEN, drop=drop)

        #send model to GPU
        model.to(device)
        
        batch_size = random.choice([16, 32, 48])
        learning_rate = random.uniform(2e-5, 5e-5)
      
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        #Set paramters and optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        #keep track of loss for plotting
        train_loss_set = []
        epochs = 4
        criterion = nn.CrossEntropyLoss()

        #training loop
        print("start training")
        print('lr: {}, batch size: {}, dropout: {}, lstm hidden size: {}, num layers: {}'.format(learning_rate, batch_size, drop, hidden_size,layers))
        for epoch in trange(epochs, desc="Epoch"):
            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0,0

            for step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(device) for t in batch)
                
                b_input_ids, b_input_mask, b_labels = batch
                state_h, state_c = model.init_state(batch_size=b_labels.shape[0])
                state_h = state_h.to(device)
                state_c = state_c.to(device)
                optimizer.zero_grad()

                b_input_ids = b_input_ids.clone().detach().to(device).long()
                outputs = model((state_h, state_c), b_input_ids, attention_mask=b_input_mask)
                #print(outputs.shape)
                #loss = criterion(outputs.transpose(1,2), b_labels)
                #loss = criterion(outputs, b_labels)
                loss = criterion(outputs.view(-1, 2),b_labels.view(-1))
                # loss, logits, states = outputs
                # print(outputs)
                train_loss_set.append(loss.item())
                loss.backward()
                optimizer.step()
                #scheduler.step()

                #gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            print(" train loss: {}".format(tr_loss/nb_tr_steps))
            #plot training loss
            plt.figure(figsize=(15,8))
            plt.title("Training loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.plot(train_loss_set)
            #plt.show()


            # Model Validation
            model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                state_h, state_c = model.init_state(batch_size=b_labels.shape[0])
                state_h = state_h.to(device)
                state_c = state_c.to(device)
                b_input_ids = b_input_ids.clone().detach().to(device).long()
                with torch.no_grad():
                    logits = model((state_h, state_c), b_input_ids, attention_mask=b_input_mask)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
            print("validation accuracy: {}".format(eval_accuracy/nb_eval_steps))
            valid_acc = eval_accuracy/nb_eval_steps
            #add code here to save model to file if it has better performance than before
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                roberta_cnn = "roberta_cnn"
                torch.save(model, f'{roberta_cnn}.pt')
                print('epoch: {}, lr: {}, batch size: {}, dropout: {}, lstm hidden size: {}, num layers: {}'.format(epoch, learning_rate, batch_size, drop, hidden_size,layers))
        
        
    model = torch.load(f'{roberta_cnn}.pt')
    #Test phase
    model.eval()

    logits_list = []
    labels_list = []

    for batch in test_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      state_h, state_c = model.init_state(batch_size=b_labels.shape[0])
      state_h = state_h.to(device)
      state_c = state_c.to(device)
      with torch.no_grad():
        logits = model((state_h, state_c), b_input_ids, attention_mask=b_input_mask)
      logits_list.append(logits)
      labels_list.append(b_labels)
    # Flattened list of logits and the corresponding labels
    logits_list = torch.cat(logits_list)
    labels_list = torch.cat(labels_list)
    softmaxes = F.softmax(logits_list, dim=1)
    _, predictions_list = torch.max(softmaxes, dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list.detach().cpu().numpy(),predictions_list.detach().cpu().numpy(), average='weighted')
    acc = accuracy_score(labels_list.detach().cpu().numpy(), predictions_list.detach().cpu().numpy())
    print('F1 Score: {}, Precision: {}, Recall: {}, Accuracy: {}'.format(f1, precision,recall,acc ))
    ece_criterion = ECE().to(device)
    ece_test = ece_criterion(logits_list, labels_list).item()
    print('ECE on test data: {}'.format(ece_test))

    return model

def Predict_Sample(device, model, sentence, max_len=100):
    tokenizer =  RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    input_id = inputs["input_ids"]
    input_mask = inputs["attention_mask"]
    input_id = torch.tensor(input_id)
    input_mask = torch.tensor(input_mask)
    model.eval()
    state_h, state_c = model.init_state(batch_size=1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    logits = model((state_h, state_c), input_id.clone().detach().to(device).long(), attention_mask=input_mask.to(device))
    prediction = logits
    softmax_logits = F.softmax(prediction, dim=1)
    score, predicted = softmax_logits.max(1)
    print('The predicted score for your input sentence "{}"  is Class {} with confidence: {}'.format(sentence,predicted.item(), score.item()))

def main():
    # use gpu otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # If you get an out of memory error reduce the batch size
    model = Run_Model(device, batch_size=32, num_epochs=2, learningrate=2e-5)
    # enter any sentence you want here
    input_sentence = "Man dies of excitement after watching paint dry."
    print("checkpoint")
    Predict_Sample(device, model, input_sentence)

if __name__ == '__main__':
    main()