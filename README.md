# BERT-Sentiment-Classification

## Sentiment Analysis of IMDb Movie Reviews Using BERT

This project implements a binary sentiment analysis classifier for IMDb movie reviews using a pre-trained BERT model. Reviews are preprocessed and transformed into input suitable for BERT, and ratings are converted into binary labels representing positive and negative sentiments. The model is fine-tuned on the processed dataset, with training and validation loops designed to evaluate performance metrics such as accuracy, precision, recall, and F1 score. The goal is to leverage BERT's language understanding capabilities to accurately classify movie reviews based on sentiment.

## Table of Contents
- [Imports and Prerequisites](#imports-and-prerequisites)
- [Helper Functions](#helper-functions)
- [Data Filtering](#data-filtering)
- [BERT Model Class](#bert-model-class)
- [Training Function](#training-function)
- [Validation](#validation-function)
- [Main](#main)
- [Result Analysis](#result-analysis)
- [Conclusion](#conclusion)

## Imports and Prerequisites

- **Purpose:** Install necessary libraries and set up the environment.
- **Key Libraries:**
  - torch and torch.nn: For building and training the neural network.
  - transformers: For utilizing the pre-trained BERT model and tokenizer.
  - pandas: For data manipulation.
  - nltk: For natural language processing tasks.
- **Device Configuration:** Utilizes GPU if available for faster computations.

```python
!pip install torchtext==0.10.0 --quiet
!pip install transformers==4.11.3 --quiet

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, classification_report, auc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import re
import string
import warnings
import spacy
import nltk
import torchtext
import transformers
from transformers import BertTokenizer, BertModel
import numpy as np
import os
import collections

nltk.download('punkt')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

```

## Helper Functions

- `removeSpecialChars(text)`

- Removes all special characters from the input text using regular expressions.

```python
def removeSpecialChars(text):
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
    return text
```

- `transform(data)`

  - Converting to lowercase.
  - Removing punctuation.
  - Removing special characters.
  - Stripping HTML tags.

```python
def transform(data):
    data = data.str.lower()
    data = data.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    data = data.apply(removeSpecialChars)
    data = data.apply(lambda x: re.compile(r'<[^>]+>').sub('', x))
    return data
```

- `ratingToInt(data)`

  - Converts the rating scores into binary labels

```python
def ratingToInt(data):
    data['rating'] = data['rating'].replace([1.0, 2.0, 3.0, 4.0], 0)
    data['rating'] = data['rating'].replace([7.0, 8.0, 9.0, 10.0], 1)
    return data
```

- `to1D(item)`
  - Converts model outputs into 1D array by selecting the index with the maximum value (i.e., the predicted class).

```python
def to1D(item):
    temp = [[item[i][0], item[i][1]].index(max([item[i][0], item[i][1]])) for i in range(len(item))]
    return temp
```

## Data Filtering

- Cleans the reviews using the transform function.
- Converts ratings to binary labels using ratingToInt.
- Tokenizes the reviews using BERT's tokenizer.
- Encodes the tokenized data into tensors suitable for model input.

```python
def filteringData(trainingData, validationData):
    trainingData['review'] = transform(trainingData['review'])
    validationData['review'] = transform(validationData['review'])
    trainingData = ratingToInt(trainingData)
    validationData = ratingToInt(validationData)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    encodingTrain = tokenizer.batch_encode_plus(
        list(trainingData['review'].values),
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    encodingVal = tokenizer.batch_encode_plus(
        list(validationData['review'].values),
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    return trainingData, validationData, encodingTrain, encodingVal
```


## BERT Model Class

- Defines the neural network architecture using a pre-trained BERT model.
- **BERT Layer:** Loads the pre-trained bert-base-cased model.
- **Dropout Layer:** Prevents overfitting by randomly setting input units to zero during training.
- **Linear Layer:** Maps the BERT output to the number of classes (2 in this case).
- **ReLU Activation:** Applies a rectified linear unit activation function.

```python
class BERT(nn.Module):

    def __init__(self, num_classes, dropValue):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropValue)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        out = self.relu(linear_output)
        return out
```

## Training Function

- Iterates over batches from the training dataloader.
- Computes predictions and loss.
- Performs backpropagation and optimizer step.
- Calculates batch-wise accuracy.

```python
def training(bertModel, optimizer, lossFunction, trainDataloader):
    batch_losses = []
    correct = 0

    for batchID, batchMask, batchY in trainDataloader:
        yPred = bertModel.forward(batchID, batchMask)
        trainLoss = lossFunction(yPred, batchY.long())
        batch_losses.append(trainLoss.item())
        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()
        yPred1D = to1D(yPred)
        yPred1D = torch.tensor(yPred1D, device=device)
        c = (batchY == yPred1D).float()
        correct += c.sum() / len(c)

    print('     Calculated Loss         ----->    ', round(sum(batch_losses) / len(trainDataloader), 4), '\n')
    print('     Calculated Accuracy     ----->    ', np.round((correct / len(trainDataloader)).cpu().detach().numpy(), 4), '\n')
```

## Validation Function

- **Loss:** Average validation loss.
- **Accuracy:** Overall accuracy on the validation set.
- **Recall, Precision, F1 Score:** Computed separately for positive and negative classes.

```python
def validation(bertModel, optimizer, lossFunction, valDataloader):
    validationCheck = 0
    validationBatchLosses = []
    batchRecall0 = []
    batchPrecision0 = []
    batchF1score0 = []
    batchRecall1 = []
    batchPrecision1 = []
    batchF1score1 = []

    for batchID, batchMask, batchY in valDataloader:
        yPred = bertModel.forward(batchID, batchMask)
        valLoss = lossFunction(yPred, batchY.long())
        validationBatchLosses.append(valLoss.item())
        yPred1D = to1D(yPred)
        yPred1D = torch.tensor(yPred1D, device=device)
        temp = (batchY == yPred1D).float()
        validationCheck += temp.sum() / len(temp)
        tempYPredArray = yPred1D.cpu().detach().numpy()
        tempBatchYArray = batchY.cpu().detach().numpy()
        batchRecall1.append(recall_score(tempBatchYArray, tempYPredArray, average='micro', labels=[1.0], zero_division=0))
        batchPrecision1.append(precision_score(tempBatchYArray, tempYPredArray, average='micro', labels=[1.0], zero_division=0))
        batchF1score1.append(f1_score(tempBatchYArray, tempYPredArray, average='micro', labels=[1.0], zero_division=0))
        batchRecall0.append(recall_score(tempBatchYArray, tempYPredArray, average='micro', labels=[0.0], zero_division=0))
        batchPrecision0.append(precision_score(tempBatchYArray, tempYPredArray, average='micro', labels=[0.0], zero_division=0))
        batchF1score0.append(f1_score(tempBatchYArray, tempYPredArray, average='micro', labels=[0.0], zero_division=0))

    print('     Calculated Loss        ----->    ', round(sum(validationBatchLosses) / len(valDataloader), 4), '\n')
    print('     Calculated Accuracy    ----->    ', np.round((validationCheck / len(valDataloader)).cpu().detach().numpy(), 4), '\n')
    print('\n \n')
    print('|| ---------------------------- Negative Reviews ---------------------------- ||')
    print('\n')
    print('     Calculated Recall      ----->    ', np.round(sum(batchRecall0) / len(batchRecall0), 4), '\n')
    print('     Calculated Precision   ----->    ', np.round(sum(batchPrecision0) / len(batchPrecision0), 4), '\n')
    print('     Calculated F1 score    ----->    ', np.round(sum(batchF1score0) / len(batchF1score0), 4))
    print('\n \n')
    print('|| ---------------------------- Positive Reviews ---------------------------- ||', '\n')
    print('\n')
    print('     Calculated Recall      ----->    ', np.round(sum(batchRecall1) / len(batchRecall1), 4), '\n')
    print('     Calculated Precision   ----->    ', np.round(sum(batchPrecision1) / len(batchPrecision1), 4), '\n')
    print('     Calculated F1 score    ----->    ', np.round(sum(batchF1score1) / len(batchF1score1), 4), '\n')
    print('\n \n')
```

## Main

- Loads and preprocesses the data.
- Splits the data into training and validation sets.
- Initializes the model, optimizer, and loss function.
- Trains the model over multiple epochs.
- Validates the model after each epoch.
- Prints training and validation statistics.

```python
# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Path
dataPath = 'drive/MyDrive/ColabNotebooks/haha/imdb-reviews.csv'
print('\nShould take about 3-6 minutes to complete running ...\n')

# Load Dataset
dataset = pd.read_csv(dataPath, sep='\t')

# Split Dataset
trainingData = dataset.sample(frac=0.8, random_state=25)
validationData = dataset.drop(trainingData.index)

# Data Filtering
filteredTraining, filteredValidation, encodingTrain, encodingVal = filteringData(trainingData, validationData)

# Prepare Tensors
input_ids_train = encodingTrain['input_ids']
attention_masks_train = encodingTrain['attention_mask']
labels_train = torch.tensor(filteredTraining.rating.values, device=device)

input_ids_val = encodingVal['input_ids']
attention_masks_val = encodingVal['attention_mask']
labels_val = torch.tensor(filteredValidation.rating.values, device=device)

# Create Datasets
trainDataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
valDataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Hyperparameters
num_classes = 2
dropValue = 0.3
learningRate = 0.00001
batch_size = 8

# Data Loaders
trainDataloader = DataLoader(trainDataset, sampler=RandomSampler(trainDataset), batch_size=batch_size)
valDataloader = DataLoader(valDataset, sampler=RandomSampler(valDataset), batch_size=batch_size)

# Model, Optimizer, Loss Function
bertModel = BERT(num_classes, dropValue)
bertModel.to(device)
optimizer = torch.optim.Adam(bertModel.parameters(), learningRate)
lossFunction = torch.nn.CrossEntropyLoss().to(device)

# Training Loop
for epoch in range(5):
    print('\n \n')
    print('|| #################################### Epoch Number ', epoch + 1, '#################################### ||')
    print('\n \n')
    print('                      ################# Training Stats   ################')
    print('\n')
    training(bertModel, optimizer, lossFunction, trainDataloader)
    print('\n                    ################# Validation Stats ################')
    print('\n')
    validation(bertModel, optimizer, lossFunction, valDataloader)
    print('\n \n')
    print('|| #################################### End of Epoch ', epoch + 1, '#################################### ||')
```

## Result Analysis

After training the model over 5 epochs, the BERT-based sentiment classifier demonstrates strong performance on the IMDb movie reviews dataset. The training and validation statistics indicate that the model effectively learns from the data and generalizes well to unseen reviews.

- **Training Loss:** Consistently decreases over epochs, showing that the model is learning and fitting the training data effectively.
- **Training Accuracy:** Increases steadily with each epoch, reflecting improved performance on the training set.
- **Validation Metrics:** The model maintains high performance on the validation set, suggesting good generalization without significant overfitting.
  - **Validation Loss:** Decreases over epochs, indicating that the model's predictions are becoming more accurate on unseen data.
  - **Validation Accuracy:** Remains high across epochs, demonstrating strong predictive capabilities on the validation set.
  - **Precision and Recall:** High precision and recall values for both positive and negative classes, showing the model's ability to correctly identify sentiments.

## Conclusion

This project successfully demonstrates the application of a pre-trained BERT model for binary sentiment analysis on IMDb movie reviews. By fine-tuning BERT on the specific dataset, the model leverages its deep understanding of language to achieve high accuracy and robust performance metrics.

- **BERT's Effectiveness:** The fine-tuned model achieves strong results, validating BERT's capability in capturing complex linguistic patterns and context necessary for sentiment analysis.
- **Data Preprocessing:** Rigorous cleaning and preprocessing of the text data significantly contribute to the model's performance, ensuring that irrelevant noise does not hinder learning.
- **Evaluation Metrics:** High precision, recall, and F1 scores across both positive and negative classes confirm the model's effectiveness in sentiment classification.
- **Fine-tuning:** The approach demonstrates that fine-tuning a pre-trained language model on a specific task can yield significant performance gains even with a relatively small dataset.
