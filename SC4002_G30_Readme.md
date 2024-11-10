# Sentiment Classification using Pretrained Word Embeddings

## Project Overview
This project implements a sentiment classification system using pretrained word embeddings (Word2vec or Glove) and various neural network architectures (RNN, biLSTM, biGRU, CNN). The objective is to classify sentences based on sentiment labels using a range of model enhancements and techniques for handling out-of-vocabulary (OOV) words.

## Group Information
- **Course**: SC4002
- **Group ID**: GRP 30
- **Group Members**:
  - Leong Kit Ye
  - Abdul Rahim Sahana 
  - Banerjee Mohor
  - Lim Jun Rong, Ryan
  - Augustine Jesuraj Senchia Gladine 
  - Jeyaseelan Harish Vasanth

## Instructions



## Model and Notebook Mapping

This table maps the different parts of the model development and evaluation process to the corresponding notebooks used in the project. Each part of the process is divided into specific tasks, and the table shows which notebook handles those tasks.

| **Part of the Question**                         | **Notebook**                        |
|--------------------------------------------------|-------------------------------------|
| Part 1: Data Preprocessing & Model Setup         | Questions_1_2_3_1_and_2.ipynb      |
| Part 2a: Finding the Best Model Configuration    | Questions_1_2_3_1_and_2.ipynb      |
| Part 2b: Reporting the Accuracy Scores           | Questions_1_2_3_1_and_2.ipynb      |
| Part 2c: Other Methods of Final Sentence Representation | Questions_1_2_3_1_and_2.ipynb      |
| Part 3: RNN with Trainable embedings                     | Questions_1_2_3_1_and_2.ipynb|    
| Part 3: BiLSTM Model Results                   | FinalBiLSTMRun.ipynb               |
| Part 3: BiGRU Model Results                              | Final_BiGRU_Run.ipynb              |
| Part 3: CNN Model Results                                | Final_Cnn_Run.ipynb        
| Part 3: Further Improvement Strategy (Ensemble Model) | Final_Ensemble_Run.ipynb           |
        

## Dataset Preparation

To view the code for these steps, open the `Questions_1_2_3_1_and_2.ipynb` document.

In preparing the dataset for sentiment classification, we applied several preprocessing steps to optimize tokenization, vocabulary, and manage out-of-vocabulary (OOV) words. 

1. **Tokenization**:  
   - We used `nltk.word_tokenize` to tokenize texts, initially producing a vocabulary of 18,010 unique words.

2. **Punctuation and Stopword Removal**:  
   - Punctuation was removed, and common stopwords were filtered out using NLTK’s stopword corpus. This reduced the vocabulary to 17,879 words.

3. **Word Embeddings**:  
   - We used the `fasttext-wiki-news-subwords-300` Word2Vec model, initially identifying 1,960 OOV words (3,600 without stopword/punctuation removal).

4. **OOV Handling with Subword Embeddings**:  
   - A custom method was implemented to generate subwords for OOV words and average the embeddings of valid subwords found in the model. This reduced the OOV count to just one word.

This streamlined preprocessing ensures a well-prepared vocabulary and utilizes subword embeddings to handle OOV words effectively, enhancing model performance on the dataset.

## Part 2: Model Training & Evaluation - RNN

We trained and evaluated a Recurrent Neural Network (RNN) for sentiment classification. RNNs are well-suited for sequential data processing, making them effective for sentiment analysis, where the order and context of words are key to identifying sentiment. To view the code for these steps, open the `Questions_1_2_3_1_and_2.ipynb` document.

### 2a) Optimizing the Model Configuration
- **Hyperparameter Tuning**: We used Random Search to find the optimal model configuration, tuning parameters like the number of layers, hidden layer size, learning rate, batch size, and dropout rate.
- **Training Process**: Validation accuracy was monitored each epoch, with early stopping applied after 5 epochs without improvement, converging in 15 epochs.

| Hyperparameter       | Value |
|----------------------|-------|
| Number of layers     | 1     |
| Hidden layer size    | 128   |
| Learning rate        | 0.001 |
| Batch size           | 32    |
| Dropout rate         | 0.5   |

*Table 1: Best RNN model hyperparameter configuration*

### 2b) Model Performance
The RNN showed limited performance on the validation and test datasets, potentially due to the small training set, resulting in high dropout to combat overfitting.

**Results:**


- Test Accuracy: 0.76
- Precision: 0.76
- F1 Score: 0.77


![Confusion matrix for Simple RNN](https://imgur.com/afuNFUU.png)


### 2c) Sentence Representation Methods
We compared two methods for deriving sentence representations:

1. **Last Time Step Representation**: Uses the last hidden state of the RNN, encapsulating information from the entire sequence.
2. **ConcatPooling**: Combines multiple aspects of the RNN's output by concatenating the first hidden state, the last hidden state, and the max-pooled representation across all time steps.



**Results for Last Time Step Representation:**

- Test Accuracy: 0.54
- Precision: 0.54
- F1 Score: 0.47


![Confusion matrix for Average Pooling](https://imgur.com/hAuMHAV.png)


**Results for Concat Pooling Representation:**

- Test Accuracy: 0.76
- Precision: 0.75
- F1 Score: 0.76

![Confusion matrix for Average Pooling](https://imgur.com/q4a1fZi.png)



# Part 3: Model Enhancements

In this section, we experiment with various techniques to improve the performance of our sentiment classification model. These include using trainable embeddings, exploring BiLSTM and BiGRU models, and testing a CNN model.

### 3a) RNN with Trainable Embeddings



**Results:**
- Test Accuracy: 0.78
- Precision: 0.78
- F1 Score: 0.78

![Training Curve for RNN with Trainable embeddings](https://imgur.com/7dUOZYK.png)

![Confusion Matrix for RNN with Trainable embeddings](https://imgur.com/eifLaki.png)


After this, we experimented with our subword embedding solution to mitigate the influence of OOV vocabulary. This approach helped reduce the number of OOV words and allowed the model to handle previously unseen words better, improving its overall performance.

---
### 3b) RNN with OOV Mitigation

To view the code for these steps, open the `Questions_1_2_3_1_and_2.ipynb` document.

**Results:**
- Test Accuracy: 0.79
- Precision: 0.80
- F1 Score: 0.78

![RNN with OOV Mitigation](https://imgur.com/1f577sl.png)

![onfusion Matrix for RNN with OOV Mitigation](https://imgur.com/zmZF4X3.png)    


### 3c) BiLSTM

To view the code for these steps, open the `FinalBiLSTMRun.ipynb` document.


**Results:**
- Test Accuracy: 0.78
- Precision: 0.78
- F1 Score: 0.78

**Figure 4**: Training and validation loss and accuracy histories for BiGRU.

![Training Curves for BiLSTM](https://imgur.com/NqdVPmd.png)

**Figure 5**: Confusion matrix for BiGRU performance on the test set.

![onfusion Matrix for BiLSTM](https://imgur.com/7ZYGLP2.png)    


---

### 3d) BiGRU

To view the code for these steps, open the `Final_BiGRU_Run.ipynb` document.

**Results:**
- Test Accuracy: 0.78
- Precision: 0.74
- F1 Score: 0.79

**Figure 6**: Training and validation loss and accuracy histories for BiGRU.

![Training Curves for BiGRU](https://imgur.com/qi8UV4l.png)

**Figure 7**: Confusion matrix for BiGRU performance on the test set.

![onfusion Matrix for BiGRU](https://imgur.com/CSKgxfz.png)


---

### 3e) CNN

To view the code for these steps, open the `Final_Cnn_Run.ipynb` document.

**Results:**


![Training Curves for CNN](https://imgur.com/lBFScBG.png)

![Confusion Matrix for CNN](https://i.imgur.com/B5Y4WiW.png)


- Test Accuracy: 0.78
- Precision: 0.77
- F1 Score: 0.78

### 3e) Further Improvement Strategy: Ensemble Model Architecture

To view the code for these steps, open the `Final_Ensemble_Run.ipynb` document.


**Results:**


![Training Curves for Ensemble Approach](https://imgur.com/Jl1zMQy.png)

![Confusion Matrix for Ensemble Approach](https://imgur.com/B5Y4WiW.png)



---

### Summary of Results

We compared the performance of all these models using accuracy, precision, and F1 score metrics. Here’s a summary of the results:

### Final Model Performance Metrics

| Model                         | Test Accuracy | Precision | F1 Score |
|-------------------------------|---------------|-----------|----------|
| Simple RNN                            | 0.54          | 0.55      | 0.46     |
| Simple RNN with Concat Pooling                         | 0.76         | 0.75      | 0.76     |
| RNN with Trainable Embeddings|      0.78         |       0.78    |     0.78     |
| RNN with OOV Mitigation      |          0.79     |     0.80      |       0.78   |
| BiLSTM                         | 0.78          | 0.78      | 0.78     |
| BiGRU                          | 0.78          | 0.74      | 0.79     |
| CNN                            | 0.78          | 0.77      | 0.78     |
| **Ensemble Approach**                   | **0.79**      | **0.78**  | **0.80** |


