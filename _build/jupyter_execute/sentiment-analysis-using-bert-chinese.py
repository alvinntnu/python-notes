# Sentiment Analysis Using BERT

- Using `ktrain` for modeling
  - The ktrain library is a lightweight wrapper for tf.keras in TensorFlow 2, which is "designed to make deep learning and AI more accessible and easier to apply for beginners and domain experts".
- Easy to implement BERT-like pre-trained language models
- This notebook works on sentiment analysis of Chinese movie reviews, which is a small dataset. I would like to see to what extent the transformers are effective when dealing with relatively smaller training set. This in turn shows us the powerful advantages of transfer learning. 

## Installing ktrain

!pip install ktrain


## Importing Libraries

import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf

tf.__version__

## Clone Git Repository for Data

## Will need this if data is available on GitHub
# !git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git

## Data Preparation

- Mount the Google Drive first (manually via the tabs on the left of Google Colab

- The default path of the input data csv file is:

```
GOOGLE_DRIVE_ROOT/ColabData/marc_movie_review_metadata.csv
```
- In BERT, there is no need to do word segmentation. The model takes in the raw reviews as the input.

## loading the train dataset
## change the path if necessary
data = pd.read_csv('/content/drive/My Drive/ColabData/marc_movie_review_metadata.csv', dtype= str)[['reviews','rating']]
data = data.rename(columns={'reviews':'Reviews', 'rating':'Sentiment'})
data.head()

from sklearn.model_selection import train_test_split

data_train, data_test = train_test_split(data, test_size=0.1)


## dimension of the dataset

print("Size of train dataset: ",data_train.shape)
print("Size of test dataset: ",data_test.shape)

#printing last rows of train dataset
data_train.tail()

#printing head rows of test dataset
data_test.head()

## Train-Test Split

Models supported by transformers library for tensorflow 2:

- **BERT**: bert-base-uncased, bert-large-uncased,bert-base-multilingual-uncased, and others.
- **DistilBERT**: distilbert-base-uncased distilbert-base-multilingual-cased, distilbert-base-german-cased, and others
- **ALBERT**: albert-base-v2, albert-large-v2, and others
- **RoBERTa**: roberta-base, roberta-large, roberta-large-mnli
- **XLM**: xlm-mlm-xnli15–1024, xlm-mlm-100–1280, and others
- **XLNet**: xlnet-base-cased, xlnet-large-cased


# text.texts_from_df return two tuples
# maxlen means it is considering that much words and rest are getting trucated
# preprocess_mode means tokenizing, embedding and transformation of text corpus(here it is considering BERT model)


(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train,
                                                                   text_column = 'Reviews',
                                                                   label_columns = 'Sentiment',
                                                                   val_df = data_test,
                                                                   maxlen = 250,
                                                                   lang = 'zh-*',
                                                                   preprocess_mode = 'bert') # or distilbert

## size of data
print(X_train[0].shape, y_train.shape)
print(X_test[0].shape, y_test.shape)

## Define Model

## use 'distilbert' if you want
model = text.text_classifier(name = 'bert', # or distilbert
                             train_data = (X_train, y_train),
                             preproc = preproc)

## Define Learner

#here we have taken batch size as 6 as from the documentation it is recommend to use this with maxlen as 500
learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                   val_data = (X_test, y_test),
                   batch_size = 6)

## Estimate Learning Rate (Optional)

- A nice artilce on how to interpret learning rate plots. See [Keras Learning Rate Finder](https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/).

learner.lr_find(show_plot=True, max_epochs=2)

## Fit and Save Model

#Essentially fit is a very basic training loop, whereas fit one cycle uses the one cycle policy callback

learner.fit_onecycle(lr = 2e-5, epochs = 1)
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('/content/drive/My Drive/ColabData/bert-ch-marc')


## Evaluation

y_pred=predictor.predict(data_test['Reviews'].values)

# classification report

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))


y_true = data_test['Sentiment'].values
# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)

# Recall
from sklearn.metrics import recall_score
recall_score(y_true, y_pred, average=None)

# Precision
from sklearn.metrics import precision_score
precision_score(y_true, y_pred, average=None)

# F1
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average=None)


## AUC-ROC Curve
y_pred_proba = predictor.predict(data_test['Reviews'].values, return_proba=True)
print(predictor.get_classes()) # probability of each class
print(y_pred_proba[:5,])

y_true_binary = [1 if label=='positive' else 0 for label in y_true]
y_pred_proba_positive = y_pred_proba[:,1]
y_pred_proba_negative = y_pred_proba[:,0]
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true_binary, y_pred_proba_positive)


# import sklearn.metrics as metrics
# # calculate the fpr and tpr for all thresholds of the classification
# probs = model.predict_proba(X_test)
# preds = probs[:,1] # 
fpr, tpr, threshold = metrics.roc_curve(y_true_binary, y_pred_proba_positive)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
palette = plt.get_cmap('Set1')
print(palette)
plt.figure(dpi=150)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc, color=palette(2))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--', color=palette(3))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## ggplot2 version
# ## prettier?


## Add label for color aesthetic setting
f = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
f['label']= ['X' for i in range(f.shape[0])]


from plotnine import *

g = (
    ggplot(f, aes('fpr', 'tpr', color='label'))
    + geom_line( size=1)
    + geom_abline(linetype='dashed', color="lightpink")
    + labs(x = 'False Positive Rate', 
           y = 'True Positive Rate',
           title="ROC Curve")
    + scale_color_discrete(labels=['AUC = %0.2f' % roc_auc],name = ' ')
    + theme(legend_position = (.75,.25),
            legend_background = element_rect(fill='lightblue', 
                                             alpha=0.2,
                                             size=0.5, linetype="solid",
                                             colour = None)))
g


# g.save('/content/drive/My Drive/ColabData/ggplot-roc.png', width=12, height=10, dpi=300)

## Prediction and Deployment

#sample dataset to test on

data = ['前面好笑看到後面真的很好哭！推薦！',
        '也太浪費錢了，劇情普普，新鮮度可以再加強',
        '人生一定要走一遭電影院',
        '不推',
        '帶六歲孩子看，大人覺得小孩看可以，小孩也覺得好看',
        '我想看兩人如何化解對方的防線，成為彼此的救贖；在人生旅途中，留下最深刻的回憶。',
        '這部新恐龍是有史以來最好看的哆啦電影版，50週年紀念作當之無愧，真的太感人了，並且彩蛋多到數不完，這部電影不僅講述了勇氣、友誼與努力不懈的精神，也以大雄的視角來看父母的心情，不管是劇情、畫面都是一流，另外配樂非常到位，全程都不會無聊，非常推薦大人、小孩、父母去電影院看，絕對值得。',
        '看完之後覺得新不如舊，還是大雄的恐龍好看，不管是劇情還是做畫，都是大雄的恐龍好，而且大雄的新恐龍做畫有點崩壞，是有沒有好好審查啊!以一部50周年慶的電影來說有點丟臉，自從藤子不二雄過世後，哆啦A夢的電影就一直表現平平，沒有以前的那份感動。']


predictor.predict(data)

#return_proba = True means it will give the prediction probabilty for each class

predictor.predict(data, return_proba=True)

#classes available

predictor.get_classes()

## zip for furture deployment
# !zip -r /content/bert.zip /content/bert

## Deploy Model

# #loading the model

predictor_load = ktrain.load_predictor('/content/drive/My Drive/ColabData/bert-ch-marc')

# #predicting the data

predictor_load.predict(data)

## References

- [`ktrain` module](https://github.com/amaiya/ktrain)
- [Sentiment Classification Using Bert](https://kgptalkie.com/sentiment-classification-using-bert/)
- [當Bert遇上Keras：這可能是Bert最簡單的打開姿勢](http://www.ipshop.xyz/15376.html)
- [進擊的 BERT：NLP 界的巨人之力與遷移學習](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)
- [Text Classification with Hugging Face Transformers in TensorFlow 2 (Without Tears)](https://towardsdatascience.com/text-classification-with-hugging-face-transformers-in-tensorflow-2-without-tears-ee50e4f3e7ed)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Huggingface Pre-trained Models](https://huggingface.co/transformers/pretrained_models.html)