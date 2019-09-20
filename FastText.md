## Using FastText

### Download pretrained model
#! wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
#! unzip wiki-news-300d-1M.vec.zip -d fasttext_model/
### Install FastText
#! pip install fasttext

### Use FastText
'''
import re
import fasttext

## PreProcess data
input_text = """FastText is an extension to Word2Vec proposed by Facebook in 2016. Instead of feeding individual words into the Neural Network, FastText breaks words into several n-grams (sub-words). For instance, the tri-grams for the word apple is app, ppl, and ple (ignoring the starting and ending of boundaries of words). The word embedding vector for apple will be the sum of all these n-grams. After training the Neural Network, we will have word embeddings for all the n-grams given the training dataset. Rare words can now be properly represented since it is highly likely that some of their n-grams also appears in other words. I will show you how to use FastText with Gensim in the following section.
For skip-gram, the input is the target word, while the outputs are the words surrounding the target words. For instance, in the sentence “I have a cute dog”, the input would be “a”, whereas the output is “I”, “have”, “cute”, and “dog”, assuming the window size is 5. All the input and output data are of the same dimension and one-hot encoded. The network contains 1 hidden layer whose dimension is equal to the embedding size, which is smaller than the input/ output vector size. At the end of the output layer, a softmax activation function is applied so that each element of the output vector describes how likely a specific word will appear in the context. The graph below visualizes the network structure.Word2Vec is an efficient solution to these problems, which leverages the context of the target words. Essentially, we want to use the surrounding words to represent the target words with a Neural Network whose hidden layer encodes the word representation.
There are two types of Word2Vec, Skip-gram and Continuous Bag of Words (CBOW). I will briefly describe how these two methods work in the following paragraphs."""
# remove parenthesis 
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
# store as list of sentences
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
# store as list of lists of words
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)
with open('../datasets/MS_Transcript_sentlevel.txt','a') as f:
     for each_sent in sentences_ted:
         f.write(each_sent+"\n")

## Train UnSupervised
TEXT_FOR_WE_FILENAME = '../datasets/MS_Transcript_sentlevel.txt'
PRETRAINED_VECTOR_DIM = 300
PRETRAINED_FILE =  "fasttext_model/wiki-news-300d-1M.vec"
import fasttext
model_pre = fasttext.train_unsupervised(TEXT_FOR_WE_FILENAME, model='skipgram', dim=PRETRAINED_VECTOR_DIM, pretrainedVectors=PRETRAINED_FILE)
model_pre.save_model("fasttext_model/fasttext_model_unsupervised.bin")

## Train Supervised
TEXT_FOR_WE_FILENAME = '../datasets/MS_Transcript_sentlevel.txt'
PRETRAINED_VECTOR_DIM = 300
PRETRAINED_FILE =  "fasttext_model/wiki-news-300d-1M.vec"
import fasttext
model_pre = fasttext.train_supervised(TEXT_FOR_WE_FILENAME, dim=PRETRAINED_VECTOR_DIM, pretrainedVectors=PRETRAINED_FILE)
model_pre.save_model("fasttext_model/fasttext_model.bin")

## Test
model = fasttext.load_model("fasttext_model/fasttext_model.bin")
print(model['dog'])
'''