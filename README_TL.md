## Transfer Learning

###    Transfer learning scenarios
        Depending on both the size of the new dataset and the similarity of the new dataset to the original dataset, the approach for using transfer learning will be different. Keeping in mind that ConvNet features are more generic in the early layers and more original-dataset specific in the later layers, here are some common rules of thumb for navigating the four major scenarios:

        1. The target dataset is small and similar to the base training dataset.
            Since the target dataset is small, it is not a good idea to fine-tune the ConvNet due to the risk of overfitting. Since the target data is similar to the base data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, we:
                Remove the fully connected layers near the end of the pretrained base ConvNet
                Add a new fully connected layer that matches the number of classes in the target dataset
                Randomize the weights of the new fully connected layer and freeze all the weights from the pre-trained network
                Train the network to update the weights of the new fully connected layers

        2. The target dataset is large and similar to the base training dataset.
            Since the target dataset is large, we have more confidence that we won’t overfit if we try to fine-tune through the full network. Therefore, we:
                Remove the last fully connected layer and replace with the layer matching the number of classes in the target dataset
                Randomly initialize the weights in the new fully connected layer
                Initialize the rest of the weights using the pre-trained weights, i.e., unfreeze the layers of the pre-trained network
                Retrain the entire neural network

        3. The target dataset is small and different from the base training dataset.
            Since the data is small, overfitting is a concern. Hence, we train only the linear layers. But as the target dataset is very different from the base dataset, the higher level features in the ConvNet would not be of any relevance to the target dataset. So, the new network will only use the lower level features of the base ConvNet. To implement this scheme, we:
                Remove most of the pre-trained layers near the beginning of the ConvNet
                Add to the remaining pre-trained layers new fully connected layers that match the number of classes in the new dataset
                Randomize the weights of the new fully connected layers and freeze all the weights from the pre-trained network
                Train the network to update the weights of the new fully connected layers

        4. The target dataset is large and different from the base training dataset.
            As the target dataset is large and different from the base dataset, we can train the ConvNet from scratch. However, in practice, it is beneficial to initialize the weights from the pre-trained network and fine-tune them as it might make the training faster. In this condition, the implementation is the same as in case 3.

### ConvLSTM with St data
### Train & Test
all samples=1616, correct prediction=1607.0
Iter 29: mini-batch loss=0.028019, test acc=0.994431
all samples=14525, correct prediction in train=14471.0
Iter 29: mini-batch loss=0.012483, train acc=0.996282
Precision 1.0
Recall 1.0
f1_score 1.0
Optimization Finished! Max acc=0.9956683168316832
mathod=	ConvLSTM	acc=	0.994430693069307	Learning_rate=	0.001	iter_num=	30	batch_size=	25	hidden_num=	300	l2=	0.001	traintime=	1482.5602488517761	testtime=	5.537383556365967	hopnum=	0	maxacc=	0.9956683168316832
### ConvLSTM_tl with ms data
### Train & Test
all samples=1638, correct prediction in train=1301.0
Iter 19: mini-batch loss=0.547454, train acc=0.794261
Precision 0.8009404388714734
Recall 0.8009404388714734
f1_score 0.8009404388714735
Optimization Finished! Max acc=0.7304469273743017
mathod=	ConvLSTM_tl	acc=	0.7178770949720671	Learning_rate=	0.001	iter_num=	20	batch_size=	25	hidden_num=	300	l2=	0.001	traintime=	104.86466407775879	testtime=	0.8213441371917725	hopnum=	0	maxacc=	0.7304469273743017

## ConvLSTM_tl with St_test Test
inside restore : 
all samples=1616, correct prediction=293.0
mini-batch loss=1.601121, test acc=0.181312
Precision 0.15746753246753248
Recall 0.15746753246753248
f1_score 0.15746753246753248
/home/mluser/user_achyuta/mc/venvQA3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
classification_report                precision    recall  f1-score   support

           0       0.05      0.35      0.08        34
           1       0.00      0.00      0.00       438
           2       0.23      0.59      0.34       144

   micro avg       0.16      0.16      0.16       616
   macro avg       0.09      0.31      0.14       616
weighted avg       0.06      0.16      0.08       616
## ConvLSTM_tl with ms_test Test
inside restore : 
all samples=716, correct prediction=514.0
mini-batch loss=0.701609, test acc=0.717877
Precision 0.7178770949720671
Recall 0.7178770949720671
f1_score 0.7178770949720671
/home/mluser/user_achyuta/mc/venvQA3.6/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
classification_report                precision    recall  f1-score   support

           0       0.52      0.60      0.56       154
           1       0.00      0.00      0.00        74
           2       0.78      0.86      0.82       488

   micro avg       0.72      0.72      0.72       716
   macro avg       0.43      0.49      0.46       716
weighted avg       0.65      0.72      0.68       716

