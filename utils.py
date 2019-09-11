import numpy as np
import os

printmode=False

def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = np.arange(length)
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]

def load_w2v(w2v_file, embedding_dim, is_skip=False):
    '''
        Loads and returns a word2id dictionary and a word embedding
    
        Args:
            w2v_file: w2v filename
            embedding_dim: dimention of word vectors
            is_skip: is there a headline to skip
    
        Returns:
            word_dict: a dictionary whose key is a word and value is the id of the word
            w2v: a list of vectors for each word
    '''
    fp = open(w2v_file,encoding='utf8')
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    cnt = -1
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {} at {}'.format(line[0],cnt))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    print('oringin length of word_dict:', len(word_dict), ',length of w2v', len(w2v))
    if (len(word_dict)!=len(w2v)):
        raise Exception('w2v error')
    return word_dict, w2v

def load_tag_embedding(tag_file, word_dict, w2v, embedding_dim):
    '''
        entity aspect embeddings are initialized here.

        Args:
            tag_file: entity file or aspect file
            word_dict: a dictionary whose key is a word and value is the id of the word
            w2v: a list of vectors for each word
            embedding_dim: dimention of word vectors

        Returns:
            word_dict: a dictionary whose key is a word and value is the id of the word
            w2v: a list of vectors for each word
    '''
    for line in open(tag_file, encoding='utf8'):
        line = line.lower().rstrip('\n')
        if line in word_dict or line.replace(' ','') in word_dict:
            continue
        else:
            info = line.split()
            tmp = []
            for word in info:
                if word in word_dict:
                    tmp.append(w2v[word_dict[word]])
                else:
                    if printmode:
                        print('$WARNING LEVEL-1$: {} not in dic'.format(word))
            if tmp:
                word_dict[line.replace(' ','')]=len(word_dict)+1
                w2v.append(np.sum(tmp, axis=0) / len(tmp))
            else:
                word_dict[line] = len(word_dict) + 1
                w2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))
                if printmode:
                    print('$WARNING LEVEL-2$: {} not in dic'.format(line))
    return word_dict, w2v

def load_word_embedding(w2v_file, entity_id_file, embedding_dim, is_skip=False):
    '''
        Loads and returns a word2id dictionary and a word embedding;
        entity embeddings are initialized here.

        Args:
            w2v_file: w2v filename
            entity_id_file: entity list
            embedding_dim: dimention of word vectors
            is_skip: is there a headline to skip

        Returns:
            word_dict: a dictionary whose key is a word and value is the id of the word
            w2v: a list of vectors for each word
    '''

    word_dict, w2v = load_w2v(w2v_file, embedding_dim, is_skip)
    word_dict, w2v = load_tag_embedding(entity_id_file, word_dict, w2v, embedding_dim)

    w2v = np.asarray(w2v, dtype=np.float32)
    print('Shape of PreEmbedding is',w2v.shape)
    print('modified length of word_dict:', len(word_dict), ',length of w2v', len(w2v))
    return word_dict, w2v

def change_y_to_onehot(y):
    from collections import Counter
    #print(Counter(y))
    y_onehot_mapping = {}
    y_onehot_mapping['Negative'] = 0
    y_onehot_mapping['Neutral'] = 1
    y_onehot_mapping['Positive'] = 2
    n_class = 3

    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)

    return np.asarray(onehot, dtype=np.int32)

def get_max_sentence_len(input_file):
    '''
    get max sentence length of input_file.
    :param input_file: file path
    :return: maxlen: max sentence length
              posset: POS tag set
    '''
    maxlen=0
    posset = set()
    lines = open(input_file, encoding='utf8').readlines()
    for i in range(0, len(lines)):
        try:
            text, entity, senti = lines[i].rstrip().split('\t')
            if senti!='Neutral' and senti!='Positive' and senti!='Negative':
                continue
            length=len(text.strip('"').split("###"))
            if length>maxlen:
                maxlen=length
            for x in text.strip('"').split("###"):
                xsp = x.split('|')
                if len(xsp)!=4:
                    #print(lines[i],xsp)
                    #print(text)
                    raise Exception('None POS error')
                posset.add(xsp[1])
        except Exception as e:
            print("get_max_sent_len error : ",e)
    return maxlen, posset

def load_input_data_at(data, word_to_id, max_sentence_len, p2id, encoding='utf8', case = 'train'):

    x, y, sen_len = [], [], []
    entity_words = []
    aspect_words = []
    poslist = []
    idlist = []
    location_entity = []
    location_aspect = []
    
    text, entity = data['text'], data['entity']
    words = text.lower().strip('"').split("###")
    if len(words)>max_sentence_len:
        words = words[0:max_sentence_len]

    ids = []
    poss = []
    eloc = []
    for wordpos in words:
        word, pos, word_indx, disentity = wordpos.split('|')
        if word in word_to_id:
            ids.append(word_to_id[word])
            poss.append(1)
            eloc.append(int(disentity))

    eloc = [[1 - x / len(words)] for x in eloc]

    sen_len.append(len(ids))
    assert len(ids)<=max_sentence_len
    eloc += [[0]] * (max_sentence_len - len(ids))
    location_entity.append(eloc)
    entity_words.append(word_to_id.get(entity, 0))

    x.append(ids + [0] * (max_sentence_len - len(ids)))
    poslist.append(poss + [0] * (max_sentence_len - len(ids)))
    
    for item in x:
        if len(item) != max_sentence_len:
            print('$WARNING LEVEL-3$ ！', len(item))
    x = np.asarray(x, dtype=np.int32)
    poslist = np.asarray(poslist, dtype=np.int32)
    location_entity = np.asarray(location_entity,dtype=np.float32)
    return x, np.asarray(sen_len), np.asarray(entity_words), \
           np.asarray([[0,0,0]], dtype=np.int32), poslist, location_entity

def load_inputs_data_at(dataset, word_to_id, max_sentence_len, p2id, encoding='utf8', case = 'train'):

    x, y, sen_len = [], [], []
    entity_words = []
    aspect_words = []
    poslist = []
    idlist = []
    location_entity = []
    location_aspect = []

    lines = open(dataset,encoding=encoding).readlines()

    for i in range(0, len(lines)):
        try:
            text, entity, senti = lines[i].rstrip().split('\t')
            words = text.lower().strip('"').split("###")
            if len(words)>max_sentence_len:
                if case == 'train':
                    continue
                else:
                    words = words[0:max_sentence_len]

            if senti!='Neutral' and senti!='Positive' and senti!='Negative':
                continue

            ids = []
            poss = []
            eloc = []
            for wordpos in words:
                word, pos, word_indx, disentity = wordpos.split('|')
                if word in word_to_id:
                    ids.append(word_to_id[word])
                    poss.append(1)
                    eloc.append(int(disentity))

            eloc = [[1 - x / len(words)] for x in eloc]

            sen_len.append(len(ids))
            assert len(ids)<=max_sentence_len
            eloc += [[0]] * (max_sentence_len - len(ids))
            location_entity.append(eloc)
            entity_words.append(word_to_id.get(entity, 0))

            y.append(senti)
            idlist.append(text+'\t'+entity+'\t'+senti)

            x.append(ids + [0] * (max_sentence_len - len(ids)))
            poslist.append(poss + [0] * (max_sentence_len - len(ids)))
        except Exception as e:
            print("load_inputs_data_at error : ",e)
    
    y_onehot = change_y_to_onehot(y)
    for item in x:
        if len(item) != max_sentence_len:
            print('$WARNING LEVEL-3$ ！', len(item))
    x = np.asarray(x, dtype=np.int32)
    poslist = np.asarray(poslist, dtype=np.int32)
    location_entity = np.asarray(location_entity,dtype=np.float32)
    return x, np.asarray(sen_len), np.asarray(entity_words), \
           np.asarray(y_onehot), np.asarray(y), idlist, poslist, location_entity

def check_file_exist(files):
    for file in files:
        if not os.path.exists(file):
            print(file, 'not exists')
            return False
    return True

def pos2vec(posset):
    '''
    encode POSs to one-hot vectors
    :param posset: a set of POSs
    :return: dictionary, keys are POSs, values are one-hot vectors for POSs
    '''
    count=0
    p2id = dict()
    p2v = []
    for k in posset:
        vec = [0]*len(posset)
        vec[count] = 1
        count += 1
        p2id[k] = count
        p2v.append(vec)
    return p2id, p2v

def load_data_init(datset_file, test_file, w2v_file, entity_id_file, embedding_dim):
    if not check_file_exist([datset_file, test_file, w2v_file, entity_id_file]):
        raise Exception('file not exist error')
    maxlen,posset = get_max_sentence_len(datset_file)
    maxlen2,posset2 = get_max_sentence_len(test_file)
    maxlen2 = 45
    maxlen = 45
    print('sentence maxlen train/test:',maxlen,maxlen2)
    if maxlen2 > maxlen:
        maxlen = maxlen2
    # we set maxlen to 280 because there is only a tiny number of samples having more than 280 tokens
    if maxlen>280:
        maxlen = 280
    posset = posset|posset2
    posset = {'ADV', 'PROPN', 'DET', 'SPACE', 'NUM', 'PART', 'VERB', 'AUX', 'PUNCT', 'SYM', 'INTJ', 'ADP', 'CCONJ', 'X', 'NOUN', 'ADJ', 'PRON'}
    posset = posset|posset2
    p2id, p2v = pos2vec(posset)
    w2id, w2v = load_word_embedding(w2v_file, entity_id_file, embedding_dim)

    print('type',type(w2id),type(p2id))
    return w2id, w2v, p2id, p2v, maxlen

