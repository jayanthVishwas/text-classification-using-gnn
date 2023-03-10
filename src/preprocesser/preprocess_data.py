from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

import sys
# sys.path.append('../')
from src.utils.utils import clean_str, loadWord2Vec  

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("wrong number of arguments")
        sys.exit(1)

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    dataset = sys.argv[1]

    if dataset not in datasets:
        sys.exit("wrong dataset name")
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    doc_content_list = []
    
    with open('data/corpus/' + dataset + '.txt', 'rb') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip().decode('latin1'))
    
    word_freq = {}

    for doc in doc_content_list:
        word_clean = clean_str(doc).split()

        for word in word_clean:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    clean_doc_content_list = []

    for doc in doc_content_list:
        word_clean = clean_str(doc).split()

        doc_words = []
        for word in word_clean:
            if dataset=='mr' :
                doc_words.append(word)
            
            elif word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)
        
        doc_str = ' '.join(doc_words).strip()
        clean_doc_content_list.append(doc_str)

    clean_corpus = '\n'.join(clean_doc_content_list)

    with open('data/corpus/' + dataset + '.clean.txt', 'w') as f:
        f.write(clean_corpus)
    
    min_len = 10000
    aver_len = 0
    max_len = 0 

    with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            temp = line.split()
            aver_len = aver_len + len(temp)
            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)

    aver_len = 1.0 * aver_len / len(lines)
    print('Min_len : ' + str(min_len))
    print('Max_len : ' + str(max_len))
    print('Average_len : ' + str(aver_len))


    

    



