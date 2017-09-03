from scipy.sparse.csr import csr_matrix
from sklearn.preprocessing.label import LabelEncoder
from collections import Counter
import conllu
from conllu import Sent
from transition import Oracle, Config
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
import numpy as np

class Parser(object):

    __slots__ = 'pos2id', 'form2id', 'logreg', 'enc', 'unknown'

    def __init__(self):
        self.pos2id = {} #features
        self.form2id = {} #features
        self.logreg = linear_model.LogisticRegression()
        self.enc = OneHotEncoder()
        self.unknown = "</u>"

    def feature(self, sentences): #extract the features for training
        form2id_c = Counter() #tmp counters for getting rid of low freq counts
        pos2id_c = Counter()
        for sent in sentences: #fill the id's
            for w in sent.form:
                if w not in form2id_c:
                    form2id_c[w] = len(form2id_c)+1 #zero is reserved for padding
            for t in sent.upostag:
                if t not in pos2id_c:
                    pos2id_c[t] = len(pos2id_c)+1
        #get rid of the low freq counts
        for w, c in form2id_c.items(): #w = word | c = count
            if c > 1 and w not in self.form2id:
                self.form2id[w] = c
        for t, c in pos2id_c.items(): #t = postag | c = count
            if c > 1 and t not in self.pos2id:
                self.pos2id[t] = c
        #add a feature for unknown
        self.form2id[self.unknown] = len(self.form2id)+1
        self.pos2id[self.unknown] = len(self.pos2id)+1

        feature_mtrx = [] # categorical numbers
        actions = [] # list of actions (labels)
        for sent in sentences:
            o = Oracle(sent)
            c = Config(sent)
            while not c.is_terminal():
                act, arg = o.predict(c)
                assert c.doable(act)
                actions.append((act, arg)) #fill list of actions
                features = [] #feature vektor of the form stack 3rd,2nd,1st --- 1st,2nd,3rd buffer
                for i in range(1,4): #add the top 3 words from the stack and the buffer
                    if i <= len(c.stack):
                        if sent.upostag[c.stack_nth(i)] in self.pos2id: #check for unknown
                            features.append(self.pos2id[sent.upostag[c.stack_nth(i)]])
                        else:
                            features.append(self.pos2id[self.unknown])
                        if sent.form[c.stack_nth(i)] in self.form2id:
                            features.append(self.form2id[sent.form[c.stack_nth(i)]])
                        else:
                            features.append(self.form2id[self.unknown])
                    else:
                        features.append(0)
                        features.append(0)

                    if i <= len(c.input):
                        if sent.upostag[c.input_nth(i)] in self.pos2id:
                            features.append(self.pos2id[sent.upostag[c.input_nth(i)]])
                        else:
                            features.append(self.pos2id[self.unknown])
                        if sent.form[c.input_nth(i)] in self.form2id:
                            features.append(self.form2id[sent.form[c.input_nth(i)]])
                        else:
                            features.append(self.form2id[self.unknown])
                    else:
                        features.append(0)
                        features.append(0)

                feature_mtrx.append(features)

                getattr(c, act)(arg)

        #tuples to string
        tmp = []
        for t1, t2 in actions:
            tmp.append(str(t1) + " " + str(t2))
        actions = tmp

        #create sparse matrix & vektor
        X = self.enc.fit_transform(feature_mtrx)
        return (X,actions)#y

    def train(self, sentences):
        X,y = self.feature(sentences)
        self.logreg.fit(X, y)

    def parse(self, sent): #predict
        c = Config(sent)
        while not c.is_terminal():
            features = []  # feature vektor of the form stack 3rd,2nd,1st --- 1st,2nd,3rd buffer
            for i in range(1, 4):  # add the top 3 words from the stack and the buffer
                if i <= len(c.stack):
                    if sent.upostag[c.stack_nth(i)] in self.pos2id:
                        features.append(self.pos2id[sent.upostag[c.stack_nth(i)]])
                    else:
                        features.append(self.pos2id[self.unknown])
                    if sent.form[c.stack_nth(i)] in self.form2id:
                        features.append(self.form2id[sent.form[c.stack_nth(i)]])
                    else:
                        features.append(self.form2id[self.unknown])
                else:
                    features.append(0)
                    features.append(0)

                if i <= len(c.input):
                    if sent.upostag[c.input_nth(i)] in self.pos2id:
                        features.append(self.pos2id[sent.upostag[c.input_nth(i)]])
                    else:
                        features.append(self.pos2id[self.unknown])
                    if sent.form[c.input_nth(i)] in self.form2id:
                        features.append(self.form2id[sent.form[c.input_nth(i)]])
                    else:
                        features.append(self.form2id[self.unknown])
                else:
                    features.append(0)
                    features.append(0)

            # create sparse matrix & vektor
            X = self.enc.transform([features])
            y_pred = self.logreg.predict_log_proba(X).ravel() #numerical
            #get the indecies from low to high values
            distr_y = np.argsort(y_pred)
            #from high to low
            distr_y = distr_y[::-1]

            for idx in distr_y:
                label = self.logreg.classes_[int(idx)]
                act, arg = label.split()
                if c.doable(act): #check if action is possible
                    getattr(c, act)(arg) #apply the action
                    break
        return c.finish()


if '__main__' == __name__:
    p = Parser()
    sentences = conllu.load('Universal Dependencies 2.0/ud-treebanks-v2.0/UD_English/SMALL.txt')#en-ud-train.conllu
    p.train(list(sentences))

    unlabeled_true = []
    unlabeled_pred = []
    labeled_true = []
    labeled_pred = []
    s2 = conllu.load('Universal Dependencies 2.0/ud-treebanks-v2.0/UD_English/en-ud-dev.conllu')
    s2 = list(s2)
    for i,j in enumerate(s2):
        predicted_parse = p.parse(j)
        conllu.save([predicted_parse], 'RESULTS/result'+str(i))
        for t_head in j.head:
            unlabeled_true.append(t_head)
        for p_head in predicted_parse.head:
            unlabeled_pred.append(p_head)
        for t_head,t_labels in zip(j.deprel, j.head):
            labeled_true.append((t_head, t_labels))
        for p_head,p_labels in zip(predicted_parse.deprel, predicted_parse.head):
            labeled_pred.append((p_head,p_labels))

    unlabeled_accuracy = metrics.accuracy_score(unlabeled_true, unlabeled_pred)
    print(unlabeled_accuracy)

    labeled_accuracy = 0 #metrics.accuracy_score(labeled_true, labeled_pred)
    for i,j in zip(labeled_true, labeled_pred):
        if i == j:
            labeled_accuracy += 1
    labeled_accuracy /= len(labeled_true)
    print(labeled_accuracy)

    print('done')