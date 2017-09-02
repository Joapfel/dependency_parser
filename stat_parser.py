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

    __slots__ = 'pos2id', 'form2id', 'action2id', 'id2action', 'logreg', 'enc', 'le', 'unknown'

    def __init__(self):
        self.pos2id = {} #features
        self.form2id = {} #features
        self.action2id = {} #labels encoded as numbers
        self.id2action = {} #for getting actual labels from encoded numbers
        self.logreg = linear_model.LogisticRegression()
        self.enc = OneHotEncoder()
        self.le = LabelEncoder()
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
                getattr(c, act)(arg)
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


        print(actions)
        tmp = []
        for t1, t2 in actions:
            tmp.append(str(t1) + " " + str(t2))

        actions = tmp
        print(actions)

        for l in actions: #fill the labels indecies
            if l not in self.action2id:
                self.action2id[l] = len(self.action2id)
        labels = [] #numerical representation of the labels
        for l in actions:
            labels.append(self.action2id[l])
        #reverse the action2id in order to get labels from number representation
        self.id2action = {v: k for k, v in self.action2id.items()}
        #create sparse matrix & vektor
        X = self.enc.fit_transform(feature_mtrx)
        y = self.le.fit_transform(labels)#labels
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
            #y_pred = self.logreg.predict_log_proba(X).ravel() #numerical
            #prob distr for predictions
            #distr_y = np.argsort(y_pred)
            #distr_y = distr_y[::-1]
            distr_y = self.logreg.decision_function(X).ravel()
            #print(distr_y)
            #print(np.argsort(distr_y))
            #print(self.logreg.classes_)
            distr_y = np.argsort(distr_y) #override with indexes
            distr_y = distr_y[::-1]
            for idx in distr_y:
                label = self.logreg.classes_[int(idx)]
                act, arg = label.split()
                #act, arg = self.id2action[int(idx)] #actual action
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
        for t_labels in j.deprel:
            labeled_true.append(t_labels)
        for p_labels in predicted_parse.deprel:
            labeled_pred.append(p_labels)

    unlabeled_accuracy = metrics.accuracy_score(unlabeled_true, unlabeled_pred)
    print(unlabeled_accuracy)

    labeled_accuracy = 0
    for i,j in zip(labeled_true, labeled_pred):
        if i == j:
            labeled_accuracy += 1
    labeled_accuracy /= len(labeled_true)
    print(labeled_accuracy)

    print('done')