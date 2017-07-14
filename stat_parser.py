from conllu import Sent
from transition import Oracle, Config


class Parser(object):


    def read_treebank(self, path):
        if not isinstance(path, str):
            raise TypeError('Expected argument for the path is a string!')
        sentences = []
        with open(path, 'r') as f:
            sent = []
            for line in f: #each line represents a word
                if not line.strip() and len(sent) > 0: #if the line is empty -> add the previous sentence to the list
                    sentences.append(sent)
                    sent = []
                if line[0].isdigit():
                    sent.append(line)
        return sentences


    def feature(self, treebank_sentences):
        if not isinstance(treebank_sentences, list):
            raise TypeError('Expected argument for the treebank sentences is a list!')
        feature_mtrx = []
        actions = [] # list of actions
        for sent in treebank_sentences:
            s = Sent(sent)
            o = Oracle(s)
            c = Config(s)
            while not c.is_terminal():
                act, arg = o.predict(c)
                assert c.doable(act)
                actions.append((act, arg)) #fill list of actions
            features = [] #feature vektor of the form stack 3rd,2nd,1st --- 1st,2nd,3rd buffer
            for i in reversed(range(3)): #add the top 3 words from the stack
                features.append(c.stack_nth(i))
            for i in reversed(range(3)): #add the top 3 words from the buffer
                features.append(c.input_nth(i))
            #TODO: transform feature vektor to numbers
            #can I use the same procedure as in Tokenizer? --> concat onehots --> than sparse?
            #do I need to take care of sentence boundaries --> padding?





if '__main__' == __name__:
    p = Parser()
    sentences = p.read_treebank('Universal Dependencies 2.0/ud-treebanks-v2.0/UD_English/en-ud-train.conllu')
    for s in sentences:
        print(s)
    #p.feature(sentences)