import math, random, os
from collections import defaultdict, Counter

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    padded = start_pad(n) + text
    return [(padded[i:i + n], padded[i + n]) for i in range(len(text))]

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.counts = defaultdict(Counter)   # context -> Counter of next chars
        self.totals = defaultdict(int)       # context -> total count

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for context, char in ngrams(self.n, text):
            self.counts[context][char] += 1
            self.totals[context] += 1
            self.vocab.add(char)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        v = len(self.get_vocab())
        if context not in self.totals:
            return 1 / v if v else 0.0
        denom = self.totals[context] + self.k * v
        if denom == 0:
            return 1 / v
        return (self.counts[context][char] + self.k) / denom

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        vocab = sorted(self.get_vocab())
        cumulative = 0.0
        for c in vocab:
            cumulative += self.prob(context, c)
            if r < cumulative:
                return c
        return vocab[-1]  # guard against floating point round-off

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        out = []
        for _ in range(length):
            c = self.random_char(context)
            out.append(c)
            context = (context + c)[-self.n:] if self.n > 0 else ''
        return ''.join(out)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        grams = ngrams(self.n, text)
        if not grams:
            return float('inf')
        log_sum = 0.0
        for context, char in grams:
            p = self.prob(context, char)
            if p <= 0:
                return float('inf')
            log_sum += math.log(p)
        return math.exp(-log_sum / len(grams))

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super().__init__(n, k)
        # one add-k model for each order 0..n
        self.models = [NgramModel(i, k) for i in range(n + 1)]
        self.lambdas = [1 / (n + 1)] * (n + 1)   # equal weights by default

    def set_lambdas(self, lambdas):
        ''' Overwrites the default lambdas. lambdas[i] is the weight of the
            order-i model (i = 0..n). They are normalized to sum to 1. '''
        if len(lambdas) != self.n + 1:
            raise ValueError('need %d lambdas' % (self.n + 1))
        total = sum(lambdas)
        self.lambdas = [l / total for l in lambdas]

    def get_vocab(self):
        return self.models[0].get_vocab()

    def update(self, text):
        for m in self.models:
            m.update(text)

    def prob(self, context, char):
        p = 0.0
        for i, m in enumerate(self.models):
            sub = context[len(context) - i:] if i > 0 else ''
            p += self.lambdas[i] * m.prob(sub, char)
        return p

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################


TRAIN_DIR = 'train'
VAL_DIR = 'val'                    
TEST_FILE = 'cities_test.txt'
END = '$'                           

def read_lines(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]

def split_data():
    train = {cc: read_lines(os.path.join(TRAIN_DIR, cc + '.txt')) for cc in COUNTRY_CODES}
    dev = {cc: read_lines(os.path.join(VAL_DIR, cc + '.txt')) for cc in COUNTRY_CODES}
    return train, dev

def train_models(n, k, data):
    ''' Trains one interpolated model per country, with an end-of-text marker '''
    models = {}
    for cc in COUNTRY_CODES:
        m = NgramModelWithInterpolation(n, k)
        for city in data[cc]:
            m.update(city + END)
        models[cc] = m
    return models

def set_all_lambdas(models, lambdas):
    for m in models.values():
        m.lambdas = [1 / (m.n + 1)] * (m.n + 1)
        if lambdas:
            m.set_lambdas(lambdas)

def log_likelihood(model, text):
    total = 0.0
    for context, char in ngrams(model.n, text + END):
        total += math.log(max(model.prob(context, char), 1e-12))
    return total

def classify(models, city):
    return max(models, key=lambda cc: log_likelihood(models[cc], city))

def accuracy(models, dev):
    correct = total = 0
    for cc in COUNTRY_CODES:
        for city in dev[cc]:
            correct += classify(models, city) == cc
            total += 1
    return correct / total

def lambda_schemes(n):
    return {
        'equal': None,
        'high-order-heavy': [2 ** i for i in range(n + 1)],
        'no-unigram': [0] + [1] * n,
    }

if __name__ == '__main__':
    # part 1
    n = [2, 3, 4, 7]
    for ni in n:
        m = create_ngram_model(NgramModel, 'shakespeare_input.txt', ni)
        print(m.random_text(250))



    # part 2
    n = [2, 3, 4, 7]
    k = [0.1, 0.5, 1, 2]
    paths = ['shakespeare_sonnets.txt', 'nytimes_article.txt']
    train_path = 'shakespeare_input.txt'

    texts = {}
    for path in paths:
        with open(path, encoding='utf-8', errors='ignore') as f:
            texts[path] = ' '.join(f.read().split())

    def set_k(inter, kval):
        for m in inter.models:  
            m.k = kval

    result_fixk = [] 
    result_fixn = [] 
    for ni in n:
        inter = create_ngram_model(NgramModelWithInterpolation, train_path, ni, 1)  
        plain = inter.models[ni]                                                    
        for path in paths:
            set_k(inter, 1)
            result_fixk.append((path, ni, 1, plain.perplexity(texts[path]), inter.perplexity(texts[path])))
        if ni == 2:
            for ki in k:
                set_k(inter, ki)
                for path in paths:
                    result_fixn.append((path, 2, ki, plain.perplexity(texts[path]), inter.perplexity(texts[path])))

    for r in result_fixk: print(r)
    for r in result_fixn: print(r)

    lams = {
    'equal':            [.25, .25, .25, .25],
    'favor low order':  [.4, .3, .2, .1],
    'favor high order': [.1, .2, .3, .4],
    'no order 0':       [0, 1/3, 1/3, 1/3],
    }

    inter = create_ngram_model(NgramModelWithInterpolation, train_path, 3, 1)
    result_lambda = []
    for name, lam in lams.items():
        inter.set_lambdas(lam)
        for path in paths:
            result_lambda.append((path, name, lam, inter.perplexity(texts[path])))

    for path, name, lam, p in result_lambda:
        print('%-22s %-17s %-28s %.3f' % (path, name, lam, p))


    # part 3
    train, dev = split_data()

    best = (0, None)
    for n in range(1, 7):
        for k in [0.01, 0.05, 0.1, 0.5, 1]:
            models = train_models(n, k, train)
            for name, lam in lambda_schemes(n).items():
                set_all_lambdas(models, lam)
                acc = accuracy(models, dev)
                print('n=%d k=%-5s lambdas=%-17s dev acc=%.4f' % (n, k, name, acc), flush=True)
                if acc > best[0]:
                    best = (acc, (n, k, lam, name))

    acc, (n, k, lam, name) = best
    print('\nBest: n=%d k=%s lambdas=%s -> dev acc %.4f' % (n, k, name, acc))

<<<<<<< HEAD

=======
>>>>>>> 69e848a5c36e13cc411e3e58b15aa20c73646bec
    full = {cc: train[cc] + dev[cc] for cc in COUNTRY_CODES}
    models = train_models(n, k, full)
    set_all_lambdas(models, lam)

    with open('test_labels.txt', 'w') as out:
        for city in read_lines(TEST_FILE):
            out.write(classify(models, city) + '\n')
    print('Wrote test_labels.txt')
