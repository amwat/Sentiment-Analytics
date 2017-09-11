import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random

random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

# nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])


def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos,
                                                                                    test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos,
                                                                                    test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir + "train-pos.txt", "r") as f:
        for i, line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_pos.append(words)
    with open(path_to_dir + "train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_neg.append(words)
    with open(path_to_dir + "test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_pos.append(words)
    with open(path_to_dir + "test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def construct_binary_vectors(data, features):
    binary_vectors = []
    for item in data:
        words = set(item)
        binary_vector = []
        for feature in features:
            if feature in words:
                binary_vector.append(1)
            else:
                binary_vector.append(0)
        binary_vectors.append(binary_vector)
    return binary_vectors


def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many positive texts as negative texts, or vice-versa.

    positive_occurrences = {}
    negative_occurrences = {}

    for post in train_pos:
        for word in set(post):
            if word not in stopwords:
                if word in positive_occurrences:
                    positive_occurrences[word] += 1
                else:
                    positive_occurrences[word] = 1

    for post in train_neg:
        for word in set(post):
            if word not in stopwords:
                if word in negative_occurrences:
                    negative_occurrences[word] += 1
                else:
                    negative_occurrences[word] = 1

    features = []
    positive_threshold = 0.01 * len(train_pos)
    negative_threshold = 0.01 * len(train_neg)

    for key, value in positive_occurrences.iteritems():
        if value >= positive_threshold and value >= 2 * negative_occurrences[key]:
            features.append(key)

    for key, value in negative_occurrences.iteritems():
        if value >= negative_threshold and value >= 2 * positive_occurrences[key]:
            features.append(key)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.

    train_pos_vec = construct_binary_vectors(train_pos, features)
    train_neg_vec = construct_binary_vectors(train_neg, features)
    test_pos_vec = construct_binary_vectors(test_pos, features)
    test_neg_vec = construct_binary_vectors(test_neg, features)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def create_labeled_sentences(dataset, label):
    labeled_sentences = []
    i = 0
    for data in dataset:
        labeled_sentences.append(LabeledSentence(data,[label + "_" + str(i)]))
        i += 1
    return labeled_sentences


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.

    labeled_train_pos = create_labeled_sentences(train_pos, "TRAIN_POS")
    labeled_train_neg = create_labeled_sentences(train_neg, "TRAIN_NEG")
    labeled_test_pos = create_labeled_sentences(test_pos, "TEST_POS")
    labeled_test_neg = create_labeled_sentences(test_neg, "TEST_NEG")

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    train_pos_vec = list(model.docvecs[["TRAIN_POS_" + str(i) for i in range(len(train_pos))]])
    train_neg_vec = list(model.docvecs[["TRAIN_NEG_" + str(i) for i in range(len(train_neg))]])
    test_pos_vec = list(model.docvecs[["TEST_POS_" + str(i) for i in range(len(test_pos))]])
    test_neg_vec = list(model.docvecs[["TEST_NEG_" + str(i) for i in range(len(test_neg))]])

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LogisticRegression Model that are fit to the training data.
    """
    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(train_pos_vec + train_neg_vec, Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(train_pos_vec + train_neg_vec, Y)
    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(train_pos_vec + train_neg_vec, Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(train_pos_vec + train_neg_vec, Y)
    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    positive_predictions = model.predict(test_pos_vec)
    negative_predictions = model.predict(test_neg_vec)

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for prediction in positive_predictions:
        if prediction == "pos":
            tp += 1
        else:
            fn += 1

    for prediction in negative_predictions:
        if prediction == "neg":
            tn += 1
        else:
            fp += 1

    accuracy = float(tp + tn) / (tp + tn + fp + fn)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)


if __name__ == "__main__":
    main()
