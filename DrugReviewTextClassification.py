# COMP723: Data Mining and Knowledge Engineering
# Assignment 1: Text classification
# Name:          Megan Teh
# Student ID:    13835048

"""
This assignment implements text classification on online drug reviews to
predict the drug effectiveness. This is the main script to execute the text classification.
Two classifiers are built:
1. Random Forest algorithm using a Bag of Words model
2. Support Vector Machine algorithm using Word Embeddings

Model performance is evaluated using FPR metrics.
Both models are run twice: with pre-processed data and without pre-processing, to show
the impacts of pre-processing on model performance
"""

import gensim
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from DataMining_Assignment1_TextClassification import utils
from sklearn.svm import SVC

test_filepath = "data/drugLibTest_raw.tsv"
train_filepath = "data/drugLibTrain_raw.tsv"

print("Processing the training dataset")
train_df = utils.build_dataframe_from_filepath(raw_file_location=train_filepath)
print("Processing the test dataset")
test_df = utils.build_dataframe_from_filepath(raw_file_location=test_filepath)
print("Building the class labels")
classifications = utils.build_classifications_dict(df=train_df)

# Building the lexicon for the bag of words model (processed and unprocessed lexcions built)
lexicon_pkl_file = "data/lexicon_preprocessed.pkl"
lexicon = utils.build_processed_lexicon(df=train_df, pkl_file=lexicon_pkl_file,
                                        lexicon_column='bow_input_combined_review',
                                        lexicon_size=800, ngram_min=2, ngram_max=3, min_term_freq=5)
# Load an unprocessed lexicon, built choosing randomly selected words across the train dataset
lexicon_unprocessed = utils.random_unprocessed_lexicon()
print("Pre-processed lexicon")
print(lexicon)
print("Unprocessed lexicon")
print(lexicon_unprocessed)

train_xy_processed_pkl_file = "data/train_xy_preprocessed.pkl"
test_xy_processed_pkl_file = "data/test_xy_preprocessed.pkl"
train_xy_unprocessed_pkl_file = "data/train_xy_unpreprocessed.pkl"
test_xy_unprocessed_pkl_file = "data/test_xy_unpreprocessed.pkl"

train_xy_preprocessed = utils.build_xy_features(df=train_df, pkl_file=train_xy_processed_pkl_file,
                                                x_column="bow_input_combined_review", y_column="effectiveness",
                                                lexicon_list=lexicon, classifications_dict=classifications)

test_xy_preprocessed = utils.build_xy_features(df=test_df, pkl_file=test_xy_processed_pkl_file,
                                               x_column="bow_input_combined_review", y_column="effectiveness",
                                               lexicon_list=lexicon, classifications_dict=classifications)

train_xy_unprocessed = utils.build_xy_features(df=train_df, pkl_file=train_xy_unprocessed_pkl_file,
                                               x_column="bow_input_combined_review", y_column="effectiveness",
                                               lexicon_list=lexicon_unprocessed, classifications_dict=classifications)

test_xy_unprocessed = utils.build_xy_features(df=test_df, pkl_file=test_xy_unprocessed_pkl_file,
                                              x_column="bow_input_combined_review", y_column="effectiveness",
                                              lexicon_list=lexicon_unprocessed, classifications_dict=classifications)

X_train_bow_preprocessed, X_test_bow_preprocessed, X_train_bow_unprocessed, X_test_bow_unprocessed, \
y_train, y_test = [], [], [], [], [], []

for data in train_xy_preprocessed:
    X_train_bow_preprocessed.append(data[0])
    y_train.append(data[1])

for data in test_xy_preprocessed:
    X_test_bow_preprocessed.append(data[0])
    y_test.append(data[1])

for data in train_xy_unprocessed:
    X_train_bow_unprocessed.append(data[0])

for data in test_xy_unprocessed:
    X_test_bow_unprocessed.append(data[0])

print("y test and train sets created")

# Create a Random Forest model
print("Creating the Random Forest model (preprocessed run)")
random_forest_model_preprocessed = RandomForestClassifier(n_estimators=100, random_state=1, verbose=1,
                                                          class_weight='balanced')
print("Using the model to perform text cl")
random_forest_model_preprocessed.fit(X_train_bow_preprocessed, y_train)
y_pred_rf_preprocessed = random_forest_model_preprocessed.predict(X_test_bow_preprocessed)

# Random Forest model performance metrics
print("\nRandom Forest Accuracy (with pre processing):", metrics.accuracy_score(y_test, y_pred_rf_preprocessed))

print("Random forest confusion matrix (with pre processing): \n")
print(metrics.confusion_matrix(y_test, y_pred_rf_preprocessed))

print("\nRandom forest precision score (with pre processing):")
print(precision_score(y_test, y_pred_rf_preprocessed, average='weighted'))
print("\nRandom forest recall score (with pre processing):")
print(recall_score(y_test, y_pred_rf_preprocessed, average='weighted'))
print("\nRandom forest f1 score (with pre processing):")
print(f1_score(y_test, y_pred_rf_preprocessed, average='weighted'))

print("Random Forest classification report (with pre processing):")
print(classification_report(y_test, y_pred_rf_preprocessed))

print("Classifications key:")
for key, value in classifications.items():
    print(key, value)

# Create a Random Forest model without any preprocessing
print("Creating the Random Forest model (no pre processing run)")
random_forest_model_unprocessed = RandomForestClassifier(n_estimators=100, random_state=1, verbose=1,
                                                         class_weight='balanced')
print("Using the model to perform text cl")
random_forest_model_unprocessed.fit(X_train_bow_unprocessed, y_train)
y_pred_rf_unprocessed = random_forest_model_unprocessed.predict(X_test_bow_unprocessed)

# Random Forest model performance metrics
print("\nRandom Forest Accuracy (without preprocessing):", metrics.accuracy_score(y_test, y_pred_rf_unprocessed))

print("Random forest confusion matrix (without preprocessing): \n")
print(metrics.confusion_matrix(y_test, y_pred_rf_unprocessed))

print("\nRandom forest precision score (without preprocessing):")
print(precision_score(y_test, y_pred_rf_unprocessed, average='weighted'))
print("\nRandom forest recall score (without preprocessing):")
print(recall_score(y_test, y_pred_rf_unprocessed, average='weighted'))
print("\nRandom forest f1 score (without preprocessing):")
print(f1_score(y_test, y_pred_rf_unprocessed, average='weighted'))

print("Random Forest classification report (without preprocessing):")
print(classification_report(y_test, y_pred_rf_unprocessed))

print("Classifications key:")
for key, value in classifications.items():
    print(key, value)

# Using the SVM classifier (preprocessed)

train_corpus_preprocessed_pkl, test_corpus_preprocessed_pkl = "train_corpus_svm_preprocessed.pkl", "test_corpus_svm_preprocessed.pkl"

print("Creating the train corpus for SVM preprocessed")
train_corpus_svm_preprocessed = list(
    utils.read_corpus(df=train_df, classifications_dict=classifications, corpus_column="doc2vec_input_combined_review"))
print("Creating the test corpus for SVM preprocessed")
test_corpus_svm_preprocessed = list(
    utils.read_corpus(df=test_df, classifications_dict=classifications, corpus_column="doc2vec_input_combined_review",
                      tokens_only=True))

print("Building and training the document embeddings model for SVM preprocessed")
doc2vec_model_unprocessed = gensim.models.doc2vec.Doc2Vec(vector_size=1000, window=10, min_count=5, workers=4,
                                                          epochs=20)
print("Building the vocabulary for SVM preprocessed")
doc2vec_model_unprocessed.build_vocab(train_corpus_svm_preprocessed)
print("Training the model for SVM preprocessed")
doc2vec_model_unprocessed.train(train_corpus_svm_preprocessed, total_examples=doc2vec_model_unprocessed.corpus_count,
                                epochs=doc2vec_model_unprocessed.epochs)

# Output the word embeddings to a csv file
print("Creating document embeddings from the training dataset for SVM preprocessed")
doc2vec_file_with_preprocessing = r'data/doc2vec_preprocessed.csv'
utils.create_word_embeddings_csv(df=train_df, model=doc2vec_model_unprocessed,
                                 output_csv=doc2vec_file_with_preprocessing)
doc2vec_df_preprocessed = pd.read_csv(doc2vec_file_with_preprocessing)

print("Creating SVM model (pre processed data)")
svm_model_preprocessed = SVC(gamma='scale', class_weight='balanced', verbose=True)
print("Fitting SVM model to document embeddings for SVM preprocessed")
svm_model_preprocessed.fit(doc2vec_df_preprocessed, y_train)
print("Using the SVM model to predict text classifications for SVM preprocessed")

test_features_doc2vec_processed = []
for index, row in test_df.iterrows():
    model_vector = doc2vec_model_unprocessed.infer_vector(row['doc2vec_input_combined_review'].split())
    test_features_doc2vec_processed.append(model_vector)

y_pred_svm_preprocessed = svm_model_preprocessed.predict(test_features_doc2vec_processed)

# SVM model performance metrics
print("\nSVM Accuracy (pre-processed):", metrics.accuracy_score(y_test, y_pred_svm_preprocessed))

print("SVM confusion matrix (pre-processed): \n")
print(metrics.confusion_matrix(y_test, y_pred_svm_preprocessed))

print("\nSVM precision score (pre-processed):")
print(precision_score(y_test, y_pred_svm_preprocessed, average='weighted'))
print("\nSVM recall score (pre-processed):")
print(recall_score(y_test, y_pred_svm_preprocessed, average='weighted'))
print("\nSVM f1 score (pre-processed):")
print(f1_score(y_test, y_pred_svm_preprocessed, average='weighted'))

print("SVM classification report (pre-processed):")
print(classification_report(y_test, y_pred_svm_preprocessed))

print("Classifications key:")
for key, value in classifications.items():
    print(key, value)

# Using the SVM classifier (without preprocessing)
print("Creating the train corpus for SVM without preprocessing")
train_corpus_unprocessed_pkl, test_corpus_unprocessed_pkl = "train_corpus_svm_unprocessed.pkl", "test_corpus_svm_unprocessed.pkl"
train_corpus_svm_unprocessed = list(
    utils.read_corpus(df=train_df, classifications_dict=classifications, corpus_column="raw_total_row"))
print("Creating the test corpus for SVM without preprocessing")
test_corpus_svm_unprocessed = list(
    utils.read_corpus(df=test_df, classifications_dict=classifications, corpus_column="raw_total_row",
                      tokens_only=True))

print("Building and training the document embeddings model for SVM without preprocessing")
doc2vec_model_unprocessed = gensim.models.doc2vec.Doc2Vec(vector_size=1000, window=10, min_count=5, workers=4,
                                                          epochs=20)
print("Building the vocabulary for SVM without preprocessing")
doc2vec_model_unprocessed.build_vocab(train_corpus_svm_unprocessed)
print("Training the model for SVM without preprocessing")
doc2vec_model_unprocessed.train(train_corpus_svm_unprocessed, total_examples=doc2vec_model_unprocessed.corpus_count,
                                epochs=doc2vec_model_unprocessed.epochs)

# Output the word embeddings to a csv file
print("Creating document embeddings from the training dataset for SVM without preprocessing")
doc2vec_file_no_preprocessing = r'data/doc2vec_no_preprocessing.csv'
utils.create_word_embeddings_csv(df=train_df, model=doc2vec_model_unprocessed, output_csv=doc2vec_file_no_preprocessing)
doc2vec_df = pd.read_csv(doc2vec_file_no_preprocessing)

print("Creating SVM model (unprocessed data)")
svm_model_unprocessed = SVC(gamma='scale', class_weight='balanced', verbose=True)
print("Fitting SVM model to document embeddings for SVM without preprocessing")
svm_model_unprocessed.fit(doc2vec_df, y_train)
print("Using the SVM model to predict text classifications without preprocessing")

test_features_doc2vec_unprocessed = []
for index, row in test_df.iterrows():
    model_vector = doc2vec_model_unprocessed.infer_vector(row['doc2vec_input_combined_review'].split())
    test_features_doc2vec_unprocessed.append(model_vector)
y_pred_svm_unprocessed = svm_model_unprocessed.predict(test_features_doc2vec_unprocessed)

# SVM model performance metrics
print("\nSVM Accuracy (no preprocessing):", metrics.accuracy_score(y_test, y_pred_svm_unprocessed))

print("SVM confusion matrix (no preprocessing): \n")
print(metrics.confusion_matrix(y_test, y_pred_svm_unprocessed))

print("\nSVM precision score (no preprocessing):")
print(precision_score(y_test, y_pred_svm_unprocessed, average='weighted'))
print("\nSVM recall score (no preprocessing):")
print(recall_score(y_test, y_pred_svm_unprocessed, average='weighted'))
print("\nSVM f1 score (no preprocessing):")
print(f1_score(y_test, y_pred_svm_unprocessed, average='weighted'))

print("SVM classification report (no preprocessing):")
print(classification_report(y_test, y_pred_svm_unprocessed))

print("Classifications key:")
for key, value in classifications.items():
    print(key, value)

print('End of text classification')
