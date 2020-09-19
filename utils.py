# COMP723: Data Mining and Knowledge Engineering
# Assignment 1: Text classification
# Name:          Megan Teh
# Student ID:    13835048

"""
This assignment implements text classification on online drug reviews to
predict the drug effectiveness. This is a utilities script with helper functions
used for pre-processing data to build the text classification models.
"""

import pickle
import spacy
import pandas
import numpy
from os.path import exists
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
    strip_numeric, strip_short
from gensim.models.doc2vec import TaggedDocument
from num2words import num2words


# Conducts the following pre-processing steps on an input string: removes any tags e.g. <b>, removes punctuation, multiple
# white spaces/line breaks, numeric characters, and removes all words that are 2 characters long.
# Returns the cleaned text as a list of tokens
def clean_text(text):
    custom_filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric,
                      strip_short]
    cleaned_text = preprocess_string(text, filters=custom_filters)
    return cleaned_text


# Loads a Pandas Dataframe from the filepath with the raw data. Load this dataframe from an existing pickle file if it
# exists
def build_dataframe_from_filepath(raw_file_location):
    pkl_processed_file = raw_file_location[:-8] + "_processed.pkl"

    # Load the pickle file if this data has already been processed in the past and saved
    if exists(pkl_processed_file):
        print("Loading the pre-processed dataset from a pickle file. Filename: " + pkl_processed_file)
        return pandas.read_pickle(pkl_processed_file)

    # Load raw dataframe from pkl file, if it exists, otherwise create a pkl file for the raw data
    pickle_filepath = raw_file_location[:-3] + "pkl"
    if exists(pickle_filepath):
        print("Loading the raw dataset from a pickle file for processing. Filename: " + pickle_filepath)
        df = pandas.read_pickle(pickle_filepath)
    else:
        print("Creating a new dataframe from the raw data")
        df = pandas.read_csv(raw_file_location, sep='\t')
        df.drop_duplicates()
        df.to_pickle(pickle_filepath)

    # Create new columns to store the individual cleaned reviews (not combined)
    df["normalised_benefits_review"] = numpy.nan
    df["normalised_side_effects_review"] = numpy.nan
    df["normalised_comments_review"] = numpy.nan
    # New column that combines all 3 cleaned reviews into one combined clean review
    df["normalised_combined_review"] = numpy.nan
    df["rating_to_word"] = numpy.nan
    df['raw_total_row'] = numpy.nan

    # Combining the review data from the 3 columns: benefitsReview, sideEffectsReview and commentsReview
    df["bow_input_combined_review"] = numpy.nan

    # Final review data format to input into the doc2vec model used to generate word embeddings
    df["doc2vec_input_combined_review"] = numpy.nan

    df = df.astype('object')
    print("Normalising the 3 review columns data and converting the ratings from numbers to words "
          "and combining all unprocessed rows into one column")
    for index, row in df.iterrows():
        # All raw, unprocessed review columns concatenated as a string to test model performance on unprocessed data
        df.loc[index, 'raw_total_row'] = ' '.join(list(map(str, row.tolist())))

        # Clean and tokenize the review data columns
        if not pandas.isnull(row['benefitsReview']):
            df.loc[index, "normalised_benefits_review"] = clean_text(row['benefitsReview'])
        else:
            df.loc[index, "normalised_benefits_review"] = []

        if not pandas.isnull(row['sideEffectsReview']):
            df.loc[index, "normalised_side_effects_review"] = clean_text(row['sideEffectsReview'])
        else:
            df.loc[index, "normalised_side_effects_review"] = []

        if not pandas.isnull(row['commentsReview']):
            df.loc[index, "normalised_comments_review"] = clean_text(row['commentsReview'])
        else:
            df.loc[index, "normalised_comments_review"] = []

        # Convert the rating column from numbers to words
        df.loc[index, "rating_to_word"] = num2words(row["rating"])

    print("Combining the 3 normalised review columns data into one new column")
    df["normalised_combined_review"] = df['normalised_benefits_review'] + df['normalised_side_effects_review'] \
                                   + df['normalised_comments_review']

    print("Building the custom stopwords list")
    # Further processing steps of lemmatisation, stop word removal, stemming & feature engineering
    # df["normalised_combined_review_str"] = df["normalised_combined_review"]
    nltk_stopwords = stopwords.words('english')
    exclusions = ['hadn', 'weren', "shan't", 'needn', 'couldn', "mustn't", 'hasn', 'won', "weren't", "hasn't", 'aren',
                  'down', "needn't", 'only', "all", 'didn', "those",'no', 'so', "don't", "shouldn't", "isn't",
                  "doesn't", "wasn't", 'not', 'most', 'shouldn', "mightn't", 'few', 'doesn', 'mightn', "didn't", "any",
                  'haven', "nor", 'more', 'mustn', 'other', 'some', "all", "won't", "haven't", 'isn', 'same', 'wouldn',
                  "wouldn't", "aren't", "couldn't"]

    customised_stopwords = [stop_word for stop_word in nltk_stopwords if stop_word not in exclusions]

    wl = WordNetLemmatizer()
    print("POS Tagging, lemmatization and running NER tool")
    # Creating doc2vec (embeddings) model input review
    nlp = spacy.load("en_core_web_sm")
    # Additional custom exclusions which can be considered as corpus/domain-specific stop words
    custom_exclusions = ['take', 'medication', 'medicine', 'pill', 'drug', 'start', 'tablet', 'get']
    for index, row in df.iterrows():
        review_tokens = row["normalised_combined_review"]
        all_postags = pos_tag(review_tokens)
        postags = []
        # Only include POS types: nouns, verbs, adverbs and determiners (some determiners contain negation words that
        # are important to keep for sentiment analysis). Also, only keep words at least 3 characters long
        for postag in all_postags:
            word, pos_type = str(postag[0]), str(postag[1])
            if len(word) > 2 and word not in customised_stopwords and (
                    pos_type.startswith("NN") or pos_type.startswith("VB") or pos_type.startswith("JJ") or
                    pos_type.startswith("RB") or pos_type.startswith("DT")):
                postags.append(postag)
        # Lemmatise the review
        lemmatised_review = []
        for postag in postags:
            word = postag[0]
            pos_type = get_wordnet_pos(postag[1])
            if pos_type == '':
                lemma = wl.lemmatize(word)
            else:
                lemma = wl.lemmatize(word, pos_type)
            lemmatised_review.append(lemma)

        lemmatised_review_str = ' '.join(lemmatised_review)

        # Spacy's Named Entity Recognition tool is used to remove named entity types: Organisations, facilities,
        # organisations, locations, products, languages, dates, times, percentages, money, quantities and ordinals
        lemmatised_review_nes_removed = []
        # Segmenting text into noun phrases before named entities can be identified
        noun_chunks = nlp(lemmatised_review_str)
        for token in noun_chunks:
            if token.ent_type == 0:
                lemmatised_review_nes_removed.append(token.text)

        combined_review_cleaned_str = ' '.join(lemmatised_review_nes_removed)
        df.loc[index, "normalised_combined_review"] = combined_review_cleaned_str

        final_doc2vec_features = [word for word in lemmatised_review_nes_removed if word not in custom_exclusions]
        final_doc2vec_features_str = ' '.join(final_doc2vec_features)
        final_doc2vec_features_str += ' ' + row['sideEffects'].lower() + ' ' + row["rating_to_word"]
        df.loc[index, "doc2vec_input_combined_review"] = final_doc2vec_features_str

    # Creating BOW input review
    # Formatting word/tags as tuples
    print("Engineering features, more custom exclusions and stemming")
    ps = PorterStemmer()
    for index, row in df.iterrows():
        review_str = row["doc2vec_input_combined_review"]
        review_str = review_str.replace("side effects", "side_effect")
        review_str = review_str.replace("side effect", "side_effect")
        review_str = review_str.replace("side affect", "side_effect")
        review_str = review_str.replace("no side_effect", "no_side_effect")
        review_str = review_str.replace("mild side_effect", "mild_side_effect")
        review_str = review_str.replace("moderate side_effect", "moderate_side_effect")
        review_str = review_str.replace("extremely severe side_effect", "extremely_severe_side_effect")
        review_str = review_str.replace("severe side_effect", "severe_side_effect")
        review_str = review_str.replace("have have", "have")
        full_review_tokens = word_tokenize(review_str)
        # Feature engineering the step-scale data from the "rating" and "sideEffects" column
        full_review_tokens[-1] = full_review_tokens[-1] + "_rating"
        review_tokens = [word for word in full_review_tokens if word not in custom_exclusions]
        # Stem words for BOW model
        stemmed_review_tokens = [ps.stem(word) for word in review_tokens]
        stemmed_review_tokens_str = ' '.join(stemmed_review_tokens)
        df.loc[index, "bow_input_combined_review"] = stemmed_review_tokens_str

    # Save processed file to pickle file so processing does not need to be repeated
    print("Saving processed dataframe to pickle file")
    df.to_pickle(pkl_processed_file)
    return df


# Helper function that gets the POS type of a token for named entity recognition (using spaCy)
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# Returns the corpus derived from a column in a dataframe, to use for word embeddings model
def read_corpus(df, classifications_dict, corpus_column, tokens_only=False):
    for row_tuple in df.itertuples():
        drug_review = df.loc[row_tuple.Index, corpus_column]
        tokens = simple_preprocess(drug_review)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            effectiveness = classifications_dict[df.loc[row_tuple.Index, "effectiveness"]]
            yield TaggedDocument(tokens, [effectiveness])


# Creates word embedding vectors using the doc2vec model and saves them to a CSV file
def create_word_embeddings_csv(df, model, output_csv):
    with open(output_csv, "w+") as f:
        for index, row in df.iterrows():
            model_vector = model.infer_vector(row['doc2vec_input_combined_review'].split())
            if index == 0:
                header = ",".join(str(ele) for ele in range(1000))
                f.write(header)
                f.write("\n")
            line1 = ",".join([str(vector_element) for vector_element in model_vector])
            f.write(line1)
            f.write('\n')
        f.close()


# Builds a dictionary of unique class labels and maps them to integers for representation in a model y dataset
def build_classifications_dict(df):
    class_names = list(df.effectiveness.unique())
    classifications_dict = {}
    for index, classification in enumerate(class_names):
        classifications_dict[classification] = index + 1
    return classifications_dict


# Creates the xy vectors representing the model input features
def build_xy_features(df, pkl_file, x_column, y_column, lexicon_list, classifications_dict):

    # Load the pickle file if this data has already been processed in the past and saved
    if exists(pkl_file):
        print("Loading the pre-processed featureset from a pickle file. Filename: " + pkl_file)
        return pandas.read_pickle(pkl_file)

    ngrams = []
    for term in lexicon_list:
        ngrams.append(term[0])

    feature_set = []
    for index, row in df.iterrows():
        bow_review_str = row[x_column]
        # If there is no review available for a row, no need to process it,
        # continue on processing the next line
        if pandas.isnull(bow_review_str):
            continue
        features = numpy.zeros(len(lexicon_list))
        classification = classifications_dict[row[y_column]]

        for ngram in ngrams:
            if ngram in bow_review_str:
                index_value = ngrams.index(ngram)
                features[index_value] += 1

        features = list(features)
        feature_set.append([features, classification])

    with open(pkl_file, "wb") as f:
        pickle.dump(feature_set, f)
    print("Saved xy features as pickle file: " + pkl_file)
    return feature_set


# Builds a lexicon from a dataframe based on TF-IDF scores and N-gram size
def build_processed_lexicon(df, pkl_file, lexicon_column, lexicon_size=800, ngram_min=2, ngram_max=3, min_term_freq=5):
    if exists(pkl_file):
        print("Loading lexicon from file: " + pkl_file)
        return pandas.read_pickle(pkl_file)

    tfifd_vectorizer = TfidfVectorizer(ngram_range=(ngram_min, ngram_max), sublinear_tf=True,
                         max_features=int(lexicon_size * 1.5), min_df=min_term_freq)
    train_tfifd_vectorizer = tfifd_vectorizer.fit_transform(df[lexicon_column])
    vocab = tfifd_vectorizer.get_feature_names()
    distance = numpy.sum(train_tfifd_vectorizer, axis=0)
    vocab_tfidf_df = pandas.DataFrame(distance, columns=vocab)
    tfidf_df_ranked = vocab_tfidf_df.sort_values(vocab_tfidf_df.first_valid_index(), axis=1, ascending=False)

    lexicon = []
    for (words, score) in tfidf_df_ranked.iteritems():
        lexicon.append((words, score.values[0]))

    # Return lexicon/vocabulary list with the size specified by the size parameter
    final_lexicon = lexicon[:lexicon_size]

    with open(pkl_file, "wb") as f:
        pickle.dump(final_lexicon, f)
    print("Saved lexicon as pickle file: " + pkl_file)
    return final_lexicon


# Loads the lexicon for the "unprocessed run". The words in the lexicon are selected at random from the reviews data
def random_unprocessed_lexicon():
    random_lexicon = ['word.', 'alleviated', 'said,', 'inserted,', 'vicious', 'calcium', 'Memory', 'apetite', 'limbs-',
                      'trazodone', 'colon', 'Lamotrogine', '97.5', 'OVERSTRAIN', 'work..', '(meat(', 'wanted.',
                      'Reduces', 'lose)', 'cheek,', 'patient', 'performace', 'metabolized', 'ached', 'senses',
                      'happned', 'nosebleeds,', "'here'", 'resolution.', '(much', 'external', 'zantac', 're-organize',
                      'cry,', "That's", 'SYMPTONS...STOPPED', 'However,I', 'ascertain', 'lips', 'Chills', 'word-recall',
                      'above,', 'OBGYN.', 'She', 'helpful.', 'coniderably', 'hypnotics.', 'High', 'nonprescription',
                      'specialist)', 'Prometrium', 'worthwhle.', 'possibilty', "(I've", 'stupud,', '10mg/day',
                      'probelem.', 'quit,', 'valum', '18-24', 'diagnosed,', 'Colonoscopy', '28,', 'nostrums,',
                      'occasion.', 'Puff"', 'xanax,', 'Retinova', 'unability', 'help', 'check-up', 'circular',
                      'employment.', 'remeron', 'syndrome).', 'TIL', 'life..', 'kg,', 'absence', 'responsiveness',
                      'yawning.', 'lack', "SSRI'S,", 'categorized', 'worry/', 'Uncle', 'fluconazole', 'ANITHYST',
                      'casings', 'hits', 'phenominal', 'starters,', 'tonight.', 'Adding', 'heat,', 'intertwined,',
                      'motionless', 'applying.it', 'texture.', 'reluctantly', 'daily', 'HUNG', 'Chantix.', 'Avage.',
                      'cage', 'antiacids.', "'asshole'.", 'returns.', 'Percocet,', 'mother', '5.1', 'weak,', '(Beck',
                      '(Ibuprofen)', 'rule', '(daily', 'extremley', 'challenges.', 'dosages', 'asprin', 'couch',
                      'ANTIBOTICS', 'different.', 'sleeping,heart', 'itthe', 'Tussionex', 'augmentation).', 'pepacid',
                      'rod', 'doctor;', 'functionally', 'befefits', '(No', 'presentations.', 'forty', '(minimised',
                      'waxing', 'charm--fast', '"P"', 'spining.', '(for', 'counts', 'curing', 'senior', 'astonishing.',
                      'rely', 'indenting', 'limited', 'More', 'blurry', 'Light', 'retina-a', 'pregnancy', 'tweaked',
                      'allerigies', '300mg.', 'restructure', 'Salofalk', 'REDNESS', 'anger', 'natal)', 'traumatizing',
                      '7', 'Risperdal', 'Ultimately', 'esphogus', 'exstremely', 'ins', 'AeroChamber.', '6months',
                      '(tibia', 'PCOS.', 'assessment.', '2-3am,', "Cosmetics'", 'erection', 'gradual.', 'murder!',
                      'ones!', 'tore', 'tells', 'sedated', 'HEAL', 'Affects', 'backpain.', 'annual', 'Unihroid',
                      'Seroquel...', '15)', 'zombie--totally', '(SPF', '(myself', 'sooner', 'hobbies', 'straigntened',
                      'away!', 'place,', 'Burning,', 'surreal.', 'mantally', 'servere,', 'hospitalcould',
                      'water/fluids.', 'tim.', 'SCAR', 'session', 'herpes', 'COLD.', 'worried,', '5x', 'searched',
                      'of,', 'into.', 'ocean.', 'attempted', 'boost.', 'interacts', 'irritating', 'battling', 'U.S.)',
                      'benefits.', 'Sundays.', 'evedybody', 'enzymes,', 'surpised', '"punch".', 'number', 'unusual,',
                      'bend', 'retail', 'shoot', 'Accutane.', 'sensitivy', 'embolism', 'tauted', 'comfort.', 'alphagan',
                      'http://www.marketamerica.com/topproducts-13009/isotonix-opc3.htm', 'call', 'survived', 'cleard',
                      'hurting', 'sratch', 'School', 'Prednisone', 'responded', 'receptors', 'proud', 'looking',
                      'tranquilizer.', 'dress', 'implants', 'addicts', 'photosensitivity', 'wax', "keeled'",
                      'organizational', 'calm,', 'trip.', '5).', 'rupture.', '(like', 'witout', 'hematocrit',
                      'nervousness.', 'METHIMAZOLE', 'modify.', 'inhalor.', 'Previcid.', 'FMS', 'screwing', 'Washed',
                      '"chill",', 'ejaculate', 'partners', 'periodically.', 'anti-depressents', 'factories,',
                      '150mg/day.', 'happened,', 'terribly.', '(though', 'determine', 'numb', "MAOI's", '25mg.',
                      'WEEKS', 'accounts', 'Anyone.', 'depression?', 'Parlodel', 'chewable', '85,', 'trazadone.',
                      'maintenace', 'South', 'Yesterday', 'believes', 'lump', '(reduced', 'solids', '1am,', 'OUTWEIGH',
                      'Psoriasis', 'F.', 'pinched', 'continually;', 'Topamax,', 'year,maybe', 'afer', 'sooner--my',
                      'Photo-sensitivity.', 'worse!!', 'often', '(3)', 'organised', '105', 'inch', 'mid-upper',
                      'heartbeat', 'another', 'Relaxing.', 'hours;', 'ect.', 'Infection,', 'Amoung', 'hours,',
                      'diagnosis).', 'eliminated,', 'Time', 'overt', 'DOCTER', 'wks', 'Spinal', 'poorly', 'tingling)',
                      'regrowing', 'PPI', 'resilient', 'comments.', 'spacey,', 'suboxone.', 'pandoras', '"patch"',
                      '(nap)', 'stupor-like', 'pms', 'WHISLT', 'mins', 'wrinkled', 'fibriods.', 'chills', 'adhered',
                      'OCD.', 'allot', 'gat', 'weekend.', 'heaves', 'Allow', 'Sundberg,', 'called,', 'folds', 'jumpy',
                      'formed', 'taught', 'acid', '50+', '+Night-time', 'Sulindac,', 'relaxation', 'hs.', '37',
                      'clotrimazole', 'Though,', 'K', 'propertly,', 'flashes;they', 'edgy,', 'upset,', 'Hopkins.',
                      'garage).', 'lungs.', 'out,"', 'medications"', 'Fluoxetine,', 'performed', '(Temporary)',
                      'spine),', 'noticed', 'needs.', 'cramps,', 'passive', 'minimizing', 'sporadically', 'Initially,',
                      'acted.', 'risperidone.', 'manageable).', 'rising', 'MAY', 'sometimes)', 'prescribed,', '#3),',
                      'Fortnightly', 'medicaiton', '"kicked', 'sorting', '15.', 'mfrs.', 'Similar', 'actions', '14.5)',
                      'forbade', 'loss)', 'moisturization.', 'unborn', 'moods;', 'exema.', 'decision.', 'live',
                      'insturctions', 'concern,', 'intractable', 'betaseron', 'TAKEN', 'EYES,', 'increased', 'Lessened',
                      'mnths', 'entering', 'again.', '18.', 'therapies.', 'Fell', 'clock)', 'driving', 'considered.',
                      'side-effects,', 'African', 'lines', 'orgasm.', 'Yasmine', 'Mayo', 'Ponstel', 'drawbacks',
                      'window.', 'aloe', 'television', 'Begginng', 'refusing', "Shuermann's", 'Witout', 'mention',
                      'expected,', 'born,', 'themselves),', 'heartburn,', 'stiffness', 'age,', 'awe', 'pregnancy,',
                      'Voice', 'intensified.', 'performance', 'ages', 'dopey.', 'golf', 'addictive,', 'far/so', 'lip',
                      'tea', 'itself!', 'thirty', 'yet!)', 'enormous', 'radical', 'hardly', 'Unlike', '"normal."',
                      'survive', 'instance.', 'introducing', 'diminishes.', 'Getting', 'matters', 'chiropractor,',
                      'excersise;', 'patients,', 'etc..)', 'biaxin', 'hell.', 'diagnosised', 'Penlac', 'vommiting,',
                      'range"', 'force!', 'spinal', 'motrin,', 'Oxy', 'ideas.', '3:20am,', 'clearer.',
                      'magnesium-containing', 'fun.', 'stiff', 'adapalene.', '(140/90', 'largely', 'derms', 'messing',
                      'formula', 'arrest.', 'muscels', 'MONTHS.', 'ago.', 'ski', 'constipation.', 'plastic', 'ME.',
                      '(ezetimibe)', 'hydrogen', ',stay', 'neurological', 'DOSAGE', 'inflamatories', 'route',
                      'Bio-Advance,', 'vera', 'POINT', 'synthetic', 'creases', 'Well...', 'naps', 'indigestion',
                      'matter', 'Lyrica;.', 'movements', 'medication!', "(I'm", 'info.', 'thing', 'belly', 'Alcohol.',
                      'Vitamins', 'body-mind', 'Bright', 'prime', 'relates', 'vit./min.', 'raise', "Benadryl...I'd",
                      '97.8', 'reply', 'lips,', 'abfter', 'ucler', 'biest', 'thoughout', 'medicxal', 'amongst',
                      'thing.So', 'Africa)', 'Adapalene', 'becomming', 'VEINS', 'soem', 'used.', 'tightening', 'b12',
                      'voices', 'correctly,', 'results,', 'lethargic,', 'pediatrics', 'syc', 'existent', 'thereafter,',
                      '"zombie"', 'DESPARATE.', 'moderation', 'rheumatoid', 'good', 'buttocks.', '-2.9.',
                      'ameliorate/deter', 'boyfriend.', 'Dose', 'should,', 'focusing;', 'advance)Start', '80',
                      'acne,blackheads', 'lamisil', 'details,', 'enlargement,', '3.', 'luxury-', 'Lantus', 'limiting',
                      'Eventual', 'exactly', 'thirst)', 'respiratory', 'perceived.', 'flaky,', 'reward', 'intentions',
                      'degree,', 'Etreme', 'calmness', '"eh...who', 'of:', 'sex', 'Miss', 'Topomax', '(prior',
                      'hormones.', 'irritation,', 'robbed', '-stop', 'disappeaed,', '(constipation)', 'nite', 'aspect',
                      'x-rays', '@', '10,000', 'Tazret.', '-Cleared', 'trazaone', 'breast,', 'peaceful', 'adhd',
                      'trips', 'DRESSING', '(vigorously)!', 'partner', 'MONTH!', 'tweek', 'immunosuppressants.',
                      'chatter', 'clots,', 'ringing', 'er', 'Dr.', 'denial', 'graduation,', 'candy', 'pill/day.',
                      'tremendous.', 'takin', '(use', '(hands,', '3am', 'pupils,', 'most.', 'drilled', "don't",
                      'Monday', 'ginger', 'sick,', 'vicodin', 'means', 'drug!!', 'choices...', 'perenniel', 'knees...',
                      '(104/70)', 'worthwhile,', 'ANYTHING', 'paying', '10kg', 'speeches', 'If', 'Low',
                      'evenings/early', 'IV', '"deal"', 'Boyfriend', 'teeth.', '12.5,', 'weeks)', 'MIN-OVRAL',
                      'counting!)', 'ITCHING', 'tremendous', 'benifits', 'lifesaver', 'perfectly', 'Would', 'describe',
                      'Valtrez', 'premenstrual', 'ambulance.', 'ex', 'tissue.', 'brief', 'disciplined', 'feeling.',
                      'growing,', 'FROM', '(as', 'Young".', 'through', 'veins', 'cheap', 'CHILDREN.', 'sessions',
                      'ankles.', 'activited', 'OBAGI', 'minimum,', '60-mg', 'raging', 'QUIT', 'hgher', 'cracker.', '"I',
                      "'ll", 'sodas', 'ml', 'tired,', 'gagging.', 'sparingly', 'dysthymia', 'BRAIN.', 'sinusitis.',
                      'Oh,', 'quck', 'rollercoaster', 'Much', 'decrease', '5lbs', 'priced', 'tighter/drier,',
                      'ceratinly', 'classifications', 'describes', 'Levoxyl)', 'ho9ur', 'recovery.', 'Part',
                      'Chapstick', '"appetite"', 'happening/being', 'table,', 'routinely.', '(kidney', 'reactivate,',
                      'comfort', 'OXYCODONE', 'augment', '25mg,', 'Shame', 'dry-heaving,', 'experiencing', 'insistant',
                      'control;', 'less...', 'albuterol.', '2.)Then', 'AUSTRAILAN', 'Redness', 'disorders....but',
                      'Fuzziness', 'energy--kept', 'indispensible', 'measures]', 'religious', 'temporarily,', 'Lower',
                      'discouraged', 'movenments', 'erectile', 'alchohol', 'flue', 'apathy,', 'shattering,but',
                      'http://www.askapatient.com/viewrating.asp?drug=19658&name=CLARITIN', 'alternative',
                      'Hyperthyroidism', 'syntroid', 'retaining', 'anxious.', 'oct', 'phenergan', 'ordinarily',
                      'better', 'over.My', 'proscar.', 'takiing', 'allergins.', 'purpose', 'shower.', 'insulin',
                      'esteem.', 'awkward.', 'Doeses', 'choking', 'surroundings,']
    return random_lexicon
