# Capstone Project: NLP
# This program contains two functions, one to preprocess a column of text data for analysis, 
#  and another to analyse the sentiment and subjectivity of the text.
# There is also a block of code to test similarity.
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
nlp_md = spacy.load('en_core_web_md')

# Read in Amazon Reviews csv file using pandas.
dataframe = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# dropna empty data
drop_data = dataframe.dropna(subset=['reviews.text'])
# Assign variable to reviews with empty rows removed.
reviews_column = drop_data['reviews.text']

# Define function to pre-process data;
    # for word in review
       # if a stopword or punctuation continue
       # else lemmatise and append to cleaned sentence.
    # return cleaned sentence.
def clean_data(review):
    cleaned = ""
    for word in nlp(review):
        if word.is_stop or word.is_punct:
            continue
        else:
            lemma_text = word.lemma_
            cleaned += lemma_text + " "
    return cleaned 

# Define function for sentiment analyisis;
    # tokenise arguemnt 
    # assign variable to polarity of arguement 
    # assign variable to sentiment.subjectvity of arguement 
    # if/ elif/ else statements to determine print statement based on polarity rank
    # if/ elif/ else statements to determine print statement based on subjectivity rank
    # print statement for readability.
def sentiment_analysis(reviews):
    reviews = nlp(reviews)
    polarity = reviews._.blob.polarity 
    sentiment = reviews._.blob.sentiment.subjectivity
    if polarity > 0.5:
        print_polarity = (f"This review has a strong positive sentiment of {polarity}")
    elif polarity >= 0:
        print_polarity = (f"This review has a positive sentiment of {polarity}" )
    elif polarity >= -0.5:
        print_polarity = (f"This review has a negative sentiment of {polarity}")
    else:
        print_polarity = (f" This review has a strong negative sentiment of {polarity}")
    if sentiment > 0.75:
        print_sentiment = (f" but is highly objective, with a subjectivity of {sentiment}")
    elif sentiment > 0.5:
        print_sentiment = (f" but is relatively objective, with a subjectivity of {sentiment}")
    elif sentiment  >= -0.25:
        print_sentiment = (f" and is relatively subjective, with a subjectivity of {sentiment}")
    else:
        print_sentiment = (f" and is highly subjective, with a subjectivity of {sentiment}")
    print("\nSentiment Analysis\nSentiment:    [Negative  (-1 - 1) Positive]\nSubjectivity: [Subjective (0 - 1) Objective]")
    print(f"\n'{reviews}'")
    return print_polarity + print_sentiment 

reviews = reviews_column[567]
print(f" \n {sentiment_analysis(reviews)}")

# Semantic Similarity 

# Assign four variables to tokenised, clean sample reviews.
sample1 = nlp_md(clean_data(reviews_column[2]))
sample2 = nlp_md(clean_data(reviews_column[6]))
sample3 = nlp_md(clean_data(reviews_column[49]))
sample4 = nlp_md(clean_data(reviews_column[186]))

# Assign a variable to tokenised, clean test review.
Test_review = nlp_md(clean_data(reviews_column[88]))
print(f"\nSemantic Similarity\nTest Review: {Test_review} \n")

# Add sample reviews to a list.
sample_list = [sample1, sample2, sample3, sample4]

# Number and print sample reviews.
count = 0
for sample in sample_list:
    count += 1
    print(f"Sample {count}. {sample}")

# print sample number and similarity 
count = 0
for sample in sample_list: 
    similarity = sample.similarity(Test_review)
    count += 1
    print(f"\n Test Sample Similarity {count}. {similarity}\n")

# Terminate.
exit()





