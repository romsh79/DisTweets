import pandas as pd
import numpy as np
import plotly.express as px
from urllib.request import urlopen
import streamlit as st

import pickle
import nltk
import string
from nltk import pos_tag
from nltk.corpus import wordnet
import spacy
from nltk.stem import WordNetLemmatizer
stopwords = nltk.corpus.stopwords.words('english')

# load trained model
with open('model_Dis_Tweet.pkl', 'rb') as file: 
    model = pickle.load(file)

stop = set(stopwords)
punctuation = list(string.punctuation)
stop.update(punctuation)

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)
nlp = spacy.load('en_core_web_lg')
	
	
# Header    
st.title('Classification of twitter records')
st.write("""Here we will identify twitter records on the disaster content""")
# Upload file of twitter records
uploaded_file = st.file_uploader(
        label='Select the file to classify')
# Tune the side panel
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is Open Source dashboard.
    """
)
st.sidebar.info("Some text [link in web](https://github.com/yuliianikolaenko/COVID_dashboard_proglib).")
st.sidebar.image("grandma.jpg")

# Params of the data we want to classify
#DATA = ('Data/DisasterTweets/test.csv')
DATA_COLUMNS = ['id','text']
@st.cache # for optimization of application

# Create function for data loading
def load_data():
    df = pd.read_csv(uploaded_file, sep=',', skipinitialspace=True)
    return df
# Apply function
df = load_data()

# How many data rows to show
showdata = st.sidebar.slider("How many data rows to show", min_value=1, max_value=30, value=5)

# Create checkbox on the side panel to show the data
show_data = st.sidebar.checkbox('Show raw data')
if show_data == True:
    st.subheader('Raw data')
    st.markdown(
        "#### Data on twitter records")
    st.write(df[DATA_COLUMNS].head(showdata))

# PERFORM CLASSIFICATION
result = st.button('Start prediction')
if result:
	X_for_pred = df.text
	# lemmatize
	X_for_pred = X_for_pred.apply(lemmatize_words)
	# combine all the word vectors into a single document vector
	with nlp.disable_pipes():
		X_vec = np.array([nlp(text).vector for text in X_for_pred])
	# classification
	preds = model['model'].predict(X_vec)
	# create dataframe
	pred_df = pd.DataFrame(np.array([df['id'].values, preds]).transpose(),columns=[['id','target']])
	pred_df['text'] = df.text.values
	# print the result
	st.write('**Results of classification**')
	st.write(pred_df)
