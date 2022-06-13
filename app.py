import pandas as pd
import numpy as np
import streamlit as st
import pickle
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
import spacy
from nltk.stem import WordNetLemmatizer

!python -m spacy download en_core_web_lg

stopwords = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he',
'him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who',
'whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing',
'a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during',
'before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when',
'where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very',
's','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',
"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",
'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
stop = set(stopwords)
punctuation = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']
stop.update(punctuation)

# load trained model
model = pickle.load(open('model_Dis_Tweet.pkl', 'rb'))
	
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

# Create function for data loading
if uploaded_file:
	df = pd.read_csv(uploaded_file, sep=',', skipinitialspace=True)
	
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
