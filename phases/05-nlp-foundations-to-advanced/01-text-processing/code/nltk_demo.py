import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Tải các tài nguyên cần thiết
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

text = "The cats were running."

#step1: Tokenization
tokens = word_tokenize(text)

#step2: stemming (dùng Porter Stemming hoàn chỉnh từ thư viện)
stemmer = PorterStemmer()

stems = [stemmer.stem(token) for token in tokens]

#step3: Pos Tagging
tagged = pos_tag(tokens)

#step4: convert pos tags
def convert_pos(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"

#step5: lemmatization
lemmatizer = WordNetLemmatizer()

lemmas = [ lemmatizer.lemmatize(token.lower(), pos=convert_pos(tag)) for token, tag in tagged ]

print("Tokens:", tokens)
print("Stems:", stems)
print("POS:", tagged)
print("Lemmas:", lemmas)