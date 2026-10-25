# Dùng thư viện có sẵn scikit-learn thay vì tự viết tay từng step
from sklearn.feature_extraction.text import(
    CountVectorizer,
    TfidfVectorizer
)

docs = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat ran"
]

# Bag of words
bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)

print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

print(tfidf_matrix.toarray().round(3))