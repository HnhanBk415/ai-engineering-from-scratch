# Câu 1 Tính độ tương đồng giữa hai vector TF-IDF đã chuẩn hóa L2. 
# Kiểm tra hai văn bản giống nhau có similarity = 1 và không có từ chung có similarity = 0.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "the cat sat",
    "the cat sat",
    "dog runs fast"
]

# Step 1: tạo vector ID - IDF
vectorize = TfidfVectorizer()
X = vectorize.fit_transform(docs)

# Step2: Tính cosine similarity
sim1 = cosine_similarity(X[0], X[1]) # doc1 giống doc2 -> cosine_similariry = 1
sim2 = cosine_similarity(X[0], X[2]) # doc1 khác doc3 -> cosine_similarity khác 1

# print("Giống nhau:", sim1[0][0])
# print("Không có từ chung:", sim2[0][0])


# Câu 2: Thêm khả năng tạo n-gram vào BoW, trong đó n là số token liên tiếp tạo thành một đặc trưng.
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words_ngrams(docs, n = 2):
    vectorizer = CountVectorizer(
        ngram_range =(n , n)
    )

    matrix = vectorizer.fit_transform(docs)

    return vectorizer.get_feature_names_out()

docs = [
    "the cat sat",
    "the cat ran"
]

vocab = bag_of_words_ngrams(docs, n=2)

print("Vocabulary:", vocab)
