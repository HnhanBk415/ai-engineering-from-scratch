import math

# Tính DF
def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])

    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df

# Tính IDF
def inverse_document_frequency(df, n_docs):
    return[ 
        math.log((n_docs + 1) / (d + 1)) + 1 
        for d in df
    ]

# Ghép lại tính TF - IDF
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)

    ans = []

    for row in bow_matrix:
        length = sum(row)

        tf = [c/ length if length else 0 for c in row]

        ans.append([t * i for t, i in zip(tf, idf)])

    return ans