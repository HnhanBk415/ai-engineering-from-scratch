
# Cho docs là danh sách những document đã tokenize
docs = [
    ["the", "cat", "sat"],
    ["the", "dog", "sat"],
    ["the", "cat", "ran"]
]
# B1: Build vocabulary

def build_vocab(docs):
    vocab = {} # Tạo dictionary để lưu từ và chỉ số cột tương ứng
    
    # Duyệt từng document, sau đó duyệt từng token trong document.
    for doc in docs:
        for token in doc:
            if token  not in vocab:
                index = len(vocab)
                vocab[token] = index

    return vocab

# output của vocab: {'the': 0, 'cat': 1, 'sat': 2, 'dog': 3, 'ran': 4}
# Build bag of words
# Thứ nhất – Khởi tạo ma trận toàn số 0.
# Thứ hai – Xác định document đang xử lý.
# Thứ ba – Đếm từ.

def bag_of_word(docs, vocab):
    matrix = []

    for doc in docs:
        row = [0] * len(vocab) # mỗi hàng có số cột bằng với số chiều dài của vocab luôn
        matrix.append(row) 
        #[0,0,0,0,0]
        #[0,0,0,0,0]
        #[0,0,0,0,0]

    for i in range(len(docs)):
        doc = docs[i]
        for token in doc:
            if token in vocab:
                index = vocab[token]
                matrix[i][index] = matrix[i][index] + 1

    return matrix

vocab = build_vocab(docs)
print(bag_of_word(docs, vocab))