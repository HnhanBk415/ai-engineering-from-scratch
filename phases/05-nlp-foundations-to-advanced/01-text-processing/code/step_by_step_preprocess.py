import re

# Tokeniization theo regex
def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)

# Stemming bằng thuật toán Porter Stemming
# Step 1a: xử lý sses, ies, ss, s
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]

# words = ["caresses", "ponies", "caress", "cats"]

# for word in words:
#     print(word, "->", stem_step_1a(word))


# Lemmatizatio -> đưa từ về dạng từ điển
# Nhận một từ và từ loại (POS - Part of speech).
# Chuyển từ về chữ thường.
# Tra bảng LEMMA_TABLE.
# Nếu có kết quả, trả về lemma.
# Nếu không, thử một vài quy tắc dự phòng.
# Nếu vẫn không có, giữ từ ở dạng chữ thường.

LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)

    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]

    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]

    return word.lower()

# print(lemmatize("ran", "VERB"))       # run
# print(lemmatize("better", "ADJ"))     # good
# print(lemmatize("cats", "NOUN"))      # cat
# print(lemmatize("watched", "VERB"))   # watched

def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}

text = "I'm Nguyen Hoang Nhan, I want to better NLP skills"
print(preprocess(text))