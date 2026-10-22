# Câu 1
# Easy. Extend tokenize to keep URLs as single tokens. 
# Test: tokenize("Visit https://example.com today.") should produce one URL token.
import spacy
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

nlp = spacy.blank("en") # Tách token, nhận diện URL, bằng tiếng Anh

def tokenize(text):
    doc = nlp(text) # phân tách văn bản thành các token
    return [token.text for token in doc]

text = "Visit https://example.com today."

print(tokenize(text))

# Câu 2
# Medium. Implement Porter step 1b. 
# If a word contains a vowel and ends in ed or ing, remove it. 
# Handle the double-consonant rule (hopping -> hop, not hopp).

#logic: nếu âm cuối có ed or ing thì xóa tạm thời, rồi check từ còn lại là nguyên âm hay mấy phụ âm kép hay không, xóa hoặc khôi phục lại
def porter_step_1b(word):
    word = word.lower()
    vowels = "aoeui" # nguyên âm

    # Check hậu tố ed or ing
    if word.endswith("ed"):
        stem = word[:-2]
    elif word.endswith("ing"):
        stem = word[:-3]
    else: return word

    #check stem có nguyên âm không
    # any() trả về True nếu ít nhất một phần tử thỏa mãn điều kiện.
    if not any(char in vowels for char in stem): 
        return word

    # xử lý phụ âm kép
    if(
        len(stem) >= 2 and stem[-1] == stem[-2] 
        and stem[-1] not in vowels
        and stem[-1] not in "lsz"
    ): 
        stem = stem[:-1]

    return stem

# Câu 3
# Hard. Build a lemmatizer that uses WordNet as a lookup table but falls back to your Porter stemmer when WordNet has no entry. 
# Measure accuracy on a tagged corpus against plain WordNet and plain Porter.

def my_porter(word):
    return porter_step_1b(word)


def wordnet_only(word, pos):
    word = word.lower()
    lemma = wordnet.morphy(word, pos)

    if lemma is not None:
        return lemma

    return word

def hybrid_lemmatize(word, pos):
    word = word.lower()

    # Ưu tiên WordNet
    lemma = wordnet.morphy(word, pos)

    if lemma is not None:
        return lemma

    # WordNet không tìm thấy -> Porter
    return my_porter(word)

GOLD_DATA = [
    ("cats", "n", "cat"),
    ("dogs", "n", "dog"),
    ("children", "n", "child"),
    ("mice", "n", "mouse"),
    ("running", "v", "run"),
    ("playing", "v", "play"),
    ("hopping", "v", "hop"),
    ("walking", "v", "walk"),
    ("went", "v", "go"),
    ("were", "v", "be"),
    ("better", "a", "good"),
    ("best", "a", "good"),
]

def evaluate(dataset, method):
    correct = 0

    for word, pos, gold in dataset:
        predicted = method(word, pos)

        if predicted == gold:
            correct += 1

    return correct / len(dataset)

porter_nltk = PorterStemmer()

methods = {
    "WordNet": wordnet_only,
    "Porter NLTK": lambda w, p: porter_nltk.stem(w),
    "My Porter": lambda w, p: my_porter(w),
    "Hybrid": hybrid_lemmatize,
}

print("ACCURACY COMPARISON")

for name, method in methods.items():
    accuracy = evaluate(GOLD_DATA, method)
    print(f"{name:15}: {accuracy:.2%}")
