# Câu 1 Tạo corpus gồm 20 câu về mèo và chó.
# Train Word2Vec trong 200 epochs, sau đó kiểm tra xem dog có nằm trong top 3 từ giống cat nhất không.

from gensim.models import Word2Vec

# 1. Tạo corpus gồm 20 câu
docs = [
    "the cat is a friendly pet",
    "the dog is a friendly pet",
    "my cat likes to play outside",
    "my dog likes to play outside",
    "the cat eats food every day",
    "the dog eats food every day",
    "this cat sleeps near me",
    "this dog sleeps near me",
    "a cat lives in our house",
    "a dog lives in our house",
    "the cat needs water and food",
    "the dog needs water and food",
    "our cat is cute and happy",
    "our dog is cute and happy",
    "the cat runs in the garden",
    "the dog runs in the garden",
    "i love my little cat",
    "i love my little dog",
    "the cat plays with a ball",
    "the dog plays with a ball"
]

# 2. Tokenization
sentences = [doc.split() for doc in docs]

# 3. Train Word2Vec (Skip-gram)
model = Word2Vec(
    sentences=sentences,
    vector_size=30,
    window=3,
    min_count=1,
    sg=1,
    negative=5,
    epochs=200,
    workers=1,
    seed=42,
    sample=0
)

# 4. Tìm 3 từ giống "cat" nhất
top3 = model.wv.most_similar("cat", topn=3)

print("Top 3 similar words to cat:")

for word, score in top3:
    print(f"{word}: {score:.4f}")

# 5. Kiểm tra điều kiện bài tập
found = any(word == "dog" for word, _ in top3)

print("\nDog in top 3:", found)

# Câu 2 Thêm subsampling để giảm tần suất những từ phổ biến, 
# sử dụng ngưỡng 10e-5, rồi đo sự thay đổi cosine similarity của các từ hiếm.

import nltk
from collections import Counter
from gensim.models import Word2Vec

# 1. Tải corpus Brown
nltk.download("brown", quiet=True)
from nltk.corpus import brown

sentences = [
    [word.lower() for word in sent if word.isalpha()]
    for sent in brown.sents()
]

sentences = [s for s in sentences if len(s) >= 2]

# Đếm số lần xuất hiện
freq = Counter(
    word for sent in sentences for word in sent
)

# 2. Cấu hình chung
params = dict(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=3,
    sg=1,
    negative=5,
    epochs=10,
    workers=1,
    seed=42
)

# 3. Model không subsampling
model_normal = Word2Vec(
    **params,
    sample=0
)

# 4. Model có subsampling
model_subsample = Word2Vec(
    **params,
    sample=1e-5
)

# 5. Các cặp từ để kiểm tra
candidate_pairs = [
    ("doctor", "nurse"),
    ("king", "queen"),
    ("cat", "dog"),
    ("car", "truck"),
    ("music", "song"),
    ("artist", "painter")
]

print("COMPARISON")
print("-" * 65)

changes = []

for w1, w2 in candidate_pairs:
    # Chỉ xét các từ đủ điều kiện và tương đối ít gặp
    if not (3 <= freq[w1] <= 250 and
            3 <= freq[w2] <= 250):
        continue

    if w1 not in model_normal.wv:
        continue
    if w2 not in model_normal.wv:
        continue

    before = model_normal.wv.similarity(w1, w2)
    after = model_subsample.wv.similarity(w1, w2)

    delta = after - before
    changes.append(delta)

    print(f"{w1} - {w2}")
    print(f"  Frequency: {freq[w1]}, {freq[w2]}")
    print(f"  Before: {before:.4f}")
    print(f"  After : {after:.4f}")
    print(f"  Change: {delta:+.4f}")
    print()

if changes:
    print("Average change:",
          round(sum(changes) / len(changes), 4))
else:
    print("No eligible word pairs found.")
