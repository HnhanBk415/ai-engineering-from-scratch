# Câu 1 Tạo character n-grams cho playing và played, rồi đo mức độ trùng lặp giữa hai tập bằng Jaccard similarity.

# 1. Tạo character n-grams
def char_ngrams(word, n_min=3, n_max=6):
    word = f"<{word}>"

    grams = {word}

    for n in range(n_min, n_max + 1):
        for i in range(len(word) - n + 1):
            grams.add(word[i:i+n])

    return grams


# 2. Tạo hai tập n-grams
grams1 = char_ngrams("playing")
grams2 = char_ngrams("played")

# 3. Tính giao và hợp
intersection = grams1 & grams2
union = grams1 | grams2

# 4. Jaccard similarity
jaccard = len(intersection) / len(union)


