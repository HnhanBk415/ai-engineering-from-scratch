# Mở rộng thuật toán BPE để theo dõi sự tăng trưởng của vocabulary. 
# Vẽ đồ thị thể hiện số token trên mỗi ký tự của corpus khi số lần merge tăng lên.
from collections import Counter
import nltk
import matplotlib.pyplot as plt

# 1. Tải dữ liệu
nltk.download("brown", quiet=True)
from nltk.corpus import brown

words = [
    word.lower()
    for word in brown.words()
    if word.isalpha()
][:10000]

corpus = Counter(words)

# 2. Train BPE và theo dõi thống kê
def learn_bpe(corpus, k_merges=120):
    vocab = Counter({
        tuple(word) + ("</w>",): freq
        for word, freq in corpus.items()
    })

    total_chars = sum(
        len(word) * freq
        for word, freq in corpus.items()
    )

    alphabet = {
        char for word in corpus for char in word
    } | {"</w>"}

    def token_ratio():
        # Không đếm </w> đứng riêng
        total_tokens = sum(
            sum(t != "</w>" for t in tokens) * freq
            for tokens, freq in vocab.items()
        )
        return total_tokens / total_chars

    ratios = [token_ratio()]
    vocab_sizes = [len(alphabet)]
    merges = []

    for _ in range(k_merges):
        pair_freq = Counter()

        # Đếm các cặp token liền nhau
        for tokens, freq in vocab.items():
            for pair in zip(tokens, tokens[1:]):
                pair_freq[pair] += freq

        if not pair_freq:
            break

        best, count = pair_freq.most_common(1)[0]

        if count < 2:
            break

        merges.append(best)
        new_vocab = Counter()

        # Ghép cặp có tần suất cao nhất
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i+1]) == best
                ):
                    new_tokens.append(
                        tokens[i] + tokens[i+1]
                    )
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_vocab[tuple(new_tokens)] += freq

        vocab = new_vocab

        ratios.append(token_ratio())
        vocab_sizes.append(
            len(alphabet) + len(merges)
        )

    return merges, ratios, vocab_sizes


# 3. Huấn luyện
merges, ratios, vocab_sizes = learn_bpe(
    corpus,
    k_merges=120
)

# 4. Thống kê
print("Total merges:", len(merges))
print("Initial vocabulary:", vocab_sizes[0])
print("Final vocabulary:", vocab_sizes[-1])

print("Initial tokens/char:", round(ratios[0], 4))
print("Final tokens/char:", round(ratios[-1], 4))

print("Final chars/token:", round(1 / ratios[-1], 4))

# 5. Vẽ đồ thị nén dữ liệu
plt.figure(figsize=(9, 5))

plt.plot(
    range(len(ratios)),
    ratios,
    marker="o",
    markevery=10
)

plt.title("BPE Compression")
plt.xlabel("Number of merges")
plt.ylabel("Tokens per corpus character")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Vẽ đồ thị tăng trưởng vocabulary
plt.figure(figsize=(9, 5))

plt.plot(
    range(len(vocab_sizes)),
    vocab_sizes
)

plt.title("BPE Vocabulary Growth")
plt.xlabel("Number of merges")
plt.ylabel("Vocabulary size")
plt.grid(True)
plt.tight_layout()
plt.show()
