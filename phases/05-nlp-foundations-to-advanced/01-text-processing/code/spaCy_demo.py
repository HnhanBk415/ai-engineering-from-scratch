import spacy

#Nạp pipeline NLP
nlp = spacy.load("en_core_web_sm")

text = "The cats were running."

doc = nlp(text)

for token in doc:
    print(
        token.text, # văn bản gốc
        token.pos_, # Từ loại tổng quát
        token.tag_, # Nhãn từ loại chi tiết
        token.lemma_ # Dạng từ điển
    )