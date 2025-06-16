import math
import string
from collections import Counter

# ===== Step 0: Clean Up Functions =====
def clean_up(text):
    stop_words = set(["and", "or", "the", "is", "in", "to", "a", "of", "for", "on", "with", "as", "by", "an", "are", "i"])
    translator = str.maketrans("", "", string.punctuation)
    cleaned = text.translate(translator).lower()
    words = cleaned.split()
    clean_text = ' '.join([word for word in words if word not in stop_words])
    return clean_text

# ===== Step 1: Tokenization =====
def tokenize(text):
    return text.split()

# ===== Step 2: Build Vocabulary =====
def build_vocabulary(docs):
    vocab = sorted(set(word for doc in docs for word in doc))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_to_index

# ===== Step 3: Compute Term Frequency (TF) =====
def compute_tf(doc, word_to_index):
    tf_vector = [0] * len(word_to_index)
    word_counts = Counter(doc)
    for word, count in word_counts.items():
        if word in word_to_index:
            tf_vector[word_to_index[word]] = count
    return tf_vector

# ===== Step 4: Compute Inverse Document Frequency (IDF) =====
def compute_idf(docs, vocab):
    N = len(docs)
    idf_vector = []
    for word in vocab:
        df = sum(1 for doc in docs if word in doc)
        idf = math.log((N + 1) / (df + 1)) + 1
        idf_vector.append(idf)
    return idf_vector

# ===== Step 5: Compute TF-IDF =====
def compute_tfidf(tf, idf):
    return [tf[i] * idf[i] for i in range(len(tf))]

# ===== Step 6: Cosine Similarity =====
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def magnitude(v):
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + 1e-9)

# ===== Main Program =====

# فایل‌های متنی پیش‌فرض
doc1_path = "doc1.txt"
doc2_path = "doc2.txt"

# خواندن اسناد
with open(doc1_path, "r", encoding="utf-8") as f:
    text1 = f.read()

with open(doc2_path, "r", encoding="utf-8") as f:
    text2 = f.read()

# تمیز کردن و توکنیزه کردن
doc1_tokens = tokenize(clean_up(text1))
doc2_tokens = tokenize(clean_up(text2))

# ساخت واژگان
vocab, word_to_index = build_vocabulary([doc1_tokens, doc2_tokens])

# محاسبه TF
tf1 = compute_tf(doc1_tokens, word_to_index)
tf2 = compute_tf(doc2_tokens, word_to_index)

# محاسبه IDF
idf = compute_idf([doc1_tokens, doc2_tokens], vocab)

# محاسبه TF-IDF
tfidf1 = compute_tfidf(tf1, idf)
tfidf2 = compute_tfidf(tf2, idf)

# تابع برای محاسبه شباهت کسینوسی برای یک کلمه
def compute_word_similarity(word, word_to_index, idf, tfidf1, tfidf2):
    if word in word_to_index:
        # ساخت بردار TF برای کلمه (0 برای سایر کلمات)
        tf_word = [0] * len(word_to_index)
        tf_word[word_to_index[word]] = 1  # کلمه یکبار ظاهر می‌شود
        
        # محاسبه TF-IDF برای کلمه
        tfidf_word = compute_tfidf(tf_word, idf)
        
        # محاسبه شباهت با سند1 و سند2
        similarity_doc1 = cosine_similarity(tfidf_word, tfidf1)
        similarity_doc2 = cosine_similarity(tfidf_word, tfidf2)
        
        return similarity_doc1, similarity_doc2
    else:
        return 0.0, 0.0

# دریافت کلمات برای بررسی شباهت از کاربر
words_input = input("Enter words to check similarity (comma separated): ")
words_to_check = [word.strip() for word in words_input.split(",")]


