import os
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Thêm đường dẫn thư mục gốc dự án vào hệ thống
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from core.data_loader import load_dataset_from_folder

def train():
    train_path = os.path.join(project_root, "data", "Train_Full12")
    test_path = os.path.join(project_root, "data", "Test_Full12")
    
    print("1. Đang nạp dữ liệu từ thư mục Train và Test...")
    df_train = load_dataset_from_folder(train_path)
    df_test = load_dataset_from_folder(test_path)
    
    X_train, y_train = df_train['text'], df_train['label']
    X_test, y_test = df_test['text'], df_test['label']
    
    # 2. Vector hóa văn bản với TF-IDF (trích xuất đặc trưng từ vựng tiếng Việt)
    print("2. Đang vector hóa văn bản bằng TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 3. Huấn luyện mô hình Naive Bayes (Chương 5)
    print("3. Đang huấn luyện mô hình Multinomial Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # 4. Đánh giá mô hình trên tập Test_Full12 độc lập
    print("4. Đánh giá kết quả trên tập Test_Full12:")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"-> Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
    print(f"==========================================\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 5. Lưu mô hình và bộ vectorizer ra thư mục models/
    model_save_path = os.path.join(current_dir, "disease_classifier.pkl")
    vec_save_path = os.path.join(current_dir, "tfidf_vectorizer.pkl")
    
    joblib.dump(model, model_save_path)
    joblib.dump(vectorizer, vec_save_path)
    
    print(f"-> Đã lưu mô hình tại: {model_save_path}")
    print(f"-> Đã lưu Vectorizer tại: {vec_save_path}")

if __name__ == "__main__":
    train()