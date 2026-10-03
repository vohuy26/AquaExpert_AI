import os
import pandas as pd

def load_dataset_from_folder(base_folder_path):
    """
    Đọc toàn bộ file .txt từ các thư mục con (mỗi thư mục con là một nhãn bệnh).
    """
    texts = []
    labels = []
    
    if not os.path.exists(base_folder_path):
        print(f"[Cảnh báo] Không tìm thấy thư mục: {base_folder_path}")
        return pd.DataFrame({'text': [], 'label': []})
        
    for disease_name in os.listdir(base_folder_path):
        disease_dir = os.path.join(base_folder_path, disease_name)
        
        if os.path.isdir(disease_dir):
            for file_name in os.listdir(disease_dir):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(disease_dir, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            if content:
                                texts.append(content)
                                labels.append(disease_name)
                    except Exception as e:
                        print(f"Lỗi khi đọc file {file_path}: {e}")
                        
    return pd.DataFrame({'text': texts, 'label': labels})

if __name__ == "__main__":
    # Tự động lấy đường dẫn tuyệt đối tới thư mục gốc dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    train_path = os.path.join(project_root, "data", "Train_Full12")
    test_path = os.path.join(project_root, "data", "Test_Full12")
    
    print(f"-> Đang tìm dữ liệu Train tại: {train_path}")
    df_train = load_dataset_from_folder(train_path)
    df_test = load_dataset_from_folder(test_path)
    
    if len(df_train) > 0:
        print(f"-> Thành công! Đã đọc {len(df_train)} mẫu Train từ {df_train['label'].nunique()} loại bệnh.")
        print(f"-> Đã đọc {len(df_test)} mẫu Test.")
    else:
        print("-> Chưa đọc được dữ liệu. Kiểm tra lại vị trí thư mục Train_Full12 trong data/")