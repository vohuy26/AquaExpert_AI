import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_and_preprocess_water_data():
    """
    Tự động tìm đọc các file IoTPond*.csv hoặc waterquality.csv
    và chuẩn hóa về các cột: temp, ph, do, salinity
    """
    # Tìm kiếm các file csv trong thư mục data hoặc thư mục gốc
    csv_files = glob.glob("data/*.csv") + glob.glob("*.csv")
    water_dfs = []
    
    for f in csv_files:
        try:
            df_temp = pd.read_csv(f)
            # Chuẩn hóa tên cột về chữ thường và xóa khoảng trắng thừa
            df_temp.columns = [c.strip().lower() for c in df_temp.columns]
            
            # Map các tên cột tương đương trong file IoT
            rename_map = {}
            for col in df_temp.columns:
                if 'temp' in col: rename_map[col] = 'temp'
                elif col in ['ph', 'ph_value']: rename_map[col] = 'ph'
                elif 'do' in col or 'oxygen' in col: rename_map[col] = 'do'
                elif 'sal' in col or 'cond' in col: rename_map[col] = 'salinity'
                elif 'nh3' in col or 'ammonia' in col: rename_map[col] = 'nh3'
                elif 'no2' in col or 'nitrite' in col: rename_map[col] = 'no2'
            
            df_temp = df_temp.rename(columns=rename_map)
            
            # Giữ lại các cột cần thiết
            req_cols = [c for c in ['temp', 'ph', 'do', 'salinity'] if c in df_temp.columns]
            if len(req_cols) >= 3:
                water_dfs.append(df_temp[req_cols])
        except Exception as e:
            continue

    if not water_dfs:
        print("⚠️ Không tìm thấy file CSV đo đạc, tạo dataset mẫu chuẩn theo ngưỡng sinh học...")
        np.random.seed(42)
        n = 1200
        data = {
            'temp': np.random.uniform(22.0, 36.0, n),
            'ph': np.random.uniform(6.5, 9.5, n),
            'do': np.random.uniform(1.5, 8.0, n),
            'salinity': np.random.uniform(2.0, 35.0, n),
            'nh3': np.random.uniform(0.01, 0.4, n),
            'no2': np.random.uniform(0.01, 0.6, n)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.concat(water_dfs, ignore_index=True).dropna()
        # Bổ sung NH3/NO2 giả định nếu file IoT thiếu 2 chỉ số này
        if 'nh3' not in df.columns:
            df['nh3'] = np.random.uniform(0.01, 0.25, len(df))
        if 'no2' not in df.columns:
            df['no2'] = np.random.uniform(0.01, 0.35, len(df))

    return df

def label_water_quality(row):
    """
    Gán nhãn chuẩn theo quy chuẩn kỹ thuật quốc gia về nước nuôi trồng thủy sản
    0: An toàn (Optimal)
    1: Cảnh báo (Warning)
    2: Nguy hại (Hazardous)
    """
    score = 0
    # Đánh giá pH
    if row['ph'] < 7.0 or row['ph'] > 9.0: score += 2
    elif row['ph'] < 7.5 or row['ph'] > 8.5: score += 1

    # Đánh giá DO
    if row['do'] < 3.0: score += 3
    elif row['do'] < 4.0: score += 1

    # Đánh giá nhiệt độ
    if row['temp'] < 24.0 or row['temp'] > 34.0: score += 2
    elif row['temp'] < 26.0 or row['temp'] > 32.0: score += 1

    # Đánh giá khí độc
    if row.get('nh3', 0) > 0.15: score += 2
    elif row.get('nh3', 0) > 0.1: score += 1

    if row.get('no2', 0) > 0.25: score += 2
    elif row.get('no2', 0) > 0.2: score += 1

    if score >= 4:
        return 2  # Nguy hại
    elif score >= 2:
        return 1  # Cảnh báo
    return 0      # An toàn

def train_model():
    print("🔄 Đang xử lý dữ liệu chất lượng nước...")
    df = load_and_preprocess_water_data()
    
    # Gán nhãn tự động dựa trên quy chuẩn khoa học
    df['water_status'] = df.apply(label_water_quality, axis=1)

    features = ['temp', 'ph', 'do', 'salinity', 'nh3', 'no2']
    X = df[features]
    y = df['water_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🌲 Đang huấn luyện mô hình Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ Độ chính xác (Accuracy): {acc * 100:.2f}%\n")
    print("Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['An toàn (0)', 'Cảnh báo (1)', 'Nguy hại (2)']))

    # Lưu mô hình đã huấn luyện
    os.makedirs("models", exist_ok=True)
    model_path = "models/water_classifier.pkl"
    joblib.dump(rf_model, model_path)
    print(f"💾 Đã lưu mô hình thành công vào: {model_path}")

if __name__ == "__main__":
    train_model()