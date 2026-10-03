import pandas as pd
from core.inference_engine import AquaInferenceEngine

def run_tests():
    engine = AquaInferenceEngine(rules_path="data/rules.json")
    df_cases = pd.read_csv("data/diagnosis_cases.csv")
    
    correct = 0
    total = len(df_cases)
    
    print("=" * 75)
    print(f"{'MÃ CA':<8}{'LOÀI':<20}{'BỆNH KỲ VỌNG':<25}{'CHẨN ĐOÁN':<25}{'KẾT QUẢ'}")
    print("=" * 75)

    for _, row in df_cases.iterrows():
        species = row['species']
        symptoms = [row['symptoms'].lower()]
        
        water_state = {
            "low_DO": row['water_status'] in ["Nguy hại", "Cảnh báo"],
            "high_NH3_NO2": row['water_status'] == "Nguy hại",
            "high_temperature": "Nắng nóng" in str(row['weather']),
            "abnormal_pH": row['water_status'] == "Nguy hại",
            "high_organic_matter": row['water_status'] in ["Nguy hại", "Cảnh báo"]
        }
        
        weather_risk = {
            "prolonged_heat_or_heavy_rain": any(w in str(row['weather']) for w in ["Mưa lớn", "Nắng nóng"]),
            "temperature_change": "Biến động" in str(row['weather'])
        }

        # Gọi hàm chẩn đoán
        res = engine.diagnose(
            species=species,
            symptom_input=symptoms,
            water_state=water_state,
            weather_risk=weather_risk
        )
        
        predicted_disease = res["predicted_disease"]
        
        is_correct = (predicted_disease == row['expected_disease'])
        if is_correct:
            correct += 1
            status_text = "✅ ĐÚNG"
        else:
            status_text = "❌ SAI"

        print(f"{row['case_id']:<8}{species:<20}{row['expected_disease']:<25}{predicted_disease:<25}{status_text}")

    print("=" * 75)
    acc = (correct / total) * 100
    print(f"🎯 KẾT QUẢ KIỂM THỬ: {correct}/{total} ca đúng -> Độ chính xác: {acc:.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()