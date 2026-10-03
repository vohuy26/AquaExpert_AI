import os
import json
import joblib
import numpy as np

class AquaExpertSystem:
    def __init__(self, rules_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        
        model_path = os.path.join(project_root, "models", "disease_classifier.pkl")
        vec_path = os.path.join(project_root, "models", "tfidf_vectorizer.pkl")
        
        if rules_path is None:
            rules_path = os.path.join(project_root, "data", "rules.json")
        elif not os.path.isabs(rules_path):
            rules_path = os.path.join(project_root, rules_path)
            
        # Nạp mô hình NLP nếu tồn tại
        self.model = joblib.load(model_path) if os.path.exists(model_path) else None
        self.vectorizer = joblib.load(vec_path) if os.path.exists(vec_path) else None
        
        # Nạp cơ sở tri thức luật
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

    def evaluate_water_suitability(self, params):
        """
        Đánh giá mức độ phù hợp của nguồn nước đo đạc đối với cả 5 đối tượng nuôi:
        - 3 loài Tôm: Tôm thẻ chân trắng, Tôm sú, Tôm càng xanh
        - 2 loài Cá: Cá rô phi, Cá tra
        """
        species_ranges = {
            "Tôm thẻ chân trắng": {"temp": (26.0, 32.0), "ph": (7.5, 8.5), "do": 4.0, "sal": (5.0, 30.0), "nh3": 0.1, "no2": 0.2},
            "Tôm sú": {"temp": (28.0, 32.0), "ph": (7.5, 8.5), "do": 3.5, "sal": (15.0, 25.0), "nh3": 0.1, "no2": 0.2},
            "Tôm càng xanh": {"temp": (26.0, 31.0), "ph": (7.0, 8.5), "do": 3.0, "sal": (0.0, 10.0), "nh3": 0.1, "no2": 0.2},
            "Cá rô phi": {"temp": (25.0, 32.0), "ph": (6.5, 8.5), "do": 3.0, "sal": (0.0, 15.0), "nh3": 0.2, "no2": 0.5},
            "Cá tra": {"temp": (26.0, 30.0), "ph": (6.5, 8.0), "do": 2.0, "sal": (0.0, 5.0), "nh3": 0.3, "no2": 0.5}
        }
        
        suitability_results = {}
        t = params.get("temperature", 29.0)
        p = params.get("ph", 7.8)
        d = params.get("do", 5.0)
        s = params.get("salinity", 15.0)
        nh3 = params.get("nh3", 0.02)
        no2 = params.get("no2", 0.05)

        for sp, rng in species_ranges.items():
            score = 0.0
            total_criteria = 6.0
            
            # 1. Nhiệt độ
            if rng["temp"][0] <= t <= rng["temp"][1]: 
                score += 1.0
            elif rng["temp"][0] - 2.0 <= t <= rng["temp"][1] + 2.0: 
                score += 0.5
            
            # 2. Độ pH
            if rng["ph"][0] <= p <= rng["ph"][1]: 
                score += 1.0
            elif rng["ph"][0] - 0.5 <= p <= rng["ph"][1] + 0.5: 
                score += 0.5
            
            # 3. Oxy hòa tan (DO)
            if d >= rng["do"]: 
                score += 1.0
            elif d >= rng["do"] - 1.0: 
                score += 0.5
            
            # 4. Độ mặn (Salinity)
            if rng["sal"][0] <= s <= rng["sal"][1]: 
                score += 1.0
            elif max(0.0, rng["sal"][0] - 3.0) <= s <= rng["sal"][1] + 5.0: 
                score += 0.5
            
            # 5 & 6. Khí độc NH3 và NO2
            score += 1.0 if nh3 <= rng["nh3"] else 0.0
            score += 1.0 if no2 <= rng["no2"] else 0.0
            
            suitability_results[sp] = round((score / total_criteria) * 100, 1)
            
        return suitability_results

    def assess_water_status(self, ph, temp, do, salinity, nh3, no2=0.0):
        """
        Đánh giá trạng thái rủi ro môi trường ao nuôi tổng quát
        """
        issues = []
        status = "An toàn"
        
        # Kiểm tra pH
        if ph < 6.8 or ph > 9.0:
            issues.append(f"Độ pH nguy hiểm ({ph})")
            status = "Nguy hại"
        elif ph < 7.5 or ph > 8.5:
            issues.append(f"Độ pH chưa tối ưu ({ph})")
            if status != "Nguy hại": 
                status = "Cảnh báo"
            
        # Kiểm tra DO
        if do < 3.0:
            issues.append(f"Oxy hòa tan DO rất thấp ({do} mg/L) - Nguy cơ ngạt khí")
            status = "Nguy hại"
        elif do < 4.0:
            issues.append(f"Oxy hòa tan DO thấp ({do} mg/L)")
            if status != "Nguy hại": 
                status = "Cảnh báo"
            
        # Kiểm tra NH3
        if nh3 > 0.15:
            issues.append(f"Khí độc NH3 cao ({nh3} mg/L)")
            status = "Nguy hại"
        elif nh3 > 0.08:
            issues.append(f"Khí độc NH3 tích tụ ({nh3} mg/L)")
            if status != "Nguy hại": 
                status = "Cảnh báo"

        # Kiểm tra NO2
        if no2 > 0.25:
            issues.append(f"Khí độc NO2 vượt ngưỡng ({no2} mg/L)")
            status = "Nguy hại"
        elif no2 > 0.15:
            issues.append(f"Khí độc NO2 cao ({no2} mg/L)")
            if status != "Nguy hại": 
                status = "Cảnh báo"
            
        # Kiểm tra nhiệt độ
        if temp < 22.0 or temp > 34.0:
            issues.append(f"Nhiệt độ nước bất lợi gây sốc nhiệt ({temp}°C)")
            if status != "Nguy hại": 
                status = "Cảnh báo"
            
        return status, issues

    def diagnose(self, species="Tôm thẻ chân trắng", symptom_input="", ph=7.8, temp=29.0, do=5.0, salinity=15.0, nh3=0.02, no2=0.05, weather="Bình thường / Nắng nhẹ", water_state=None, weather_risk=None, **kwargs):
        """
        Động cơ chẩn đoán kết hợp:
        1. Phân tích văn bản qua mô hình Machine Learning NLP (TF-IDF)
        2. Đánh giá điều kiện nước và thời tiết
        3. Suy diễn tiến (Forward Chaining) dựa trên Knowledge Base (rules.json)
        """
        # Hỗ trợ nhận triệu chứng đa dạng (danh sách hoặc chuỗi)
        if isinstance(symptom_input, list):
            symptom_text = " ".join(symptom_input)
        elif "symptoms" in kwargs and isinstance(kwargs["symptoms"], list):
            symptom_text = " ".join(kwargs["symptoms"])
        elif "symptom_text" in kwargs:
            symptom_text = str(kwargs["symptom_text"])
        else:
            symptom_text = str(symptom_input)
            
        full_text = symptom_text.lower()
        inference_trace = []
        top_predictions = []
        primary_disease = None
        base_confidence = 50.0

        # 1. Dự đoán qua Machine Learning NLP (nếu có model và vectorizer)[cite: 1, 6]
        if self.model and self.vectorizer and symptom_text.strip():
            try:
                text_vec = self.vectorizer.transform([symptom_text])
                probabilities = self.model.predict_proba(text_vec)[0]
                classes = self.model.classes_
                top_indices = np.argsort(probabilities)[::-1][:3]
                top_predictions = [
                    {"Tên bệnh": classes[i], "Xác suất (%)": round(float(probabilities[i]) * 100, 2)}
                    for i in top_indices
                ]
                primary_disease = top_predictions[0]["Tên bệnh"]
                base_confidence = top_predictions[0]["Xác suất (%)"]
                inference_trace.append(f"Mô hình Machine Learning NLP: Dự đoán khả nghi nhất là **{primary_disease}** ({base_confidence}%).")
            except Exception:
                pass

        # 2. Đánh giá chất lượng nước & cảnh báo môi trường[cite: 6]
        water_status, water_issues = self.assess_water_status(ph, temp, do, salinity, nh3, no2)
        env_warnings = list(water_issues)
        
        weather_str = str(weather)
        if "Mưa lớn" in weather_str or "Mưa rào" in weather_str:
            env_warnings.append("Mưa lớn kéo dài gây phân tầng nước và tụt pH đáy")
        elif "Nắng nóng" in weather_str or "Nắng gắt" in weather_str:
            env_warnings.append("Nắng nóng gay gắt làm tăng độc tính NH3 và nguy cơ tảo tàn")
        elif "Biến động" in weather_str or "Chuyển mùa" in weather_str:
            env_warnings.append("Biến động nhiệt độ đột ngột dễ gây sốc môi trường và giảm sức đề kháng")

        # Thiết lập trạng thái môi trường nước nếu chưa truyền dạng dict
        if water_state is None:
            water_state = {
                "low_DO": do < 3.5,
                "high_temperature": temp > 31.0,
                "high_NH3_NO2": (nh3 > 0.1 or no2 > 0.2),
                "abnormal_pH": (ph < 7.5 or ph > 8.5),
                "low_salinity": salinity < 5.0,
                "high_salinity": salinity > 25.0,
                "high_organic_matter": water_status in ["Cảnh báo", "Nguy hại"]
            }

        # Thiết lập rủi ro thời tiết nếu chưa truyền dạng dict
        if weather_risk is None:
            weather_risk = {
                "prolonged_heat_or_heavy_rain": any(k in weather_str for k in ["Mưa lớn", "Mưa rào", "Nắng nóng", "Nắng gắt"]),
                "temperature_change": any(k in weather_str for k in ["Biến động", "Chuyển mùa"])
            }

        # 3. Động cơ suy diễn tiến Forward Chaining + Certainty Factor (CF)
        diagnosis_results = []
        for rule in self.rules:
            # Kiểm tra loài nuôi có khớp với luật hay không
            app_species = rule.get("applicable_species", ["Tôm thẻ chân trắng", "Tôm sú"])
            if species not in app_species:
                continue

            cf = 0.0
            evidence_details = []

            # A. Tính trọng số triệu chứng lâm sàng
            symptom_weights = rule.get("symptom_weights", {})
            for sym, weight in symptom_weights.items():
                if sym.lower() in full_text:
                    cf += weight
                    evidence_details.append(f"Triệu chứng lâm sàng: '{sym}' (+{int(weight * 100)}%)")

            # B. Tính trọng số chất lượng nước
            for cond_name, weight in rule.get("water_conditions", {}).items():
                if water_state.get(cond_name, False):
                    cf += weight
                    evidence_details.append(f"Môi trường nước: '{cond_name}' (+{int(weight * 100)}%)")

            # C. Tính trọng số rủi ro thời tiết
            for cond_name, weight in rule.get("weather_conditions", {}).items():
                if weather_risk.get(cond_name, False):
                    cf += weight
                    evidence_details.append(f"Yếu tố thời tiết: '{cond_name}' (+{int(weight * 100)}%)")

            cf = min(cf, 1.0)
            threshold = rule.get("base_threshold", 0.35)

            # Đạt ngưỡng kích hoạt
            if cf >= threshold:
                risk_level = "CAO" if cf >= 0.70 else ("TRUNG BÌNH" if cf >= 0.50 else "THẤP")
                diagnosis_results.append({
                    "disease_code": rule["disease"],
                    "disease_vietnamese": rule.get("disease_vi", rule["disease"]),
                    "confidence": round(cf * 100, 1),
                    "risk_level": risk_level,
                    "evidence": evidence_details,
                    "recommendation": rule.get("treatment", "")
                })

        # Sắp xếp các bệnh có độ tin cậy cao nhất lên đầu
        diagnosis_results.sort(key=lambda x: x["confidence"], reverse=True)

        # 4. Tổng hợp kết quả chẩn đoán chính
        if diagnosis_results:
            top_rule = diagnosis_results[0]
            pred_code = top_rule["disease_code"]
            pred_vi = top_rule["disease_vietnamese"]
            final_cf = top_rule["confidence"]
            final_treatment = top_rule["recommendation"]
            for ev in top_rule["evidence"]:
                inference_trace.append(f"Luật kích hoạt [{pred_code}]: {ev}")
        else:
            pred_code = primary_disease if primary_disease else "Không xác định"
            pred_vi = primary_disease if primary_disease else "Chưa phát hiện nguy cơ bệnh rõ ràng"
            final_cf = base_confidence if primary_disease else 0.0
            final_treatment = "Tiếp tục theo dõi các chỉ số môi trường, duy trì quạt khí tạo oxy hòa tan ổn định."

        # Trả về kết quả hoàn chỉnh dưới dạng Dictionary (tương thích app.py và test_inference.py)
        result_dict = {
            "predicted_disease": pred_code,
            "disease_vi": pred_vi,
            "confidence": final_cf,
            "water_status": water_status,
            "env_warnings": env_warnings,
            "treatment": final_treatment,
            "top_predictions": top_predictions,
            "inference_trace": inference_trace,
            "all_matched_rules": diagnosis_results
        }
        return result_dict

# Định nghĩa Alias tên lớp để tương thích hoàn toàn
AquaInferenceEngine = AquaExpertSystem