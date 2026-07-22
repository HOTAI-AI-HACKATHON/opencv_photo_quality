import os
import cv2
import numpy as np

def check_image_quality(image_path, blur_thresh=35.0, min_bright=40, max_bright=220):
    """
    對照片進行前置品質檢查（清晰度、過暗、過曝）
    
    :param image_path: 輸入圖片路徑
    :param blur_thresh: 降噪後的清晰度分數門檻（低於此值代表太模糊）
    :param min_bright: 最低平均亮度 (0~255)
    :param max_bright: 最高平均亮度 (0~255)
    :return: (bool, str, dict) -> (是否通過, 檢測說明, 詳細數據)
    """
    if not os.path.exists(image_path):
        return False, f"❌ 找不到檔案：{image_path}", {}

    img = cv2.imread(image_path)
    if img is None:
        return False, "❌ 無法讀取圖片檔案", {}

    # 轉為灰階圖進行分析
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------- 1. 模糊度檢測 (含高斯降噪) ----------------
    # 先做 3x3 高斯模糊，濾除桌面紋理與高頻雜訊，避免背景紋理騙過檢測
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_score = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()

    # ---------------- 2. 亮度檢測 ----------------
    mean_brightness = np.mean(gray)

    # 收集數據
    metrics = {
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2)
    }

    print("\n---------------- 📊 照片品質分析報告 ----------------")
    print(f"▸ 降噪後清晰度得分 : {blur_score:.2f} (合格標準: ≥ {blur_thresh})")
    print(f"▸ 平均亮度得分     : {mean_brightness:.2f} (合格標準: {min_bright} ~ {max_bright})")
    print("----------------------------------------------------\n")

    # ---------------- 3. 判定邏輯 ----------------
    if blur_score < blur_thresh:
        return False, f"⚠️ 照片過度模糊 (得分: {blur_score:.1f})，請重新對焦並保持手部穩定", metrics

    if mean_brightness < min_bright:
        return False, f"⚠️ 照片太暗 (亮度: {mean_brightness:.1f})，請至光線充足處拍攝", metrics

    if mean_brightness > max_bright:
        return False, f"⚠️ 照片過曝/反光過強 (亮度: {mean_brightness:.1f})，請調整拍攝角度避免光線直射", metrics

    return True, "✅ 照片品質良好，符合拍攝標準", metrics


# ==========================================
# 程式執行進入點
# ==========================================
if __name__ == "__main__":
    # 填入你要測試的照片檔名（需與此 .py 檔在同一資料夾）
    test_image_name = "test4.png" 
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, test_image_name)

    # 執行品質檢測
    is_valid, message, metrics = check_image_quality(
        image_path=image_path,
        blur_thresh=35.0,  # 可依據實測狀況調整 (分數低於 35 判定為太糊)
        min_bright=40,     # 最低亮度
        max_bright=220     # 最高亮度
    )

    print(f"【檢測結果】: {message}")