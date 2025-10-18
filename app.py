import streamlit as st
import pickle
import numpy as np
import os
import torch
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image, ImageDraw, ImageFont

# LƯU Ý: Đảm bảo bạn có file 'main_script.py' trong cùng thư mục này
# File này phải chứa hàm 'load_model' mà 'app (1).py' của bạn sử dụng.
try:
    from main_script import load_model
except ImportError:
    st.error("LỖI CRITICAL: Không tìm thấy file 'main_script.py'.")
    st.error("Vui lòng tạo file 'main_script.py' chứa hàm 'load_model' để tải mô hình Faster R-CNN.")
    st.stop()


# =======================
# 1. Cấu hình chung
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================
# 2. Mô hình 1: PHÂN LOẠI (ViT + Softmax)
# ============================================

@st.cache_resource
def load_classification_models():
    """Tải mô hình ViT và mô hình Softmax đã huấn luyện."""
    st.info("Đang tải mô hình Phân loại (ViT + Softmax)...")
    
    # 1. Tải mô hình ViT
    try:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        transform_for_vit = weights.transforms()
        vit_model = vit_b_16(weights=weights).to(DEVICE)
    except Exception:
        st.warning("Không thể tải weights mới của ViT, thử phương pháp cũ.")
        vit_model = vit_b_16(pretrained=True).to(DEVICE)
        transform_for_vit = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.456, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if hasattr(vit_model, "heads"):
        vit_model.heads.head = torch.nn.Identity()
    else:
        vit_model.head = torch.nn.Identity()
    vit_model.eval()

    # 2. Tải mô hình Softmax
    MODEL_PATH = os.path.join(SCRIPT_DIR, "softmax_model.pkl")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Lỗi: Không tìm thấy 'softmax_model.pkl'.")
        return None, None, None, None, None, None, None

    # Tải file .pkl (chỉ chứa W, b, label_map)
    with open(MODEL_PATH, "rb") as f:
        softmax_model = pickle.load(f)

    # --- SỬA LỖI KEYERROR BẰNG CÁCH GÁN GIÁ TRỊ GIẢ ---
    # Chúng ta bỏ qua bước chuẩn hóa vì file .pkl không có 'mean' và 'std'
    # CẢNH BÁO: Điều này sẽ làm cho dự đoán BỊ SAI
    st.warning("Cảnh báo: Không tìm thấy 'mean' và 'std' trong file model. Bỏ qua bước chuẩn hóa. Kết quả phân loại (Mô hình 1) sẽ KHÔNG chính xác.")
    feature_mean = 0.0  # Gán giá trị giả
    feature_std = 1.0   # Gán giá trị giả (để phép chia không bị lỗi)
    # --- KẾT THÚC SỬA LỖI ---

    W, b = softmax_model["W"], softmax_model["b"]
    original_label_map = softmax_model["label_map"]
    label_map = {v: k for k, v in original_label_map.items()}
    
    st.success("Tải xong mô hình Phân loại.")
    return vit_model, transform_for_vit, W, b, feature_mean, feature_std, label_map
# --- KẾT THÚC HÀM ---


def extract_vit_features(pil_image, vit, transform, device):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_t = transform(pil_image)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        features = vit(batch_t)
    return features.cpu().numpy()

def softmax(z):
    if z.ndim == 1: z = z.reshape(1, -1)
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict_classification(X_feature, W, b):
    scores = X_feature @ W + b
    probs = softmax(scores)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ================================================
# 3. Mô hình 2: PHÁT HIỆN LỖI (Faster R-CNN)
# ================================================

@st.cache_resource
def load_detection_model():
    """Tải mô hình Faster R-CNN."""
    st.info("Đang tải mô hình Phát hiện lỗi (Faster R-CNN)...")
    model_path = os.path.join(SCRIPT_DIR, 'fasterrcnn_phone_defect.pth')
    if not os.path.exists(model_path):
        st.error(f"Lỗi: Không tìm thấy 'fasterrcnn_phone_defect.pth' tại '{model_path}'.")
        return None, None
        
    model = load_model(model_path) # <-- Sử dụng hàm từ 'main_script.py'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    st.success("Tải xong mô hình Phát hiện lỗi.")
    return model, device

def predict_for_webapp(model, device, image_pil, score_thresh=0.6):
    """Dự đoán và trả về trạng thái + ảnh đã vẽ."""
    transform = T.ToTensor()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]
    
    image_with_boxes = image_pil.copy() 
    draw = ImageDraw.Draw(image_with_boxes)
    label_map = {1: "KHÔNG VỠ", 2: "VỠ"}
    
    is_defective = False
    detected_object = False 

    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score > score_thresh:
            detected_object = True 
            box = box.cpu().numpy()
            label_id = label.cpu().numpy().item()
            
            if label_id == 2: 
                is_defective = True
            
            color = "lime" if label_id == 1 else "red"
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
            
            text = f"{label_map.get(label_id, 'N/A')}: {score:.2f}"
            text_x, text_y = box[0], max(0, box[1] - 20)
            bbox = draw.textbbox((text_x, text_y), text)
            draw.rectangle(bbox, fill="black")
            draw.text((text_x, text_y), text, fill="yellow")

    if not detected_object:
        return "KHÔNG TÌM THẤY", image_pil
    elif is_defective:
        return "VỠ", image_with_boxes
    else:
        return "KHÔNG VỠ", image_with_boxes


# ===================================
# 4. GIAO DIỆN STREAMLIT GỘP
# ===================================
st.set_page_config(layout="wide", page_title="Phân tích Điện thoại")
st.title("Ứng dụng Phân tích Lỗi Màn hình Điện thoại (Gộp 2 Mô hình)")
st.write("Tải lên một ảnh để chạy đồng thời mô hình Phân loại (ViT) và Phát hiện lỗi (Faster R-CNN).")
st.divider()

# --- Tải tất cả mô hình ---
class_models = load_classification_models()
vit_model, transform_for_vit, W, b, feature_mean, feature_std, label_map = class_models
detect_model, detect_device = load_detection_model()

# --- Kiểm tra nếu tải mô hình thất bại ---
if any(m is None for m in class_models) or detect_model is None:
    st.error("Một hoặc cả hai mô hình đã không thể tải. Vui lòng kiểm tra lại file và đường dẫn.")
    st.stop()

# --- Giao diện tải file ---
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh 1 lần duy nhất
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    # Hiển thị ảnh gốc
    st.image(image_pil, caption="Ảnh bạn đã tải lên", use_column_width=False, width=400)
    st.divider()

    # Tạo 2 cột cho 2 kết quả
    col1, col2 = st.columns(2)

    # --- CỘT 1: KẾT QUẢ PHÂN LOẠI (ViT) ---
    with col1:
        st.header("1. Kết quả Phân loại (Tổng thể)")
        with st.spinner('Đang phân loại...'):
            # 1. Trích xuất đặc trưng ViT
            vit_features = extract_vit_features(image_pil, vit_model, transform_for_vit, DEVICE)
            
            # 2. Chuẩn hóa (BƯỚC NÀY SẼ BỊ BỎ QUA VÌ MEAN=0, STD=1)
            standardized_features = (vit_features - feature_mean) / feature_std
            
            # 3. Dự đoán (SẼ BỊ SAI)
            pred_class, probs = predict_classification(standardized_features, W, b)
            
            predicted_label_index = int(pred_class[0])
            predicted_label_name = label_map[predicted_label_index]

            if predicted_label_name == "defective":
                st.error(f"👉 Dự đoán: {predicted_label_name.upper()} (Lỗi)")
            elif predicted_label_name == "non-defective":
                st.success(f"👉 Dự đoán: {predicted_label_name.upper()} (Không lỗi)")
            else:
                st.warning(f"👉 Dự đoán: {predicted_label_name.upper()}")

            st.subheader("📊 Xác suất (ViT):")
            for i, p in enumerate(probs[0]):
                class_name = label_map[i]
                st.write(f"- **{class_name}**: `{p:.4f}`")

    # --- CỘT 2: KẾT QUẢ PHÁT HIỆN LỖI (FASTER R-CNN) ---
    with col2:
        st.header("2. Kết quả Phát hiện (Chi tiết)")
        with st.spinner('Đang phát hiện lỗi...'):
            # Chạy dự đoán và nhận trạng thái
            # Chuyển .copy() để mô hình này vẽ lên ảnh
            detection_status, result_image = predict_for_webapp(
                detect_model, 
                detect_device, 
                image_pil.copy(), 
                score_thresh=0.5
            )
            
            if detection_status == "VỠ":
                st.error("❌ **KẾT QUẢ:\nPhát hiện VỠ**")
            elif detection_status == "KHÔNG VỠ":
                st.success("✅ **KẾT QUẢ:\nKhông phát hiện vỡ**")
            else: # "KHÔNG TÌM THẤY"
                st.warning("⚠️ **Không phát hiện thấy điện thoại trong ảnh.**")
            
            st.image(result_image, caption="Ảnh đã vẽ khung", use_column_width=True)

else:
    st.info("Hãy tải một ảnh lên để xem kết quả từ cả hai mô hình.")
