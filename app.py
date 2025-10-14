import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

# Import hàm load_model từ script gốc của bạn
from main_script import load_model 

# --- Tối ưu hóa: Tải mô hình một lần duy nhất ---
@st.cache_resource
def get_model():
    model_path = 'fasterrcnn_phone_defect.pth' # Đảm bảo file model ở cùng thư mục
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("✅ Model loaded and cached.")
    return model, device

# --- Hàm dự đoán cho Web App (Đã cập nhật logic trả về) ---
def predict_for_webapp(model, device, image_pil, score_thresh=0.8):
    """
    Hàm này nhận ảnh PIL, dự đoán và trả về:
    - detection_status: Một chuỗi ("VỠ", "KHÔNG VỠ", "KHÔNG TÌM THẤY").
    - result_image_pil: Ảnh PIL đã vẽ khung.
    """
    transform = T.ToTensor()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]
    
    image_with_boxes = image_pil.copy() 
    draw = ImageDraw.Draw(image_with_boxes)
    label_map = {1: "KHÔNG VỠ", 2: "VỠ"}
    
    is_defective = False
    detected_object = False # Thêm cờ để kiểm tra có phát hiện đối tượng nào không

    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score > score_thresh:
            detected_object = True # Nếu có ít nhất 1 đối tượng, đặt cờ là True
            box = box.cpu().numpy()
            label_id = label.cpu().numpy().item()
            
            if label_id == 2: 
                is_defective = True
            
            color = "lime" if label_id == 1 else "red"
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
            
            text = f"{label_map.get(label_id, 'N/A')}: {score:.2f}"
            
            text_x = box[0]
            text_y = max(0, box[1] - 20)
            bbox = draw.textbbox((text_x, text_y), text)
            draw.rectangle(bbox, fill="black")
            draw.text((text_x, text_y), text, fill="yellow")

    # --- LOGIC TRẢ VỀ MỚI ---
    if not detected_object:
        return "KHÔNG TÌM THẤY", image_pil # Trả về ảnh gốc nếu không tìm thấy gì
    elif is_defective:
        return "VỠ", image_with_boxes
    else:
        return "KHÔNG VỠ", image_with_boxes

# --- Bắt đầu xây dựng giao diện (Đã cập nhật logic hiển thị) ---
st.set_page_config(layout="wide", page_title="Phone Defect Detection")

st.title("Ứng dụng Phát hiện Lỗi Màn hình Điện thoại")
st.write("Tải lên một ảnh điện thoại để mô hình dự đoán các vết nứt, vỡ.")

model, device = get_model()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

with col2:
    st.write("### Kết quả dự đoán")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner('Đang xử lý...'):
            # Chạy dự đoán và nhận trạng thái
            detection_status, result_image = predict_for_webapp(model, device, image, score_thresh=0.8)
            
            # --- LOGIC HIỂN THỊ MỚI ---
            if detection_status == "VỠ":
                st.error("❌ **KẾT QUẢ:\nFaster R-CNN: PHÁT HIỆN VỠ**")
                st.image(result_image, caption="Ảnh Kết Quả", use_container_width=True)
            
            elif detection_status == "KHÔNG VỠ":
                st.success("✅ **KẾT QUẢ:\nFaster R-CNN: KHÔNG VỠ**")
                st.image(result_image, caption="Ảnh Kết Quả", use_container_width=True)
            
            else: # Trường hợp "KHÔNG TÌM THẤY"
                st.warning("⚠️ **Không phát hiện thấy điện thoại trong ảnh.**")
                st.write("Vui lòng thử lại với một ảnh khác rõ ràng hơn hoặc đảm bảo ảnh có chứa điện thoại.")
                st.image(result_image, caption="Ảnh đã tải lên", use_container_width=True)
    else:
        st.info("Hãy tải một ảnh lên để xem kết quả.")