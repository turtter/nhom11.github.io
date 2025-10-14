import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet_50_FPN_Weights
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
import numpy as np
import os
import csv

# ==============================
# 1️⃣ Load model (Giữ nguyên)
# ==============================
# Lưu ý: Khi dùng Streamlit, bạn sẽ thêm @st.cache_resource vào trước hàm này
def load_model(weight_path="fasterrcnn_phone_defect.pth"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet_50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

# ==============================
# 2️⃣ Crop mở rộng 5% (Giữ nguyên)
# ==============================
def crop_with_margin(img, box, margin_ratio=0.05):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_w * margin_ratio))
    y1 = max(0, int(y1 - box_h * margin_ratio))
    x2 = min(w, int(x2 + box_w * margin_ratio))
    y2 = min(h, int(y2 + box_h * margin_ratio))
    cropped = img[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)

# ======================================================
# 3️⃣ ✨ Tách hàm dự đoán và lưu kết quả
# ======================================================
def process_image(model, image_path, device, score_thresh=0.8):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = T.ToTensor()
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]
    
    # --- CÁC THƯ MỤC LƯU TRỮ ---
    output_folders = ["detec", "non_detec", "train/defective", "train/non_defective", "results_visualized"]
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)
    
    # --- GHI LOG CSV ---
    log_path = os.path.join("train", "log_results.csv")
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["image", "class", "score", "crop_path"])

        # --- HIỂN THỊ VÀ LƯU ---
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_rgb)
        
        detected = False
        label_map = {1: "KHÔNG_VỠ", 2: "VỠ"}

        for i, (box, score, label) in enumerate(zip(outputs["boxes"].cpu().numpy(), outputs["scores"].cpu().numpy(), outputs["labels"].cpu().numpy())):
            if score < score_thresh:
                continue

            detected = True
            
            # Crop và resize
            cropped, expanded_box = crop_with_margin(img_rgb, box)
            resized = cv2.resize(cropped, (126, 224), interpolation=cv2.INTER_AREA)

            # Vẽ bounding box
            x1, y1, x2, y2 = expanded_box
            color = "lime" if label == 1 else "red"
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            text = f"{label_map.get(label, 'unknown')} ({score:.2f})"
            ax.text(x1, y1 - 10, text, color="yellow", fontsize=10, backgroundcolor="black")
            
            # Lưu các vùng crop
            filename_base = os.path.splitext(os.path.basename(image_path))[0]
            class_name = label_map.get(label)
            crop_filename = f"{filename_base}_{i}_{class_name}_{score:.2f}.jpg"
            
            # Lưu vào detec/non_detec
            class_folder = "non_detec" if label == 1 else "detec"
            crop_path = os.path.join(class_folder, crop_filename)
            cv2.imwrite(crop_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            
            # Lưu vào train/
            train_class_folder = "non_defective" if label == 1 else "defective"
            train_crop_path = os.path.join("train", train_class_folder, crop_filename)
            cv2.imwrite(train_crop_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            
            # Ghi log
            writer.writerow([image_path, class_name, f"{score:.2f}", train_crop_path])

        if detected:
            plt.axis("off")
            # ✨ THAY ĐỔI QUAN TRỌNG: Lưu ảnh thay vì show()
            visualized_path = os.path.join("results_visualized", os.path.basename(image_path))
            fig.savefig(visualized_path, bbox_inches='tight', pad_inches=0)
            print(f"🖼️  Đã lưu ảnh kết quả vào: {visualized_path}")
        else:
            print(f"⚠️ Không phát hiện vật thể nào trong ảnh: {image_path}")
        
        # Đóng plot để giải phóng bộ nhớ
        plt.close(fig)

# ==============================
# 4️⃣ Xử lý ảnh hoặc thư mục
# ==============================
def main():
    model_path = 'fasterrcnn_phone_defect.pth' # Giả sử file model nằm cùng thư mục
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model đã được tải và chạy trên {device.type.upper()}")

    path = input("Nhập đường dẫn ảnh hoặc thư mục cần test: ").strip()

    if os.path.isdir(path):
        imgs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for img_path in imgs:
            print("-" * 50)
            print(f"🔍 Đang xử lý: {img_path}")
            process_image(model, img_path, device, score_thresh=0.8)
    elif os.path.isfile(path):
        process_image(model, path, device, score_thresh=0.8)
    else:
        print(f"❌ Đường dẫn không hợp lệ: {path}")

if __name__ == "__main__":
    main()