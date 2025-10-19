def predict_for_webapp(model, device, image_pil, score_thresh=0.6):
    """
    Hàm dự đoán cho webapp — mô hình 3 lớp:
    1: non_defective_phone
    2: defective
    3: non-phone
    """
    transform = T.ToTensor()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    image_with_boxes = image_pil.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # --- Gắn nhãn ---
    label_map = {
        1: "Non-defective Phone",
        2: "Defective",
        3: "Non-phone"
    }

    detected_object = False
    detected_labels = []

    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score > score_thresh:
            detected_object = True
            label_id = int(label.cpu().item())
            detected_labels.append(label_id)

            box = box.cpu().numpy()
            color = {1: "lime", 2: "red", 3: "orange"}.get(label_id, "white")

            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
            text = f"{label_map[label_id]}: {score:.2f}"

            text_x, text_y = box[0], max(0, box[1] - 20)
            bbox = draw.textbbox((text_x, text_y), text)
            draw.rectangle(bbox, fill="black")
            draw.text((text_x, text_y), text, fill="yellow")

    # --- Quy tắc quyết định kết quả ---
    if not detected_object:
        detection_status = "NO_OBJECT"
    elif 3 in detected_labels:
        detection_status = "NON_PHONE"
    elif 2 in detected_labels:
        detection_status = "DEFECTIVE"
    elif 1 in detected_labels:
        detection_status = "NON_DEFECTIVE"
    else:
        detection_status = "UNKNOWN"

    return detection_status, image_with_boxes
