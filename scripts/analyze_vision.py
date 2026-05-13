import os
import json
from ultralytics import YOLO 

# 👑 完整版 COCO 数据集 80 类物体标准美语音标库
PHONETIC_DB = {
    "person": "[ˈpɜːrsn]", "bicycle": "[ˈbaɪsɪkl]", "car": "[kɑːr]", "motorcycle": "[ˈmoʊtərsaɪkl]",
    "airplane": "[ˈerpleɪn]", "bus": "[bʌs]", "train": "[treɪn]", "truck": "[trʌk]", "boat": "[boʊt]",
    "traffic light": "[ˈtræfɪk laɪt]", "fire hydrant": "[ˈfaɪər ˈhaɪdrənt]", "stop sign": "[stɑːp saɪn]",
    "parking meter": "[ˈpɑːrkɪŋ ˈmiːtər]", "bench": "[bentʃ]", "bird": "[bɜːrd]", "cat": "[kæt]",
    "dog": "[dɔːɡ]", "horse": "[hɔːrs]", "sheep": "[ʃiːp]", "cow": "[kaʊ]", "elephant": "[ˈelɪfənt]",
    "bear": "[ber]", "zebra": "[ˈziːbrə]", "giraffe": "[dʒəˈræf]", "backpack": "[ˈbækpæk]",
    "umbrella": "[ʌmˈbrelə]", "handbag": "[ˈhændbæɡ]", "tie": "[taɪ]", "suitcase": "[ˈsuːtkeɪs]",
    "frisbee": "[ˈfrɪzbi]", "skis": "[skiːz]", "snowboard": "[ˈsnoʊbɔːrd]", "sports ball": "[spɔːrts bɔːl]",
    "kite": "[kaɪt]", "baseball bat": "[ˈbeɪsbɔːl bæt]", "baseball glove": "[ˈbeɪsbɔːl ɡlʌv]",
    "skateboard": "[ˈskeɪtbɔːrd]", "surfboard": "[ˈsɜːrfbɔːrd]", "tennis racket": "[ˈtenɪs ˈrækɪt]",
    "bottle": "[ˈbɑːtl]", "wine glass": "[waɪn ɡlæs]", "cup": "[kʌp]", "fork": "[fɔːrk]", "knife": "[naɪf]",
    "spoon": "[spuːn]", "bowl": "[boʊl]", "banana": "[bəˈnænə]", "apple": "[ˈæpl]", "sandwich": "[ˈsændwɪtʃ]",
    "orange": "[ˈɔːrɪndʒ]", "broccoli": "[ˈbrɑːkəli]", "carrot": "[ˈkærət]", "hot dog": "[hɑːt dɔːɡ]",
    "pizza": "[ˈpiːtsə]", "donut": "[ˈdoʊnʌt]", "cake": "[keɪk]", "chair": "[tʃer]", "couch": "[kaʊtʃ]",
    "potted plant": "[ˈpɑːtɪd plænt]", "bed": "[bed]", "dining table": "[ˈdaɪnɪŋ ˈteɪbl]",
    "toilet": "[ˈtɔɪlət]", "tv": "[tiː ˈviː]", "laptop": "[ˈlæptɑːp]", "mouse": "[maʊs]",
    "remote": "[rɪˈmoʊt]", "keyboard": "[ˈkiːbɔːrd]", "cell phone": "[sel foʊn]", "microwave": "[ˈmaɪkrəweɪv]",
    "oven": "[ˈʌvn]", "toaster": "[ˈtoʊstər]", "sink": "[sɪŋk]", "refrigerator": "[rɪˈfrɪdʒəreɪtər]",
    "book": "[bʊk]", "clock": "[klɑːk]", "vase": "[veɪs]", "scissors": "[ˈsɪzərz]",
    "teddy bear": "[ˈtedi ber]", "hair drier": "[her ˈdraɪər]", "toothbrush": "[ˈtuːθbrʌʃ]",
    # 额外补充一些变体
    "table": "[ˈteɪbl]", "desk": "[desk]", "monitor": "[ˈmɑːnɪtər]"
}

def get_phonetic(word):
    # 强制转小写并去除首尾空格，确保绝对匹配
    clean_word = word.lower().strip()
    return PHONETIC_DB.get(clean_word, "[音标待更新]")

def analyze_image_with_free_yolo(image_path):
    print("正在加载免费 YOLOv8 模型...")
    model = YOLO('yolov8n.pt') 

    print(f"正在分析图片: {image_path}...")
    results = model(image_path, conf=0.25, verbose=False) 

    hotspots = []
    result = results[0]
    img_h, img_w = result.orig_shape

    for box in result.boxes:
        class_id = int(box.cls[0])
        # 将 YOLO 返回的单词首字母大写，为了网页显示好看 (如 dog 变成 Dog)
        raw_word = model.names[class_id].capitalize() 

        phonetic = get_phonetic(raw_word)

        # 获取精确的中心点像素坐标和宽高
        xywh_px = box.xywh[0]
        x_px, y_px, w_px, h_px = xywh_px.tolist()
        
        # 精确转换为左上角坐标，并保留更高的精度（3位小数）防止网页缩放偏移
        x_min_px = x_px - (w_px / 2)
        y_min_px = y_px - (h_px / 2)

        hotspots.append({
            "id": f"auto_{raw_word}_{class_id}_{len(hotspots)}", 
            "word": raw_word, 
            "phonetic": phonetic, 
            "top": round((y_min_px / img_h) * 100, 3),   
            "left": round((x_min_px / img_w) * 100, 3),
            "width": round((w_px / img_w) * 100, 3),
            "height": round((h_px / img_h) * 100, 3)
        })

    return hotspots

def update_scenes_json(new_image_file):
    base_name = os.path.basename(new_image_file)
    scene_id = os.path.splitext(base_name)[0]
    image_path_in_repo = f"images/{base_name}"
    
    print(f"正在智能扫描场景: {image_path_in_repo}...")
    hotspots_data = analyze_image_with_free_yolo(new_image_file)
    
    if not hotspots_data:
        print("💡 扫描完毕，未检测到常见物体。")
        return

    try:
        with open('scenes.json', 'r', encoding='utf-8') as f:
            current_data = json.load(f)
    except FileNotFoundError:
        # 默认开启 debugMode 为 False
        current_data = {"activeScene": scene_id, "debugMode": False, "scenes": {}}

    current_data['scenes'][scene_id] = {
        "id": scene_id,
        "title": f"🗺️ 场景: {scene_id}", 
        "imageUrl": image_path_in_repo,
        "instruction": f"✨ 魔法扫描完成！画面中发现了 {len(hotspots_data)} 个物体，点点看！",
        "hotspots": hotspots_data
    }
    
    current_data['activeScene'] = scene_id

    with open('scenes.json', 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ scenes.json 自动更新成功！")

if __name__ == "__main__":
    IMAGE_FOLDER = 'images'
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    images = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if images:
        newest_image = max(images, key=os.path.getctime)
        update_scenes_json(newest_image)
