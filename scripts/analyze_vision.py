import os
import json
from ultralytics import YOLO 

# 💡 扩充并优化了音标数据库（全小写匹配，防止大小写导致找不到音标）
PHONETIC_DB = {
    "apple": "[ˈæpl]",
    "tree": "[triː]",
    "house": "[haʊs]",
    "sunflower": "[ˈsʌnflaʊər]",
    "dog": "[dɔːɡ]",
    "cat": "[kæt]",
    "desk": "[desk]",
    "blackboard": "[ˈblækbɔːrd]",
    "chair": "[tʃer]",
    "book": "[bʊk]",
    "bowl": "[boʊl]", 
    "table": "[ˈteɪbl]",
    "potted plant": "[ˈpɑːtɪd plænt]",
    "person": "[ˈpɜːrsn]"
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
