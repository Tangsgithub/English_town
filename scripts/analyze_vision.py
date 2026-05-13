import os
import json
from ultralytics import YOLO # 👑 魔法核心：引入免费开源的 YOLO 大脑

# 💡 简易音标数据库，机器人会自动匹配音标
PHONETIC_DB = {
    "apple": "[ˈæpl]",
    "house": "[haʊs]",
    "sunflower": "[ˈsʌnflaʊər]",
    "dog": "[dɔːɡ]",
    "desk": "[desk]",
    "blackboard": "[ˈblækbɔːrd]",
    "chair": "[tʃer]",
    "book": "[bʊk]",
    "bowl": "[boʊl]", # YOLO COCO数据集里有bowl
    "table": "[ˈteɪbl]"   # YOLO COCO把课桌归类为table
}

def get_phonetic(word):
    clean_word = word.lower().strip()
    return PHONETIC_DB.get(clean_word, "[音标待更新]")

def analyze_image_with_free_yolo(image_path):
    """👑 免费大脑：运行本地 YOLOv8 模型扫描图片"""
    
    # 1. 加载模型 (n 代表 nano, 代表最轻量、在 CPU 上跑无压力的版本)
    print("正在加载免费 YOLOv8 模型...")
    model = YOLO('yolov8n.pt') 

    # 2. 运行模型
    print(f"正在分析图片: {image_path}...")
    results = model(image_path, conf=0.25, verbose=False) 

    hotspots = []
    result = results[0]
    img_h, img_w = result.orig_shape

    # 循环检测出的每一个物体
    # 通用COCO数据集能认出80种日常物体
    for box in result.boxes:
        # 获取物体的英文名字
        class_id = int(box.cls[0])
        raw_word = model.names[class_id] # 拿到单词标签

        phonetic = get_phonetic(raw_word)

        # 获取检测到的像素坐标 [x_center, y_center, w, h]
        xywh_px = box.xywh[0]
        x_px, y_px, w_px, h_px = xywh_px.tolist()
        
        # 将像素坐标转换成本地网页用的百分比
        # 这里把中心点转换成本地需要的左上角坐标百分比
        x_min_px = x_px - (w_px / 2)
        y_min_px = y_px - (h_px / 2)

        hotspots.append({
            # 生成简易的唯一 Mid
            "id": f"auto_{raw_word}_{class_id}_{len(hotspots)}", 
            "word": raw_word, # YOLO检测到的标签
            "phonetic": phonetic, # 匹配出的红色音标
            "top": round((y_min_px / img_h) * 100, 2),   # 转换百分比并保留两位小数
            "left": round((x_min_px / img_w) * 100, 2),
            "width": round((w_px / img_w) * 100, 2),
            "height": round((h_px / img_h) * 100, 2)
        })

    return hotspots

def update_scenes_json(new_image_file):
    """自动把新数据注入你根目录下的 scenes.json 数据仓库"""
    base_name = os.path.basename(new_image_file)
    scene_id = os.path.splitext(base_name)[0]
    
    # 图片在网页上的引用路径
    image_path_in_repo = f"images/{base_name}"
    
    # 👑 魔法核心：调用免费开源大脑
    print(f"正在智能扫描场景: {image_path_in_repo}...")
    hotspots_data = analyze_image_with_free_yolo(new_image_file)
    
    if not hotspots_data:
        print("💡 开源模型扫描完毕，但没有在该画面检测到清晰的常见物体。")
        return

    # 读取旧的数据仓库
    try:
        with open('scenes.json', 'r', encoding='utf-8') as f:
            current_data = json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，初始化全空架构
        current_data = {"activeScene": scene_id, "debugMode": False, "scenes": {}}

    # 把新场景数据塞进去
    current_data['scenes'][scene_id] = {
        "id": scene_id,
        "title": f"✨ AIGC智能扫描: {scene_id}", # 临时标题，可在JSON里手动改
        "imageUrl": image_path_in_repo,
        "instruction": f"AIGC智能扫描完成！画面共有 {len(hotspots_data)} 个物体，点点看！",
        "hotspots": hotspots_data
    }
    
    # 自动把新场景设置为默认激活的
    current_data['activeScene'] = scene_id

    # 全自动写回根目录的数据仓库
    with open('scenes.json', 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ scenes.json 已成功自动更新！检测到 {len(hotspots_data)} 个热点，成本: $0。")

if __name__ == "__main__":
    IMAGE_FOLDER = 'images'
    
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    # 找到 images 文件夹下最新上传的图片
    images = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print("💡 暂无图片。请将风景图传到 images/ 下。")
    else:
        # 假设机器人只处理最新的一张
        newest_image = max(images, key=os.path.getctime)
        update_scenes_json(newest_image)
