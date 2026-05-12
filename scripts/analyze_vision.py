import os
import json
from ultralytics import YOLO # 👑 魔法核心：引入免费开源的 YOLO 大脑

# 💡 模拟一个简单的音标数据库（保持原样）
PHONETIC_DB = {
    "apple": "[ˈæpl]",
    "house": "[haʊs]",
    "sunflower": "[ˈsʌnflaʊər]",
    "dog": "[dɔːɡ]",
    "desk": "[desk]",
    "blackboard": "[ˈblækbɔːrd]",
    "chair": "[tʃer]",
    "book": "[bʊk]",
    "bowl": "[boʊl]", # YOLO 默认数据集里有碗，所以加上它
    "table": "[ˈteɪbl]"   # YOLO 把课桌归类为 table
}

def get_phonetic(word):
    # YOLO 返回的单词大部分是复数或带空格，需要处理一下
    clean_word = word.lower().strip()
    # 如果数据库里有就用，没有就留空或使用通用标志
    return PHONETIC_DB.get(clean_word, "[音标待更新]")

def analyze_image_with_free_yolo(image_path):
    """👑 免费大脑：在机器人本地跑 YOLO 模型扫描图片"""
    
    # 1. 加载模型
    # 我们使用 'yolov8n.pt' (n 代表 nano, 代表最轻量、最快、跑在 CPU 上无压力的版本)
    # 机器人会自动去下载这个几兆的模型文件，完全免费
    print("正在加载免费 YOLOv8 模型...")
    model = YOLO('yolov8n.pt') 

    # 2. 运行模型进行物体检测
    print(f"正在分析图片: {image_path}...")
    # conf=0.25 表示置信度大于 25% 的物体才算
    # verbose=False 让它安静一点，别输出一堆日志
    results = model(image_path, conf=0.25, verbose=False) 

    hotspots = []
    
    # YOLO 返回的是一个结果包裹
    result = results[0]

    # 获取图片的长和宽 (为了把像素坐标转成网页用的百分比)
    img_h, img_w = result.orig_shape

    # 循环 YOLO 检测出的每一个物体
    # YOLO 把所有信息（坐标、名字、Mid）都整理得很干净
    # 👑 注意：YOLO 这个免费大脑是通用的，它不仅能检测体素图，还能检测现实世界。
    # 它是基于 COCO 数据集（一个包含 80 种日常物体的免费数据集）训练的。
    # 它的“视网膜”能认出：人、车、狗、桌子、碗、苹果等。
    for box in result.boxes:
        # 获取物体的英文名字
        class_id = int(box.cls[0])
        raw_word = model.names[class_id] # 拿到原始单词

        # 构建音标
        phonetic = get_phonetic(raw_word)

        # 获取检测到的物体在图片里的具体坐标框。
        # 这里我们获取 Google 风格一样的 xywh 坐标 (xy表示左上角点坐标)
        # box.xywh[0] 返回的是 [x, y, w, h] (像素坐标)
        xywh_px = box.xywh[0]
        x_px, y_px, w_px, h_px = xywh_px.tolist()
        
        # 将像素坐标精准转换成本地网页用的百分比
        # 👑 魔法：YOLO 默认返回的是物体的中心点坐标，我们需要把它处理成左上角坐标
        x_min_px = x_px - (w_px / 2)
        y_min_px = y_px - (h_px / 2)

        # 构建适合你网页格式的数据
        hotspots.append({
            # 生成一个简易的唯一 Mid
            "id": f"auto_{raw_word}_{class_id}_{len(hotspots)}", 
            "word": raw_word, # YOLO 检测到的物体名称
            "phonetic": phonetic, # 加上音标
            "top": round((y_min_px / img_h) * 100, 2),   # 转换百分比并保留两位小数
            "left": round((x_min_px / img_w) * 100, 2),
            "width": round((w_px / img_w) * 100, 2),
            "height": round((h_px / img_h) * 100, 2)
        })

    return hotspots

def update_scenes_json(new_image_file):
    """自动把新数据注入你原来的数据仓库 (保持原样)"""
    base_name = os.path.basename(new_image_file)
    scene_id = os.path.splitext(base_name)[0]
    
    image_path_in_repo = f"images/{base_name}"
    
    # 👑 魔法发生的地方：改用免费的 Python 大脑
    print(f"正在智能扫描新场景: {image_path_in_repo}...")
    hotspots_data = analyze_image_with_free_yolo(new_image_file)
    
    if not hotspots_data:
        print("💡 开源模型扫描完毕，但没有检测到清晰的物体。")
        return

    # 读取旧的数据仓库
    try:
        with open('scenes.json', 'r', encoding='utf-8') as f:
            current_data = json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，初始化一个空架构
        current_data = {"activeScene": scene_id, "debugMode": False, "scenes": {}}

    # 把 AI 扫描出来的新场景塞进去
    current_data['scenes'][scene_id] = {
        "id": scene_id,
        "title": f"✨ AIGC智能扫描: {scene_id}", # 临时起个标题
        "imageUrl": image_path_in_repo,
        "instruction": f"AIGC智能扫描完成！画面里共有 {len(hotspots_data)} 个物体，点点看！",
        "hotspots": hotspots_data
    }
    
    # 自动把新场景设置为当前默认场景
    current_data['activeScene'] = scene_id

    # 写回数据仓库
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
        print("💡 暂无新图片可扫描。请在 images/ 下上传体素场景图。")
    else:
        # 假设机器人只处理最新上传的那张图
        newest_image = max(images, key=os.path.getctime)
        update_scenes_json(newest_image)
