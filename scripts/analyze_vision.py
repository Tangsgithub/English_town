import os
import json
from ultralytics import YOLO # 👑 魔法核心：引入免费开源的 YOLO 大脑

# 💡 模拟一个简单的音标数据库（保持原样，机器人会从这里找音标）
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
    """👑 免费大脑：在机器人本地电脑上现场跑 YOLO 模型扫描图片"""
    
    # 1. 加载轻量级模型 (n 代表 nano, 代表最快、跑在 CPU 上无压力的版本)
    print("正在加载免费 YOLOv8 模型...")
    model = YOLO('yolov8n.pt') 

    # 2. 运行模型进行物体检测
    print(f"正在智能分析图片: {image_path}...")
    # conf=0.25 表示置信度大于 25% 的物体才算检测成功
    # verbose=False 让它安静一点，别输出一堆垃圾日志
    results = model(image_path, conf=0.25, verbose=False) 

    hotspots = []
    
    # YOLO 返回的是一个结果包裹
    result = results[0]

    # 获取图片的长和宽 (为了把像素坐标转成网页用的百分比)
    img_h, img_w = result.orig_shape

    # 循环 YOLO 检测出的每一个物体
    # YOLO 这个通用大脑不仅能认出体素图，还能认出：人、车、狗、桌子、碗、苹果等日常物体（基于 COCO 数据集）。
    for box in result.boxes:
        # 获取物体的英文名字
        class_id = int(box.cls[0])
        raw_word = model.names[class_id] # 拿到原始单词标签

        # 构建音标
        phonetic = get_phonetic(raw_word)

        # 获取物体在图片里的具体坐标框。
        # YOLO 默认返回的是物体的中心点坐标 [x_center, y_center, w, h]
        xywh_px = box.xywh[0]
        x_px, y_px, w_px, h_px = xywh_px.tolist()
        
        # 👑 核心魔法：将 YOLO 中心点坐标处理成 CSS 用的左上角坐标百分比
        x_min_px = x_px - (w_px / 2)
        y_min_px = y_px - (h_px / 2)

        # 构建适合你网页加载的 JSON 数据格式
        hotspots.append({
            # 生成一个简易的唯一 Mid
            "id": f"auto_{raw_word}_{class_id}_{len(hotspots)}", 
            "word": raw_word, # YOLO 检测到的物体名称
            "phonetic": phonetic, # 加上红色音标
            "top": round((y_min_px / img_h) * 100, 2),   # 转换百分比并保留两位小数
            "left": round((x_min_px / img_w) * 100, 2),
            "width": round((w_px / img_w) * 100, 2),
            "height": round((h_px / img_h) * 100, 2)
        })

    return hotspots

def update_scenes_json(new_image_file):
    """自动把扫描到的新 AIGC 数据注入你根目录下的 scenes.json (数据仓库)"""
    base_name = os.path.basename(new_image_file)
    scene_id = os.path.splitext(base_name)[0]
    
    # 图片在网页上的引用路径
    image_path_in_repo = f"images/{base_name}"
    
    # 👑 魔法发生的地方：改用免费的开源 Python 大脑
    print(f"正在智能扫描新场景: {image_path_in_repo}...")
    hotspots_data = analyze_image_with_free_yolo(new_image_file)
    
    if not hotspots_data:
        print("💡 开源模型扫描完毕，但没有在该画面检测到清晰的常见物体。")
        return

    # 读取旧的数据仓库
    try:
        with open('scenes.json', 'r', encoding='utf-8') as f:
            current_data = json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，初始化一个全空架构
        current_data = {"activeScene": scene_id, "debugMode": False, "scenes": {}}

    # 把 AI 扫描出来的新场景塞进去
    current_data['scenes'][scene_id] = {
        "id": scene_id,
        "title": f"✨ AIGC智能扫描: {scene_id}", # 临时起个标题，以后你可以随时改 JSON 把这里改好看点
        "imageUrl": image_path_in_repo,
        "instruction": f"AIGC智能扫描完成！画面里共有 {len(hotspots_data)} 个物体，点点看！",
        "hotspots": hotspots_data
    }
    
    # 自动把新场景设置为默认激活的场景（给用户呈现最新的）
    current_data['activeScene'] = scene_id

    # 全自动写回根目录的数据仓库
    with open('scenes.json', 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ scenes.json 已成功自动更新！AIGC 数据已注入。检测到 {len(hotspots_data)} 个热点，成本: $0。")

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
