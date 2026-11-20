# -*- coding: utf-8 -*-
"""
web/app.py - FLASK APPLICATION: HỆ THỐNG NHẬN DIỆN MÓN ĂN ĐẶC TRƯNG MIỀN BẮC VIỆT NAM
Hỗ trợ giao diện Web tương tác cao, thiết kế theo chuẩn bản vẽ Figma/UI.
"""

import sys
import json
import uuid
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Thêm thư mục gốc vào PYTHONPATH để nạp các module từ src/
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import DishPredictor

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mon-an-mien-bac-secret-key-2024-kdd'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Tối đa 10 MB
app.config['UPLOAD_FOLDER'] = BASE_DIR / "web" / "static" / "uploads"
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Nạp dữ liệu dishes_info.json
DISHES_INFO_PATH = BASE_DIR / "dishes_info.json"
dishes_data = {}
if DISHES_INFO_PATH.exists():
    with open(DISHES_INFO_PATH, 'r', encoding='utf-8') as f:
        all_dishes_data = json.load(f)
    dataset_dir = BASE_DIR / "dataset" / "Images"
    if dataset_dir.is_dir():
        dataset_classes = {path.name for path in dataset_dir.iterdir() if path.is_dir()}
        dishes_data = {
            name: details for name, details in all_dishes_data.items()
            if name in dataset_classes
        }
    else:
        dishes_data = all_dishes_data

WATER_DISHES = {'Pho', 'Bun thang', 'Bun rieu', 'Banh da cua', 'Chao long'}
CAKE_DISHES = {'Banh cuon', 'Banh chung', 'Banh gio', 'Banh duc', 'Banh gai', 'Banh mi'}
RICE_DISHES = {'Xoi xeo', 'Com rang dua bo'}
GRILLED_DISHES = {'Bun cha', 'Cha ca La Vong', 'Banh tom Ho Tay', 'Bun dau mam tom', 'Nem chua'}
DISH_COUNTS = {
    'all': len(dishes_data),
    'nuoc': sum(name in WATER_DISHES for name in dishes_data),
    'banh': sum(name in CAKE_DISHES for name in dishes_data),
    'xoi': sum(name in RICE_DISHES for name in dishes_data),
    'chien': sum(name in GRILLED_DISHES for name in dishes_data),
}
DISH_IMAGE_COUNT = sum(
    1 for path in (BASE_DIR / "dataset" / "Images").rglob("*")
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
)

# Khởi tạo mô hình dự đoán duy nhất một lần khi máy chủ khởi động
logger.info("Đang khởi tạo bộ suy luận DishPredictor...")
predictor = DishPredictor(
    models_dir=BASE_DIR / "models",
    splits_dir=BASE_DIR / "data" / "splits",
    dishes_info_path=DISHES_INFO_PATH,
    preferred_model="SoftVoting"
)


def allowed_file(filename: str) -> bool:
    """Kiểm tra phần mở rộng tệp hợp lệ."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================================================================
# CÁC ROUTE GIAO DIỆN CHÍNH
# ==============================================================================

@app.route('/')
def index():
    """Trang chủ (Landing Page): Hero section, upload tương tác và 4 món tiêu biểu."""
    # Lấy 4 món tiêu biểu hiển thị trên trang chủ
    sample_keys = ['Pho', 'Bun cha', 'Banh cuon', 'Cha ca La Vong']
    dishes_sample = {k: dishes_data[k] for k in sample_keys if k in dishes_data}
    if not dishes_sample and dishes_data:
        # Nếu không có đúng key trên, lấy 4 món đầu tiên
        dishes_sample = dict(list(dishes_data.items())[:4])
        
    return render_template(
        'index.html',
        dishes_sample=dishes_sample,
        total_dishes=len(dishes_data),
    )


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    API tiếp nhận ảnh tải lên từ FormData, chạy qua toàn bộ Pipeline:
    Tiền xử lý -> Trích đặc trưng -> Mô hình Machine Learning -> Trả về JSON kết quả.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Chưa chọn tệp ảnh để nhận diện.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tên tệp không hợp lệ.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'error': 'Định dạng tệp không được hỗ trợ. Vui lòng tải lên ảnh JPG, PNG hoặc WebP.'
        }), 400
        
    try:
        # Đọc dữ liệu bytes từ file tải lên
        file_bytes = file.read()
        
        # Xác thực ảnh
        is_valid, msg = predictor.validate_image(file_bytes)
        if not is_valid:
            return jsonify({'success': False, 'error': msg}), 400
            
        # Thực hiện suy luận qua pipeline
        result = predictor.predict(file_bytes)
        if not result.get('success'):
            status_code = 503 if predictor.is_demo_mode else 422
            return jsonify(result), status_code

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        save_path = app.config['UPLOAD_FOLDER'] / unique_name
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
        result['image_url'] = url_for('static', filename=f"uploads/{unique_name}")
        
        return jsonify(result)
    except Exception as error:
        logger.exception("Lỗi khi xử lý endpoint /predict")
        return jsonify({'success': False, 'error': f'Lỗi hệ thống: {error}'}), 500


@app.route('/dishes')
def dishes():
    """Trang danh mục các lớp món ăn trong dataset."""
    return render_template(
        'dishes.html', dishes=dishes_data, dish_counts=DISH_COUNTS
    )


@app.route('/dish/<dish_name>')
def dish_detail(dish_name: str):
    """Trang chi tiết văn hóa ẩm thực, nguồn gốc, nguyên liệu và quán ăn nổi tiếng của 1 món."""
    dish = None
    target_key = None
    
    # 1. Tìm chính xác theo key
    if dish_name in dishes_data:
        dish = dishes_data[dish_name]
        target_key = dish_name
    else:
        # 2. Tìm theo tên chuẩn hóa gạch dưới
        norm_search = dish_name.lower().replace(" ", "_")
        for k, v in dishes_data.items():
            if k.lower().replace(" ", "_") == norm_search:
                dish = v
                target_key = k
                break
                
    if not dish:
        flash(f"Không tìm thấy thông tin cho món: {dish_name}", "warning")
        return redirect(url_for('dishes'))
        
    return render_template(
        'dish_detail.html',
        dish=dish,
        dish_key=target_key,
        total_dishes=len(dishes_data),
    )


@app.route('/about')
def about():
    """Trang giới thiệu đề tài, dữ liệu và pipeline SVM + Random Forest."""
    return render_template(
        'about.html',
        total_dishes=len(dishes_data),
        total_images=DISH_IMAGE_COUNT,
    )


# ==============================================================================
# BỘ XỬ LÝ LỖI TRANG (ERROR HANDLERS)
# ==============================================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({
        'success': False,
        'error': 'Dung lượng tệp vượt quá giới hạn tối đa cho phép (10MB).'
    }), 413


if __name__ == '__main__':
    logger.info("Đang khởi động máy chủ Web MónBắc AI tại http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=True)
