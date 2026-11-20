# Nhận diện món ăn miền Bắc

## Pipeline chính

Ứng dụng web dùng pipeline MobileNetV2 + HSV làm pipeline chính:

1. Resize có padding, hiệu chỉnh sáng CLAHE.
2. Ghép embedding 1.280 chiều từ MobileNetV2 ImageNet với histogram HSV 96 chiều.
3. Fit `StandardScaler` và PCA 95% chỉ trên tập train; web nạp cùng extractor và pipeline đã huấn luyện.
4. So sánh SVM, Random Forest và SoftVoting trên cùng tập test; dùng `class_weight="balanced"`.

HSV + HOG + LBP + GLCM vẫn được trích xuất làm baseline truyền thống và được so sánh với pipeline chính trên cùng tập test. Các kết quả được ghi vào `reports/model_comparison.csv` và `reports/ablation_results.csv`.

Các lớp được lấy từ thư mục con hiện có trong `dataset/Images`. Bộ dữ liệu hiện có 17 lớp; lớp Bánh mì không có trong repo và không được huấn luyện. Ảnh tải lên thử nghiệm trong `web/static/uploads` và các artifact huấn luyện không được đưa vào Git.

## Cài đặt và chạy

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_production_model.py
python notebooks\04_modeling.py
python notebooks\05_evaluation.py
python web\app.py
```

Huấn luyện tải pretrained weights MobileNetV2 ImageNet nếu chưa có trong cache, rồi tạo `models/`, `data/features/`, `data/splits/` và các báo cáo CSV trong `reports/`. Các thư mục này được bỏ qua bởi Git. Trên máy mới, hãy chạy huấn luyện trước khi dùng chức năng nhận diện; nếu chưa có model, web vẫn mở được nhưng API trả lỗi dịch vụ chưa sẵn sàng thay vì trả kết quả giả.

## Thực nghiệm và biểu đồ

`notebooks/04_modeling.py` chạy GridSearchCV phân tầng 3-fold trên dữ liệu train để đo ảnh hưởng của `C` và `gamma`. `notebooks/05_evaluation.py` chỉ vẽ từ các metric và ablation đã đo trên tập test, không chứa điểm số nhập tay. Chạy lại huấn luyện sẽ ghi đè các kết quả hiện tại.

## Dữ liệu

Ảnh dataset được giữ trong repo hiện tại để có thể chạy lại thực nghiệm. Nếu cần giảm đáng kể dung lượng clone, cần chuyển dataset sang kho lưu trữ ngoài hoặc Git LFS; xóa file ở commit mới không làm nhỏ lịch sử Git đã có.

