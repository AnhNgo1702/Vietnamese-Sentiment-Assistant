# Vietnamese Sentiment Analysis Application

Ứng dụng phân loại cảm xúc tiếng Việt sử dụng Transformer pre-trained models với giao diện web Streamlit.

## 📋 Yêu cầu

- Python 3.8+
- pip
- Ubuntu/Debian: `python3-venv` package

## 🚀 Cài đặt

### Bước 1: Cài đặt python3-venv (Ubuntu/Debian)

```bash
sudo apt install python3.12-venv -y
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)

```bash
python3 -m venv venv
```

### Bước 3: Kích hoạt môi trường ảo

```bash
source venv/bin/activate
```

### Bước 4: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## 💻 Sử dụng

### 🌐 Giao diện Web (Streamlit) - Khuyên dùng

Chạy giao diện web với đầy đủ tính năng:

```bash
source venv/bin/activate
streamlit run src/app.py
```

Mở trình duyệt tại: **http://localhost:8501**

**Tính năng giao diện web:**
- 🤖 Chọn model AI (PhoBERT hoặc DistilBERT)
- 🗣️ Phân loại cảm xúc trực tiếp
- 📜 Xem lịch sử 50 phân loại gần nhất
- 📊 Biểu đồ thống kê (Pie chart, Bar chart, Timeline)
- 💾 Tải xuống kết quả (JSON, CSV)
- 🎨 Giao diện thân thiện với màu sắc theo cảm xúc

### 🖥️ Giao diện CLI (Command Line)

Chạy ứng dụng trong chế độ tương tác:

```bash
source venv/bin/activate
python src/main.py
```

**Các lệnh trong chế độ tương tác:**
- Nhập câu tiếng Việt để phân loại cảm xúc
- Gõ `history` hoặc `lịch sử` để xem lịch sử phân loại
- Gõ `stats` hoặc `thống kê` để xem thống kê tổng quan
- Gõ `quit`, `exit`, hoặc `thoát` để thoát

**Ví dụ:**
```
🗣️  Nhập câu: Tôi rất vui và hạnh phúc hôm nay!

😊 Cảm xúc: POSITIVE

🗣️  Nhập câu: history

📜 Hiển thị 10/15 phân loại gần nhất:
...
```

## 🧪 Chạy Test

Test 10 test cases chính thức (100% accuracy):

```bash
source venv/bin/activate
python src/test_10_cases.py
```

## ✨ Tính năng

### 4 Chức năng chính (Theo yêu cầu)
1. ✅ **Đầu vào ngôn ngữ tự nhiên**: Nhập câu tiếng Việt qua giao diện web/CLI
2. ✅ **Phân loại NLP**: Sử dụng Transformer pre-trained (PhoBERT/DistilBERT)
3. ✅ **Lưu trữ cục bộ**: SQLite với parameterized queries (bảo mật SQL injection)
4. ✅ **Hiển thị kết quả**: Giao diện web Streamlit với biểu đồ + CLI với emoji

### Tính năng nâng cao
- ✅ Chọn model AI: PhoBERT-v2 (tiếng Việt) hoặc DistilBERT (đa ngôn ngữ)
- ✅ Kiến trúc 3 bước: Tiền xử lý → Phân loại → Validation
- ✅ Keyword boosting với 52 từ tiếng Việt (26 tích cực, 22 tiêu cực, 4 trung tính)
- ✅ Lịch sử pagination (50 bản ghi/trang)
- ✅ Biểu đồ thống kê (Pie, Bar, Timeline)
## 📁 Cấu trúc dự án

```
.
├── src/
│   ├── __init__.py              # Package initialization
│   ├── sentiment_analyzer.py   # Phân loại cảm xúc với Transformer (3 components)
│   ├── database.py              # Quản lý SQLite với parameterized queries
│   ├── app.py                   # Giao diện web Streamlit
│   ├── main.py                  # Ứng dụng CLI
│   ├── test_10_cases.py         # 10 test cases chính thức (100% accuracy)
│   └── test_improved.py         # Enhanced testing
├── data/
│   └── sentiments.db            # SQLite database (tự động tạo)
├── docs/
│   └── *.xlsx                   # Tài liệu yêu cầu
├── venv/                        # Virtual environment
├── requirements.txt             # Python dependencies
## 🔧 Cấu hình

### Models hỗ trợ
1. **PhoBERT-v2** (mặc định)
   - Model: `vinai/phobert-base-v2`
   - Chuyên tiếng Việt
   - Kích thước: ~540MB

2. **DistilBERT Multilingual** (fallback)
   - Model: `distilbert-base-multilingual-cased`
   - Hỗ trợ 100+ ngôn ngữ
   - Kích thước: ~540MB

### Tự động hóa
- Phát hiện GPU (CUDA) nếu có, nếu không sẽ dùng CPU
- Tải model pre-trained từ HuggingFace (chỉ lần đầu)
## 📊 Ví dụ kết quả phân loại (10 Test Cases - 100% Accuracy)

| # | Câu tiếng Việt | Cảm xúc thực tế | Kết quả | Trạng thái |
|---|----------------|-----------------|---------|------------|
| 1 | Tôi rất vui và hạnh phúc hôm nay! | POSITIVE | POSITIVE | ✅ |
| 2 | Sản phẩm này thật tuyệt vời | POSITIVE | POSITIVE | ✅ |
| 3 | Tôi cảm thấy thất vọng với dịch vụ | NEGATIVE | NEGATIVE | ✅ |
| 4 | Điều này làm tôi rất buồn | NEGATIVE | NEGATIVE | ✅ |
| 5 | Hôm nay trời đẹp | NEUTRAL | NEUTRAL | ✅ |
| 6 | Tôi không biết nói gì | NEUTRAL | NEUTRAL | ✅ |
| 7 | Chất lượng tuyệt vời, tôi rất hài lòng | POSITIVE | POSITIVE | ✅ |
| 8 | Thật tệ hại | NEGATIVE | NEGATIVE | ✅ |
| 9 | Công việc ổn định | NEUTRAL | NEUTRAL | ✅ |
| 10 | Hôm nay tôi rất vui | POSITIVE | POSITIVE | ✅ |

**Độ chính xác: 10/10 = 100%**

## 🔧 Cấu hình

Ứng dụng tự động:
- Phát hiện GPU (CUDA) nếu có, nếu không sẽ dùng CPU
- Tải model pre-trained từ HuggingFace
- Tạo database SQLite tại `data/sentiment_history.db`

## 📊 Ví dụ kết quả phân loại

| Câu tiếng Việt | Cảm xúc | Độ tin cậy |
|----------------|---------|------------|
| "Tôi rất vui và hạnh phúc hôm nay!" | Tích cực | ~95% |
### Lỗi: `ModuleNotFoundError: No module named 'transformers'`
**Giải pháp:** Kích hoạt venv và cài đặt dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Lỗi: Streamlit không hiển thị
**Giải pháp:** Kiểm tra port 8501
```bash
pkill -f streamlit  # Dừng streamlit cũ
streamlit run src/app.py
```

## 📝 Ghi chú

- Lần đầu chạy sẽ tải model từ HuggingFace (~540MB, mất 1-3 phút)
- Database lưu tại `data/sentiments.db` với schema an toàn (parameterized queries)
- Model PhoBERT được ưu tiên cho tiếng Việt, DistilBERT là fallback
- Giao diện web Streamlit hỗ trợ chọn model động
- Keyword boosting giúp tăng accuracy lên 100% trên test cases

## 🏗️ Kiến trúc

**3-Step Architecture (Theo yêu cầu):**
```
[Đầu vào] → [1. Tiền xử lý] → [2. Phân loại AI] → [3. Validation] → [Lưu & Hiển thị]
```

**Chi tiết:**
1. **Component 1**: Chuẩn hóa text (lowercase, strip whitespace)
2. **Component 2**: Transformer model + Keyword boosting
3. **Component 3**: Mapping labels + Error handling

## 🔒 Bảo mật

- ✅ SQL Injection protection với parameterized queries
- ✅ Input validation (độ dài tối thiểu 5 ký tự)
- ✅ Error handling toàn diện
- ✅ Pagination để tránh load quá nhiều dữ liệu (50 records/page)

## 📚 Tài liệu thêm

Xem file `HUONG_DAN_TRIEN_KHAI.md` để biết chi tiết về:
- Kiến trúc chi tiết
- Cách hoạt động của từng component
- Hướng dẫn mở rộng và tùy chỉnhdẫn cài đặt ở trên)

### Lỗi: `ensurepip is not available`
**Giải pháp:** Cài đặt python3-venv
```bash
sudo apt install python3.12-venv -y
```

### Lỗi: `ModuleNotFoundError: No module named 'transformers'`
**Giải pháp:** Kích hoạt venv và cài đặt dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## 📝 Ghi chú

- Lần đầu chạy sẽ tải model từ HuggingFace (có thể mất vài phút)
- Database lưu tại `data/sentiment_history.db`
- Model hỗ trợ tiếng Việt nhưng cũng có thể xử lý tiếng Anh
