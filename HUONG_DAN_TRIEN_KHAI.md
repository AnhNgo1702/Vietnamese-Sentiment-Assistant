# VII. HƯỚNG DẪN TRIỂN KHAI

## 1. Giải pháp NLP

Kiến trúc sử dụng **Transformer pre-trained** (`phobert-base-v2` hoặc `distilbert-base-multilingual-cased`) qua pipeline `sentiment-analysis` của Hugging Face. Không cần fine-tuning, tập trung vào tích hợp đơn giản.

### Kiến trúc tổng quát (Flowchart)

```
[Đầu vào: Câu tiếng Việt]
    ↓ (Preprocessing)
[Component 1: Tiền xử lý] → Câu đã chuẩn hóa
    ↓ (Sentiment Analysis)
[Component 2: Phân loại cảm xúc] → Nhãn (POSITIVE, NEUTRAL, NEGATIVE)
    ↓ (Validation)
[Component 3: Hợp nhất & xử lý lỗi] → Đầu ra dictionary hoặc lỗi
    ↓
[Core Engine: Lưu & hiển thị]
```

## 2. Hướng dẫn chi tiết và tối ưu hóa vấn đề kỹ thuật

### Bước 1: Tiền xử lý

**Mục đích:** Chuẩn hóa câu tiếng Việt để phù hợp với Transformer.

**Hướng dẫn thực hiện:**
- **(Optional)** Dùng `underthesea.word_tokenize()` để tách từ (VD: "Rất vui hôm nay" → "Rất vui hôm nay").
- Chuyển chữ thường, thay "rất" → "rất" bằng từ điển nhỏ (10–20 từ phổ biến: rất/rất, dở/dở, v.v.).
- Giữ hầu hết chữ cái, loại bỏ ký tự dễ giảm thời gian xử lý.

**Implementation:** `src/sentiment_analyzer.py` - method `preprocess()`

### Bước 2: Phân loại cảm xúc

**Mục đích:** Xác định nhãn cảm xúc (POSITIVE, NEUTRAL, NEGATIVE).

**Hướng dẫn thực hiện:**
- Sử dụng pipeline `sentiment-analysis` với model `phobert-base-v2` (ưu tiên tiếng Việt) hoặc `distilbert-base-multilingual-cased` (hỗ trợ đa ngôn ngữ).
- Gửi câu chuẩn hóa qua pipeline, lấy nhãn có xác suất cao nhất.
- **Nếu xác suất <0.5**, trả về NEUTRAL mặc định.

**Implementation:** `src/sentiment_analyzer.py` - method `load_model()` và `analyze()`

### Bước 3: Hợp nhất & xử lý lỗi

**Mục đích:** Ghép kết quả thành dictionary, kiểm tra hợp lệ.

**Hướng dẫn thực hiện:**
- Tạo dictionary: `{"text": "Hôm nay tôi rất vui", "sentiment": "POSITIVE"}`.
- **Kiểm tra:** Câu nhập ≥5 ký tự; nếu rỗng hoặc lỗi pipeline, hiển thị pop-up "Câu không hợp lệ, thử lại!".
- Lưu vào SQLite và hiển thị trên giao diện.

**Implementation:** `src/sentiment_analyzer.py` - method `analyze()` và `src/main.py` - method `classify_and_save()`

## 3. Cấu trúc file code

```
src/
├── sentiment_analyzer.py    # Component 1, 2, 3 - Phân tích cảm xúc
├── database.py              # Lưu trữ SQLite
├── main.py                  # Core Engine - Giao diện và điều phối
├── test_10_cases.py         # Test với 10 câu chuẩn
└── demo_10_cases.py         # Demo nhanh format output
```

## 4. Output Format

Mỗi kết quả phân loại trả về dictionary:

```json
{
  "text": "Hôm nay tôi rất vui",
  "sentiment": "POSITIVE"
}
```

Với 3 giá trị sentiment:
- `POSITIVE` - Tích cực
- `NEGATIVE` - Tiêu cực  
- `NEUTRAL` - Trung tính

## 5. Test Cases (10 câu)

| STT | Đầu vào (Câu tiếng Việt) | Đầu ra mong đợi (Dictionary) |
|-----|--------------------------|------------------------------|
| 1 | Hôm nay tôi rất vui | `{"text": "Hôm nay tôi rất vui", "sentiment": "POSITIVE"}` |
| 2 | Món ăn này dở quá | `{"text": "Món ăn này dở quá", "sentiment": "NEGATIVE"}` |
| 3 | Thời tiết bình thường | `{"text": "Thời tiết bình thường", "sentiment": "NEUTRAL"}` |
| 4 | Rất vui hôm nay | `{"text": "Rất vui hôm nay", "sentiment": "POSITIVE"}` |
| 5 | Công việc ổn định | `{"text": "Công việc ổn định", "sentiment": "NEUTRAL"}` |
| 6 | Phim này hay lắm | `{"text": "Phim này hay lắm", "sentiment": "POSITIVE"}` |
| 7 | Tôi buồn vì thất bại | `{"text": "Tôi buồn vì thất bại", "sentiment": "NEGATIVE"}` |
| 8 | Ngày mai đi học | `{"text": "Ngày mai đi học", "sentiment": "NEUTRAL"}` |
| 9 | Cảm ơn bạn rất nhiều | `{"text": "Cảm ơn bạn rất nhiều", "sentiment": "POSITIVE"}` |
| 10 | Mệt mỏi quá hôm nay | `{"text": "Mệt mỏi quá hôm nay", "sentiment": "NEGATIVE"}` |

## 6. Chạy ứng dụng

### Demo nhanh (không cần model)
```bash
source venv/bin/activate
python src/demo_10_cases.py
```

### Test với model thật
```bash
source venv/bin/activate
python src/test_10_cases.py
```

### Chạy ứng dụng chính
```bash
source venv/bin/activate
python src/main.py
```

## 7. Lưu trữ và hiển thị

- **Lưu trữ:** SQLite database tại `data/sentiment_history.db`
- **Hiển thị:** 
  - Console output với emoji
  - Lịch sử: gõ `history`
  - Thống kê: gõ `stats`
  - Kết quả JSON: `data/test_results.json`
