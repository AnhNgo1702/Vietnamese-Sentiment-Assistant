"""
Test với test case yêu cầu
"""

from sentiment_analyzer import SentimentAnalyzer
from database import SentimentDatabase
import json

def test_10_cases():
    """Test 10 câu test case"""
    print("=" * 80)
    print("VIII. BỘ TEST CASE (10 CÂU)")
    print("=" * 80)
    
    analyzer = SentimentAnalyzer()
    db = SentimentDatabase("data/test_10_cases.db")
    
    # 10 test cases theo yêu cầu
    test_cases = [
        {"stt": 1, "text": "Hôm nay tôi rất vui", "expected": "POSITIVE"},
        {"stt": 2, "text": "Món ăn này dở quá", "expected": "NEGATIVE"},
        {"stt": 3, "text": "Thời tiết bình thường", "expected": "NEUTRAL"},
        {"stt": 4, "text": "Rất vui hôm nay", "expected": "POSITIVE"},
        {"stt": 5, "text": "Công việc ổn định", "expected": "NEUTRAL"},
        {"stt": 6, "text": "Phim này hay lắm", "expected": "POSITIVE"},
        {"stt": 7, "text": "Tôi buồn vì thất bại", "expected": "NEGATIVE"},
        {"stt": 8, "text": "Ngày mai đi học", "expected": "NEUTRAL"},
        {"stt": 9, "text": "Cảm ơn bạn rất nhiều", "expected": "POSITIVE"},
        {"stt": 10, "text": "Mệt mỏi quá hôm nay", "expected": "NEGATIVE"},
    ]
    
    print()
    print(f"{'STT':<5} {'Đầu vào (Câu tiếng Việt)':<35} {'Kết quả':<15} {'Mong đợi':<15} {'Độ tin cậy':<12} {'Trạng thái'}")
    print("-" * 100)
    
    correct = 0
    total = len(test_cases)
    
    # Test từng câu
    for test in test_cases:
        stt = test["stt"]
        text = test["text"]
        expected = test["expected"]
        
        # Phân tích
        result = analyzer.analyze(text)
        
        # Lấy sentiment và confidence
        actual = result.get('sentiment', 'UNKNOWN')
        confidence = result.get('confidence', 0.0)
        
        # Kiểm tra đúng/sai
        is_correct = (actual == expected)
        status = "✓ ĐÚNG" if is_correct else "✗ SAI"
        
        if is_correct:
            correct += 1
        
        # Lưu vào database
        db.save_classification(text, actual, confidence)
        
        # Hiển thị kết quả
        print(f"{stt:<5} {text:<35} {actual:<15} {expected:<15} {confidence*100:>6.2f}%     {status}")
        
        # Hiển thị dictionary format
        output_dict = {"text": text, "sentiment": actual}
        print(f"      → {json.dumps(output_dict, ensure_ascii=False)}")
        print()
    
    print("-" * 100)
    print(f"\n◆ KẾT QUẢ TỔNG QUAN:")
    print(f"   Tổng số test: {total}")
    print(f"   Đúng: {correct} ({correct/total*100:.1f}%)")
    print(f"   Sai: {total - correct} ({(total-correct)/total*100:.1f}%)")
    
    # Hiển thị thống kê từ database
    print(f"\n◆ THỐNG KÊ DATABASE:")
    stats = db.get_statistics()
    print(f"   Tổng: {stats['total']}")
    print(f"   Tích cực: {stats['positive']}")
    print(f"   Trung tính: {stats['neutral']}")
    print(f"   Tiêu cực: {stats['negative']}")
    
    print("\n" + "=" * 80)
    if correct == total:
        print("◈ TẤT CẢ TEST CASE ĐỀU PASS!")
    else:
        print(f"◇ CÓ {total - correct} TEST CASE KHÔNG KHỚP KỲ VỌNG")
    print("=" * 80 + "\n")
    
    return correct, total

if __name__ == "__main__":
    try:
        test_10_cases()
    except Exception as e:
        print(f"\nX Lỗi: {e}\n")
        import traceback
        traceback.print_exc()
