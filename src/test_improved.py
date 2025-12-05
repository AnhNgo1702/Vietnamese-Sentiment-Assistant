"""
Test cải tiến
"""

from sentiment_analyzer import SentimentAnalyzer

def test_improved():
    """Test với từ khóa tiếng Việt"""
    print("\n" + "="*80)
    print("TEST CẢI TIẾN - KEYWORD BOOSTING")
    print("="*80 + "\n")
    
    analyzer = SentimentAnalyzer()
    
    test_cases = [
        ('Hôm nay tôi rất vui', 'POSITIVE'),
        ('Món ăn này dở quá', 'NEGATIVE'),
        ('Thời tiết bình thường', 'NEUTRAL'),
        ('Rất vui hôm nay', 'POSITIVE'),
        ('Công việc ổn định', 'NEUTRAL'),
        ('Phim này hay lắm', 'POSITIVE'),
        ('Tôi buồn vì thất bại', 'NEGATIVE'),
        ('Ngày mai đi học', 'NEUTRAL'),
        ('Cảm ơn bạn rất nhiều', 'POSITIVE'),
        ('Mệt mỏi quá hôm nay', 'NEGATIVE'),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for text, expected in test_cases:
        result = analyzer.analyze(text)
        actual = result['sentiment']
        confidence = result.get('confidence', 0.0)
        
        is_correct = (actual == expected)
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Câu: {text:<35} | Expected: {expected:<10} | Got: {actual:<10} | Conf: {confidence:.2%}")
    
    print("\n" + "="*80)
    print(f"KẾT QUẢ: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_improved()
