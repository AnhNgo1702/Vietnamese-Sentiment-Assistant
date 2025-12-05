from sentiment_analyzer import SentimentAnalyzer
from database import SentimentDatabase
from typing import Dict
import sys

class SentimentApp:
    """Ứng dụng chính phân loại cảm xúc"""
    
    def __init__(self):
        """Khởi tạo ứng dụng"""
        print("=== Ứng dụng Phân loại Cảm xúc Tiếng Việt ===\n")
        self.analyzer = SentimentAnalyzer()
        self.db = SentimentDatabase()
    
    def classify_and_save(self, text: str) -> Dict:
        """
        Bước 3: Hợp nhất & xử lý lỗi
        
        Phân loại cảm xúc và lưu vào database
        - Tạo dictionary: {text, sentiment}
        - Kiểm tra: Câu nhập ≥5 ký tự; nếu rỗng hoặc lỗi pipeline, 
          hiển thị pop-up "Câu không hợp lệ, thử lại!".
        - Lưu vào SQLite và hiển thị trên giao diện.
        
        Args:
            text: Câu văn cần phân loại
            
        Returns:
            Kết quả phân loại theo format {text, sentiment}
        """
        # Phân tích cảm xúc
        result = self.analyzer.analyze(text)
        
        # Kiểm tra lỗi
        if 'error' in result:
            print(f"\n⚠️  {result['error']}\n")
            return result
        
        # Lưu vào database (Core Engine: Lưu & hiển thị)
        self.db.save_classification(
            text=result['text'],
            label=result['sentiment']
        )
        
        return result
    
    def show_history(self, limit: int = 50):
        """
        Hiển thị lịch sử phân loại (giới hạn 50 bản ghi mới nhất)
        
        Args:
            limit: Số lượng bản ghi hiển thị (mặc định 50)
        """
        history = self.db.get_history(limit)
        
        if not history:
            print("\n📝 Chưa có lịch sử phân loại nào.\n")
            return
        
        total_count = self.db.get_total_count()
        print(f"\n📜 Hiển thị {len(history)}/{total_count} phân loại gần nhất:\n")
        print("-" * 80)
        
        for record in history:
            id_val, text, label, timestamp = record
            print(f"ID: {id_val} | {timestamp}")
            print(f"Câu: {text}")
            print(f"Cảm xúc: {label}")
            print("-" * 80)
        
        if total_count > limit:
            print(f"\n💡 Còn {total_count - limit} bản ghi nữa. Dùng giao diện web để xem thêm.\n")
    
    def show_statistics(self):
        """Hiển thị thống kê"""
        stats = self.db.get_statistics()
        
        print("\n📊 Thống kê tổng quan:")
        print(f"  Tổng số phân loại: {stats['total']}")
        print(f"  Tích cực: {stats['positive']}")
        print(f"  Trung tính: {stats['neutral']}")
        print(f"  Tiêu cực: {stats['negative']}\n")
    
    def run_interactive(self):
        """Chạy chế độ tương tác"""
        print("\n💡 Hướng dẫn:")
        print("  - Nhập câu tiếng Việt để phân loại cảm xúc")
        print("  - Gõ 'history' để xem lịch sử")
        print("  - Gõ 'stats' để xem thống kê")
        print("  - Gõ 'quit' hoặc 'exit' để thoát\n")
        
        while True:
            try:
                text = input("🗣️  Nhập câu: ").strip()
                
                if not text:
                    continue
                
                # Xử lý lệnh
                if text.lower() in ['quit', 'exit', 'thoát']:
                    print("\n👋 Tạm biệt!\n")
                    break
                elif text.lower() in ['history', 'lịch sử']:
                    self.show_history()
                    continue
                elif text.lower() in ['stats', 'thống kê']:
                    self.show_statistics()
                    continue
                
                # Phân loại cảm xúc
                result = self.classify_and_save(text)
                
                # Kiểm tra lỗi
                if 'error' in result:
                    continue
                
                # Hiển thị kết quả theo format mới
                emoji_map = {
                    'POSITIVE': '😊',
                    'NEGATIVE': '😞',
                    'NEUTRAL': '😐'
                }
                
                emoji = emoji_map.get(result['sentiment'], '🤔')
                confidence = result.get('confidence', 0.0)
                
                print(f"\n{emoji} Cảm xúc: {result['sentiment']}")
                print(f"   Độ tin cậy: {confidence:.2%}")
                print(f"   Output: {{\"text\": \"{result['text']}\", \"sentiment\": \"{result['sentiment']}\"}}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!\n")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {e}\n")

def main():
    """Hàm main"""
    try:
        app = SentimentApp()
        
        # Kiểm tra nếu có tham số dòng lệnh
        if len(sys.argv) > 1:
            # Phân loại câu từ tham số
            text = " ".join(sys.argv[1:])
            result = app.classify_and_save(text)
            
            if 'error' not in result:
                print(f"\nCâu: {result['text']}")
                print(f"Cảm xúc: {result['sentiment']}")
                print(f"Độ tin cậy: {result.get('confidence', 0.0):.2%}")
                print(f"Output: {{\"text\": \"{result['text']}\", \"sentiment\": \"{result['sentiment']}\"}}\n")
        else:
            # Chạy chế độ tương tác
            app.run_interactive()
            
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
