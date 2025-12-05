import sqlite3
from datetime import datetime
from typing import List, Tuple
import os

class SentimentDatabase:
    """Quản lý cơ sở dữ liệu SQLite cho lịch sử phân loại cảm xúc"""
    
    def __init__(self, db_path: str = "data/sentiment_history.db"):
        """
        Khởi tạo kết nối database
        
        Args:
            db_path: Đường dẫn đến file database
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """
        Tạo bảng sentiments nếu chưa tồn tại
        
        Cấu trúc bảng:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - text: TEXT NOT NULL (câu văn đầu vào)
        - sentiment: TEXT NOT NULL (POSITIVE/NEGATIVE/NEUTRAL)
        - timestamp: TEXT NOT NULL (ISO format: YYYY-MM-DD HH:MM:SS)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_classification(self, text: str, label: str, confidence: float = None):
        """
        Lưu kết quả phân loại vào database với parameterized queries
        
        Giải pháp bảo mật:
        - Sử dụng parameterized queries để tránh SQL injection
        - cursor.execute("INSERT INTO sentiments VALUES (?, ?, ?, ?)", (id, text, sentiment, timestamp))
        
        Args:
            text: Câu văn đầu vào
            label: Nhãn cảm xúc (POSITIVE/NEGATIVE/NEUTRAL)
            confidence: Độ tin cậy (không lưu vào DB theo yêu cầu mới)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tạo timestamp ISO format: YYYY-MM-DD HH:MM:SS
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Sử dụng parameterized query để tránh SQL injection
        cursor.execute(
            "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
            (text, label, timestamp)
        )
        
        conn.commit()
        conn.close()
    
    def get_history(self, limit: int = 50, offset: int = 0) -> List[Tuple]:
        """
        Lấy lịch sử phân loại với giới hạn 50 bản ghi mới nhất
        
        Giải pháp tối ưu:
        - Giới hạn mặc định 50 bản ghi để tránh làm chậm giao diện
        - Hỗ trợ offset cho pagination (nút "Tải thêm")
        - Sử dụng ORDER BY timestamp DESC để lấy mới nhất
        - Parameterized query để tránh SQL injection
        
        Args:
            limit: Số lượng bản ghi tối đa (mặc định 50)
            offset: Vị trí bắt đầu (cho pagination)
            
        Returns:
            Danh sách các bản ghi lịch sử (id, text, sentiment, timestamp)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sử dụng parameterized query
        cursor.execute(
            "SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def clear_history(self):
        """
        Xóa toàn bộ lịch sử
        
        Sử dụng parameterized query (không cần vì không có tham số,
        nhưng vẫn an toàn với cú pháp trực tiếp)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sentiments")
        conn.commit()
        conn.close()
    
    def get_total_count(self) -> int:
        """
        Lấy tổng số bản ghi
        
        Returns:
            Tổng số bản ghi trong database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sentiments")
        result = cursor.fetchone()
        conn.close()
        return result[0]
    
    def get_statistics(self) -> dict:
        """
        Lấy thống kê tổng quan
        
        Sử dụng parameterized query và COUNT với CASE
        
        Returns:
            Dictionary chứa thống kê
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment = 'NEUTRAL' THEN 1 ELSE 0 END) as neutral,
                SUM(CASE WHEN sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as negative
            FROM sentiments
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total': result[0],
            'positive': result[1],
            'neutral': result[2],
            'negative': result[3]
        }
