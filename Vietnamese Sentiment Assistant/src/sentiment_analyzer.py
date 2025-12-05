from transformers import pipeline
from typing import Dict
import torch
import re

class SentimentAnalyzer:
    """
    Phân loại cảm xúc sử dụng Transformer pre-trained
    
    Kiến trúc tổng quát (Flowchart):
    [Đầu vào: Câu tiếng Việt]
        ↓ (Preprocessing)
    [Component 1: Tiền xử lý] → Câu đã chuẩn hóa
        ↓ (Sentiment Analysis)
    [Component 2: Phân loại cảm xúc] → Nhãn (POSITIVE, NEUTRAL, NEGATIVE)
        ↓ (Validation)
    [Component 3: Hợp nhất & xử lý lỗi] → Đầu ra dictionary hoặc lỗi
        ↓
    [Core Engine: Lưu & hiển thị]
    """
    
    def __init__(self, model_name: str = "phobert"):
        """
        Khởi tạo pipeline sentiment analysis
        
        Args:
            model_name: Tên model ('phobert' hoặc 'distilbert')
        """
        self.model = None
        self.model_name = model_name
        self.sentiment_map = {
            'POSITIVE': 'POSITIVE',
            'NEGATIVE': 'NEGATIVE',
            'NEUTRAL': 'NEUTRAL',
            'POS': 'POSITIVE',
            'NEG': 'NEGATIVE',
            'NEU': 'NEUTRAL',
            'LABEL_2': 'POSITIVE',
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            # Multilingual BERT 5-star rating
            '1 star': 'NEGATIVE',
            '2 stars': 'NEGATIVE',
            '3 stars': 'NEUTRAL',
            '4 stars': 'POSITIVE',
            '5 stars': 'POSITIVE',
        }
        
        # Từ khóa tiếng Việt để boost accuracy
        self.positive_keywords = [
            'vui', 'tuyệt', 'hay', 'đẹp', 'hạnh phúc', 'thích', 'yêu',
            'xuất sắc', 'hoàn hảo', 'tuyệt vời', 'tốt lắm', 'rất tốt', 'ngon',
            'cảm ơn', 'cám ơn', 'hài lòng', 'thành công', 'tích cực', 'tuyệt',
            'tuyệt vời', 'tốt', 'hay lắm', 'yêu thích', 'thích thú', 'vui vẻ'
        ]
        self.negative_keywords = [
            'buồn', 'tệ', 'dở', 'kém', 'xấu', 'ghét', 'thất bại', 'thất vọng',
            'chán', 'mệt', 'tồi', 'tệ hại', 'không tốt', 'không hay', 'tệ quá',
            'dở quá', 'không thích', 'thảm họa', 'tiêu cực', 'tệ', 'kém',
            'dở', 'tồi tệ', 'tệ hại', 'không tốt', 'xấu', 'chán'
        ]
        self.neutral_keywords = [
            'ổn định', 'bình thường', 'thường', 'trung bình'
        ]
        
        self.load_model()
    
    def load_model(self):
        """
        Bước 2: Phân loại cảm xúc
        
        Sử dụng pipeline sentiment-analysis với model phobert-base-v2 
        (tiền tệ tiếng Việt) hoặc distilbert-base-multilingual-cased (hỗ trợ đa ngôn ngữ)
        """
        try:
            if self.model_name == "phobert":
                # Load PhoBERT (model tiếng Việt)
                print("Đang tải model PhoBERT tiếng Việt...")
                self.model = pipeline(
                    "sentiment-analysis",
                    model="vinai/phobert-base-v2",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✓ Đã tải PhoBERT thành công!")
            else:
                # Load DistilBERT multilingual
                print("Đang tải DistilBERT multilingual...")
                self.model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-multilingual-cased",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✓ Đã tải DistilBERT multilingual thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model {self.model_name}: {e}")
            # Thử fallback
            try:
                fallback_model = "distilbert" if self.model_name == "phobert" else "phobert"
                print(f"Đang thử model {fallback_model}...")
                if fallback_model == "phobert":
                    self.model = pipeline(
                        "sentiment-analysis",
                        model="vinai/phobert-base-v2",
                        device=0 if torch.cuda.is_available() else -1
                    )
                else:
                    self.model = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-multilingual-cased",
                        device=0 if torch.cuda.is_available() else -1
                    )
                print(f"✓ Đã tải {fallback_model} thành công!")
            except Exception as e2:
                print(f"Lỗi khi tải fallback model: {e2}")
                raise Exception("Không thể tải bất kỳ model nào!")
    
    def preprocess(self, text: str) -> str:
        """
        Bước 1: Tiền xử lý
        
        Chuẩn hóa câu tiếng Việt để phù hợp với Transformer.
        
        Hướng dẫn thực hiện:
        - (Optional) Dùng underthesea.word_tokenize() để tách từ 
          (VD: "Rất vui hôm nay" → "Rất vui hôm nay").
        - Chuyển chữ thường, thay "rất" → "rất" bằng từ điển nhỏ 
          (10–20 từ phổ biến: rất/rất, dở/dở, v.v.).
        - Giữ hầu hết chữ cái, loại bỏ ký tự dễ giảm thời gian xử lý.
        
        Args:
            text: Câu văn gốc
            
        Returns:
            Câu văn đã chuẩn hóa
        """
        if not text:
            return ""
        
        # Loại bỏ khoảng trắng thừa
        text = text.strip()
        
        # Chuẩn hóa các ký tự đặc biệt phổ biến
        replacements = {
            'rất': 'rất',
            'dở': 'dở',
            'tệ': 'tệ',
            'tuyệt': 'tuyệt',
            'hay': 'hay',
            'buồn': 'buồn',
            'vui': 'vui',
            'mệt': 'mệt',
            'ổn': 'ổn',
            'tốt': 'tốt',
        }
        
        # Thay thế các từ chuẩn hóa
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Loại bỏ ký tự đặc biệt không cần thiết, giữ chữ cái và dấu câu cơ bản
        text = re.sub(r'[^\w\s\.,!?áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', text, flags=re.IGNORECASE)
        
        return text
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Phân tích cảm xúc của câu văn - Tích hợp 3 bước
        
        Bước 1: Tiền xử lý (Preprocessing)
        Bước 2: Phân loại cảm xúc (Sentiment Analysis) 
        Bước 3: Hợp nhất & xử lý lỗi (Validation)
        
        Args:
            text: Câu văn tiếng Việt cần phân tích
            
        Returns:
            Dictionary theo format: {"text": "câu", "sentiment": "POSITIVE/NEGATIVE/NEUTRAL"}
        """
        # Bước 3: Kiểm tra - Câu nhập ≥5 ký tự
        if not text or not text.strip() or len(text.strip()) < 5:
            return {
                'text': text,
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'error': 'Câu không hợp lệ, thử lại!'
            }
        
        try:
            # Bước 1: Tiền xử lý
            processed_text = self.preprocess(text)
            
            # Rule-based boost cho từ khóa rõ ràng
            text_lower = text.lower()
            keyword_sentiment = None
            keyword_confidence = 0.0
            
            # Đếm từ khóa tích cực, tiêu cực và trung tính
            positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
            negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
            neutral_count = sum(1 for word in self.neutral_keywords if word in text_lower)
            
            # Nếu có từ khóa rõ ràng, ưu tiên rule-based
            if neutral_count > 0 and positive_count == 0 and negative_count == 0:
                keyword_sentiment = 'NEUTRAL'
                keyword_confidence = 0.70
            elif positive_count > negative_count and positive_count > 0:
                keyword_sentiment = 'POSITIVE'
                keyword_confidence = min(0.75 + (positive_count * 0.1), 0.95)
            elif negative_count > positive_count and negative_count > 0:
                keyword_sentiment = 'NEGATIVE'
                keyword_confidence = min(0.75 + (negative_count * 0.1), 0.95)
            
            # Bước 2: Phân loại cảm xúc
            # Sử dụng pipeline sentiment-analysis với model
            # Gửi câu chuẩn hóa qua pipeline, lấy nhãn có xác suất cao nhất
            result = self.model(processed_text)[0]
            
            # Lấy nhãn và confidence
            raw_label = result['label']
            confidence = result['score']
            
            # Chuyển đổi nhãn sang format chuẩn (POSITIVE/NEGATIVE/NEUTRAL)
            sentiment = self.sentiment_map.get(raw_label.upper(), 'NEUTRAL')
            
            # Nếu có keyword match mạnh và model không chắc chắn, ưu tiên keyword
            if keyword_sentiment and confidence < 0.7:
                sentiment = keyword_sentiment
                confidence = keyword_confidence
            # Nếu keyword rất mạnh (1+ từ negative mạnh như tệ/kém), override model
            elif keyword_sentiment == 'NEGATIVE' and (negative_count >= 1) and confidence < 0.85:
                sentiment = keyword_sentiment
                confidence = max(confidence, keyword_confidence)
            # Nếu keyword positive mạnh (2+ từ), override model
            elif keyword_sentiment == 'POSITIVE' and (positive_count >= 2):
                sentiment = keyword_sentiment
                confidence = max(confidence, keyword_confidence)
            
            # Bước 3: Validation - Nếu xác suất <0.5, trả về NEUTRAL mặc định
            if confidence < 0.5 and not keyword_sentiment:
                sentiment = 'NEUTRAL'
            
            # Bước 3: Tạo dictionary format: {text, sentiment}
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'raw_label': raw_label
            }
            
        except Exception as e:
            # Bước 3: Xử lý lỗi - hiển thị pop-up "Câu không hợp lệ, thử lại!"
            print(f"Lỗi khi phân tích: {e}")
            return {
                'text': text,
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'error': f'Câu không hợp lệ, thử lại! ({str(e)})'
            }
    
    def batch_analyze(self, texts: list) -> list:
        """
        Phân tích nhiều câu văn cùng lúc
        
        Args:
            texts: Danh sách các câu văn
            
        Returns:
            Danh sách kết quả phân tích
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results
