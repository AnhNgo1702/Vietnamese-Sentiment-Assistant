"""
Giao diện web Streamlit cho ứng dụng phân loại cảm xúc tiếng Việt

Chạy: streamlit run src/app.py
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
from sentiment_analyzer import SentimentAnalyzer
from database import SentimentDatabase

# Cấu hình trang
st.set_page_config(
    page_title="Phân loại Cảm xúc Tiếng Việt",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #C8E6C9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .sentiment-negative {
        background-color: #FFCDD2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
    }
    .sentiment-neutral {
        background-color: #E0E0E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9E9E9E;
    }
    .result-box {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'db' not in st.session_state:
    st.session_state.db = SentimentDatabase()

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "phobert"

if 'analyzer' not in st.session_state or st.session_state.get('current_model') != st.session_state.selected_model:
    with st.spinner(f'Đang tải model {st.session_state.selected_model}...'):
        st.session_state.analyzer = SentimentAnalyzer(model_name=st.session_state.selected_model)
        st.session_state.current_model = st.session_state.selected_model

# Header
st.markdown('<div class="main-header">🤖 Phân loại Cảm xúc Tiếng Việt với AI</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🤖 Chọn Model")
    
    model_option = st.radio(
        "Model AI:",
        options=["phobert", "distilbert"],
        format_func=lambda x: "PhoBERT-v2 (Tiếng Việt)" if x == "phobert" else "DistilBERT (Đa ngôn ngữ)",
        index=0 if st.session_state.selected_model == "phobert" else 1,
        help="Chọn model để phân loại cảm xúc"
    )
    
    if model_option != st.session_state.selected_model:
        st.session_state.selected_model = model_option
        st.rerun()
    
    st.markdown("---")
    
    st.header("📊 Thông tin")
    st.info(f"""
    **Model đang dùng:**
    - 🧠 {st.session_state.selected_model.upper()}
    
    **Công nghệ:**
    - 🧠 Transformer Pre-trained
    - 🗄️ SQLite Database
    - 🎨 Streamlit UI
    
    **Hỗ trợ 3 loại cảm xúc:**
    - 😊 POSITIVE (Tích cực)
    - 😞 NEGATIVE (Tiêu cực)
    - 😐 NEUTRAL (Trung tính)
    """)
    
    st.markdown("---")
    
    # Thống kê
    stats = st.session_state.db.get_statistics()
    st.subheader("📈 Thống kê")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tổng", stats['total'])
        st.metric("😊 Tích cực", stats['positive'])
    with col2:
        st.metric("😞 Tiêu cực", stats['negative'])
        st.metric("😐 Trung tính", stats['neutral'])
    
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.db.clear_history()
        st.success("Đã xóa lịch sử!")
        st.rerun()

# Tab chính
tab1, tab2, tab3 = st.tabs(["🗣️ Phân loại", "📜 Lịch sử", "📊 Biểu đồ"])

# Tab 1: Phân loại cảm xúc
with tab1:
    st.header("Nhập câu tiếng Việt để phân loại")
    
    # Form nhập liệu
    with st.form(key='sentiment_form'):
        text_input = st.text_area(
            "Câu văn (tối thiểu 5 ký tự):",
            height=100,
            placeholder="Ví dụ: Hôm nay tôi rất vui...",
            help="Nhập câu tiếng Việt để phân tích cảm xúc"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit_button = st.form_submit_button("🚀 Phân loại", width='stretch')
        with col2:
            clear_button = st.form_submit_button("🔄 Xóa", width='stretch')
    
    # Xử lý khi submit
    if submit_button and text_input:
        if len(text_input.strip()) < 5:
            st.error("⚠️ Câu không hợp lệ, thử lại! (Cần ít nhất 5 ký tự)")
        else:
            with st.spinner('Đang phân tích...'):
                # Phân tích
                result = st.session_state.analyzer.analyze(text_input)
                
                # Kiểm tra lỗi
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Lưu vào database
                    st.session_state.db.save_classification(
                        text=result['text'],
                        label=result['sentiment']
                    )
                    
                    # Hiển thị kết quả
                    sentiment = result['sentiment']
                    
                    # CSS class theo sentiment
                    css_class = {
                        'POSITIVE': 'sentiment-positive',
                        'NEGATIVE': 'sentiment-negative',
                        'NEUTRAL': 'sentiment-neutral'
                    }.get(sentiment, 'sentiment-neutral')
                    
                    # Emoji
                    emoji = {
                        'POSITIVE': '😊',
                        'NEGATIVE': '😞',
                        'NEUTRAL': '😐'
                    }.get(sentiment, '🤔')
                    
                    # Tên tiếng Việt
                    sentiment_vn = {
                        'POSITIVE': 'TÍCH CỰC',
                        'NEGATIVE': 'TIÊU CỰC',
                        'NEUTRAL': 'TRUNG TÍNH'
                    }.get(sentiment, sentiment)
                    
                    # Hiển thị kết quả
                    st.markdown(f'<div class="{css_class} result-box">', unsafe_allow_html=True)
                    st.markdown(f"### {emoji} Cảm xúc: {sentiment_vn}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # JSON output
                    st.subheader("📄 Kết quả JSON")
                    output_json = {"text": result['text'], "sentiment": sentiment}
                    st.json(output_json)
                    
                    # Download JSON
                    st.download_button(
                        label="💾 Tải JSON",
                        data=json.dumps(output_json, ensure_ascii=False, indent=2),
                        file_name=f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Examples
    st.markdown("---")
    st.subheader("💡 Ví dụ mẫu")
    examples_col1, examples_col2, examples_col3 = st.columns(3)
    
    with examples_col1:
        if st.button("😊 Tích cực", width='stretch'):
            st.session_state['example'] = "Hôm nay tôi rất vui và hạnh phúc!"
            st.rerun()
    
    with examples_col2:
        if st.button("😞 Tiêu cực", width='stretch'):
            st.session_state['example'] = "Tôi cảm thấy buồn và thất vọng"
            st.rerun()
    
    with examples_col3:
        if st.button("😐 Trung tính", width='stretch'):
            st.session_state['example'] = "Hôm nay trời đẹp"
            st.rerun()

# Tab 2: Lịch sử
with tab2:
    st.header("📜 Lịch sử phân loại")
    
    history = st.session_state.db.get_history(limit=50)
    
    if not history:
        st.info("Chưa có lịch sử phân loại nào.")
    else:
        # Chuyển sang DataFrame
        df = pd.DataFrame(history, columns=['ID', 'Câu văn', 'Cảm xúc', 'Thời gian'])
        
        # Hiển thị bảng
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Câu văn": st.column_config.TextColumn("Câu văn", width="large"),
                "Cảm xúc": st.column_config.TextColumn("Cảm xúc", width="medium"),
                "Thời gian": st.column_config.DatetimeColumn("Thời gian", width="medium")
            }
        )
        
        total_count = st.session_state.db.get_total_count()
        if total_count > 50:
            st.info(f"Hiển thị 50/{total_count} bản ghi mới nhất")
        
        # Download CSV
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Tải CSV",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Tab 3: Biểu đồ
with tab3:
    st.header("📊 Biểu đồ thống kê")
    
    stats = st.session_state.db.get_statistics()
    
    if stats['total'] == 0:
        st.info("Chưa có dữ liệu để hiển thị biểu đồ.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            st.subheader("Phân bố cảm xúc")
            pie_data = pd.DataFrame({
                'Cảm xúc': ['Tích cực', 'Tiêu cực', 'Trung tính'],
                'Số lượng': [stats['positive'], stats['negative'], stats['neutral']],
                'Emoji': ['😊', '😞', '😐']
            })
            
            fig_pie = px.pie(
                pie_data, 
                values='Số lượng', 
                names='Cảm xúc',
                color='Cảm xúc',
                color_discrete_map={
                    'Tích cực': '#4CAF50',
                    'Tiêu cực': '#F44336',
                    'Trung tính': '#9E9E9E'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            st.subheader("Số lượng theo loại")
            bar_data = pd.DataFrame({
                'Cảm xúc': ['😊 Tích cực', '😞 Tiêu cực', '😐 Trung tính'],
                'Số lượng': [stats['positive'], stats['negative'], stats['neutral']]
            })
            
            fig_bar = px.bar(
                bar_data,
                x='Cảm xúc',
                y='Số lượng',
                color='Cảm xúc',
                color_discrete_map={
                    '😊 Tích cực': '#4CAF50',
                    '😞 Tiêu cực': '#F44336',
                    '😐 Trung tính': '#9E9E9E'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline
        st.subheader("Xu hướng theo thời gian")
        history = st.session_state.db.get_history(limit=100)
        if history:
            timeline_df = pd.DataFrame(history, columns=['ID', 'Câu văn', 'Cảm xúc', 'Độ tin cậy', 'Thời gian'])
            timeline_df['Thời gian'] = pd.to_datetime(timeline_df['Thời gian'])
            
            # Group by time and sentiment
            timeline_grouped = timeline_df.groupby([
                pd.Grouper(key='Thời gian', freq='H'),
                'Cảm xúc'
            ]).size().reset_index(name='Số lượng')
            
            fig_timeline = px.line(
                timeline_grouped,
                x='Thời gian',
                y='Số lượng',
                color='Cảm xúc',
                color_discrete_map={
                    'POSITIVE': '#4CAF50',
                    'NEGATIVE': '#F44336',
                    'NEUTRAL': '#9E9E9E'
                }
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Phân loại Cảm xúc Tiếng Việt với Transformer AI</p>
        <p>Developed with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
