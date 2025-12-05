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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh với màu chủ đạo #40FFF5 và đen
st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #40FFF5;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(64, 255, 245, 0.5);
        letter-spacing: 2px;
    }
    
    /* Icon Style - White Outline */
    .icon-outline {
        color: transparent;
        -webkit-text-stroke: 2px #FFFFFF;
        text-stroke: 2px #FFFFFF;
        font-size: 2rem;
        filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.3));
    }
    
    /* Sentiment Cards */
    .sentiment-positive {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #40FFF5;
        box-shadow: 0 0 30px rgba(64, 255, 245, 0.3);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #FF4040;
        box-shadow: 0 0 30px rgba(255, 64, 64, 0.3);
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #FFFFFF;
        box-shadow: 0 0 30px rgba(255, 255, 255, 0.2);
    }
    
    /* Result Box */
    .result-box {
        font-size: 2rem;
        font-weight: 700;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #40FFF5;
        text-shadow: 0 0 15px rgba(64, 255, 245, 0.5);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #1a1a1a 100%);
        border-right: 2px solid #40FFF5;
    }
    
    /* Text Colors */
    .stMarkdown, p, span, label {
        color: #FFFFFF !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #40FFF5 !important;
        text-shadow: 0 0 10px rgba(64, 255, 245, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #40FFF5 0%, #00CCC4 100%);
        color: #000000;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        box-shadow: 0 0 20px rgba(64, 255, 245, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(64, 255, 245, 0.6);
    }
    
    /* Input Fields */
    .stTextArea textarea, .stTextInput input {
        background-color: #1a1a1a !important;
        color: #FFFFFF !important;
        border: 2px solid #40FFF5 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(64, 255, 245, 0.2) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
        border-bottom: 2px solid #40FFF5;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        background-color: transparent;
        border: 2px solid transparent;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #40FFF5 0%, #00CCC4 100%);
        color: #000000;
        border: 2px solid #40FFF5;
        box-shadow: 0 0 20px rgba(64, 255, 245, 0.5);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #40FFF5 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #1a1a1a;
        border: 2px solid #40FFF5;
        border-radius: 10px;
    }
    
    /* Info/Success/Error boxes */
    .stAlert {
        background-color: #1a1a1a !important;
        border: 2px solid #40FFF5 !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }
    
    /* Divider */
    hr {
        border-color: #40FFF5 !important;
        opacity: 0.3;
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
st.markdown('''
    <div class="main-header">
        <span class="icon-outline">◈</span> PHÂN LOẠI CẢM XÚC TIẾNG VIỆT <span class="icon-outline">◈</span>
    </div>
''', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color: #40FFF5;">⚙ CHỌN MODEL</h2>', unsafe_allow_html=True)
    
    model_option = st.radio(
        "Model AI:",
        options=["phobert", "distilbert"],
        format_func=lambda x: "◆ PhoBERT-v2 (Tiếng Việt)" if x == "phobert" else "◆ DistilBERT (Đa ngôn ngữ)",
        index=0 if st.session_state.selected_model == "phobert" else 1,
        help="Chọn model để phân loại cảm xúc"
    )
    
    if model_option != st.session_state.selected_model:
        st.session_state.selected_model = model_option
        st.rerun()
    
    st.markdown("---")
    
    st.markdown('<h2 style="color: #40FFF5;">ℹ THÔNG TIN</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: #1a1a1a; padding: 1rem; border-radius: 10px; border: 2px solid #40FFF5;">
    <p><strong>Model đang dùng:</strong></p>
    <p>◆ {st.session_state.selected_model.upper()}</p>
    
    <p><strong>Công nghệ:</strong></p>
    <p>◆ Transformer Pre-trained</p>
    <p>◆ SQLite Database</p>
    <p>◆ Streamlit UI</p>
    
    <p><strong>Hỗ trợ 3 loại cảm xúc:</strong></p>
    <p>◆ POSITIVE (Tích cực)</p>
    <p>◆ NEGATIVE (Tiêu cực)</p>
    <p>◆ NEUTRAL (Trung tính)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Thống kê
    stats = st.session_state.db.get_statistics()
    st.markdown('<h2 style="color: #40FFF5;">◆ THỐNG KÊ</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("◆ Tổng", stats['total'])
        st.metric("◆ Tích cực", stats['positive'])
    with col2:
        st.metric("◆ Tiêu cực", stats['negative'])
        st.metric("◆ Trung tính", stats['neutral'])
    
    if st.button("◆ XÓA LỊCH SỬ"):
        st.session_state.db.clear_history()
        st.success("✓ Đã xóa lịch sử!")
        st.rerun()

# Tab chính
tab1, tab2, tab3 = st.tabs(["◆ PHÂN LOẠI", "◆ LỊCH SỬ", "◆ BIỂU ĐỒ"])

# Tab 1: Phân loại cảm xúc
with tab1:
    st.markdown('<h2 style="color: #40FFF5; text-align: center;">◈ NHẬP CÂU TIẾNG VIỆT ĐỂ PHÂN LOẠI ◈</h2>', unsafe_allow_html=True)
    
    # Form nhập liệu
    with st.form(key='sentiment_form'):
        text_input = st.text_area(
            "◆ Câu văn (tối thiểu 5 ký tự):",
            height=120,
            placeholder="Ví dụ: Hôm nay tôi rất vui...",
            help="Nhập câu tiếng Việt để phân tích cảm xúc"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit_button = st.form_submit_button("◆ PHÂN LOẠI", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("◆ XÓA", use_container_width=True)
    
    # Xử lý khi submit
    if submit_button and text_input:
        if len(text_input.strip()) < 5:
            st.error("◇ Câu không hợp lệ, thử lại! (Cần ít nhất 5 ký tự)")
        else:
            with st.spinner('Đang phân tích...'):
                # Phân tích
                result = st.session_state.analyzer.analyze(text_input)
                
                # Kiểm tra lỗi
                if 'error' in result:
                    st.error(f"✗ {result['error']}")
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
                    st.markdown(f"### ◈ CẢM XÚC: {sentiment_vn} ◈")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # JSON output
                    st.markdown('<h3 style="color: #40FFF5;">◆ KẾT QUẢ JSON</h3>', unsafe_allow_html=True)
                    output_json = {"text": result['text'], "sentiment": sentiment}
                    st.json(output_json)
                    
                    # Download JSON
                    st.download_button(
                        label="◆ TẢI JSON",
                        data=json.dumps(output_json, ensure_ascii=False, indent=2),
                        file_name=f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Examples
    st.markdown("---")
    st.markdown('<h3 style="color: #40FFF5; text-align: center;">◆ VÍ DỤ MẪU ◆</h3>', unsafe_allow_html=True)
    examples_col1, examples_col2, examples_col3 = st.columns(3)
    
    with examples_col1:
        if st.button("◆ TÍCH CỰC", use_container_width=True):
            st.session_state['example'] = "Hôm nay tôi rất vui và hạnh phúc!"
            st.rerun()
    
    with examples_col2:
        if st.button("◆ TIÊU CỰC", use_container_width=True):
            st.session_state['example'] = "Tôi cảm thấy buồn và thất vọng"
            st.rerun()
    
    with examples_col3:
        if st.button("◆ TRUNG TÍNH", use_container_width=True):
            st.session_state['example'] = "Hôm nay trời đẹp"
            st.rerun()

# Tab 2: Lịch sử
with tab2:
    st.markdown('<h2 style="color: #40FFF5; text-align: center;">◈ LỊCH SỬ PHÂN LOẠI ◈</h2>', unsafe_allow_html=True)
    
    history = st.session_state.db.get_history(limit=50)
    
    if not history:
        st.info("◆ Chưa có lịch sử phân loại nào.")
    else:
        # Chuyển sang DataFrame
        df = pd.DataFrame(history, columns=['ID', 'Câu văn', 'Cảm xúc', 'Thời gian'])
        
        # Hiển thị bảng
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("◆ ID", width="small"),
                "Câu văn": st.column_config.TextColumn("◆ Câu văn", width="large"),
                "Cảm xúc": st.column_config.TextColumn("◆ Cảm xúc", width="medium"),
                "Thời gian": st.column_config.DatetimeColumn("◆ Thời gian", width="medium")
            }
        )
        
        total_count = st.session_state.db.get_total_count()
        if total_count > 50:
            st.info(f"◆ Hiển thị 50/{total_count} bản ghi mới nhất")
        
        # Download CSV
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="◆ TẢI CSV",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Tab 3: Biểu đồ
with tab3:
    st.markdown('<h2 style="color: #40FFF5; text-align: center;">◈ BIỂU ĐỒ THỐNG KÊ ◈</h2>', unsafe_allow_html=True)
    
    stats = st.session_state.db.get_statistics()
    
    if stats['total'] == 0:
        st.info("◆ Chưa có dữ liệu để hiển thị biểu đồ.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            st.markdown('<h3 style="color: #40FFF5;">◆ Phân bố cảm xúc</h3>', unsafe_allow_html=True)
            pie_data = pd.DataFrame({
                'Cảm xúc': ['Tích cực', 'Tiêu cực', 'Trung tính'],
                'Số lượng': [stats['positive'], stats['negative'], stats['neutral']],
                'Emoji': ['◆', '◆', '◆']
            })
            
            fig_pie = px.pie(
                pie_data, 
                values='Số lượng', 
                names='Cảm xúc',
                color='Cảm xúc',
                color_discrete_map={
                    'Tích cực': '#40FFF5',
                    'Tiêu cực': '#FF4040',
                    'Trung tính': '#FFFFFF'
                }
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF', size=14)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            st.markdown('<h3 style="color: #40FFF5;">◆ Số lượng theo loại</h3>', unsafe_allow_html=True)
            bar_data = pd.DataFrame({
                'Cảm xúc': ['◆ Tích cực', '◆ Tiêu cực', '◆ Trung tính'],
                'Số lượng': [stats['positive'], stats['negative'], stats['neutral']]
            })
            
            fig_bar = px.bar(
                bar_data,
                x='Cảm xúc',
                y='Số lượng',
                color='Cảm xúc',
                color_discrete_map={
                    '◆ Tích cực': '#40FFF5',
                    '◆ Tiêu cực': '#FF4040',
                    '◆ Trung tính': '#FFFFFF'
                }
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,26,1)',
                font=dict(color='#FFFFFF', size=14),
                xaxis=dict(gridcolor='#333333'),
                yaxis=dict(gridcolor='#333333')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline
        st.markdown('<h3 style="color: #40FFF5;">◆ Xu hướng theo thời gian</h3>', unsafe_allow_html=True)
        history = st.session_state.db.get_history(limit=100)
        if history:
            timeline_df = pd.DataFrame(history, columns=['ID', 'Câu văn', 'Cảm xúc', 'Thời gian'])
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
                    'POSITIVE': '#40FFF5',
                    'NEGATIVE': '#FF4040',
                    'NEUTRAL': '#FFFFFF'
                }
            )
            fig_timeline.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,26,1)',
                font=dict(color='#FFFFFF', size=14),
                xaxis=dict(gridcolor='#333333'),
                yaxis=dict(gridcolor='#333333')
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #40FFF5; padding: 2rem;'>
        <h3 style='color: #40FFF5; text-shadow: 0 0 15px rgba(64, 255, 245, 0.5);'>
            ◈ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT VỚI TRANSFORMER AI ◈
        </h3>
        <p style='color: #FFFFFF; font-size: 1.1rem; margin-top: 1rem;'>
            Developed with <span style='color: #40FFF5;'>◆</span> using Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)
