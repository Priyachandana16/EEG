import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_css = '''THEME = {
    'bg': '#050509',
    'surface': 'rgba(255, 255, 255, 0.015)',
    'surface_card': 'rgba(255, 255, 255, 0.025)',
    'border': 'rgba(255, 255, 255, 0.05)',
    'border_focus': 'rgba(139, 92, 246, 0.5)',
    'text_primary': '#F8FAFC',
    'text_secondary': '#94A3B8',
    'text_muted': '#475569',
    'teal': '#2DD4BF',
    'cyan': '#22D3EE',
    'indigo': '#818CF8',
    'violet': '#C084FC',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#F43F5E',
    'coral': '#FB7185',
    'grid': 'rgba(255, 255, 255, 0.03)',
    'font_family': "'Outfit', -apple-system, sans-serif",
    'mono_family': "'Fira Code', monospace"
}

def inject_css():
    """
    Injects ultra-premium Glassmorphism custom CSS with modern typography and glowing accents.
    """
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600;700&display=swap');
        
        /* Root & Global Typography */
        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Outfit', -apple-system, sans-serif;
            color: #F8FAFC !important;
        }

        /* App Background with Subtle Dual Radial Glow */
        .stApp {
            background-color: #050509;
            background-image: 
                radial-gradient(circle at 15% 10%, rgba(139, 92, 246, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 90%, rgba(45, 212, 191, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }

        /* Hide Streamlit Clutter & Clean Layout */
        #MainMenu, footer, .stDeployButton { display: none !important; }
        header[data-testid="stHeader"] { background: transparent !important; }
        .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1440px; }
        [data-testid="stSidebarCollapseButton"] {
            color: #F8FAFC !important;
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px);
        }

        /* Sidebar Glassmorphism */
        section[data-testid="stSidebar"] {
            background: rgba(5, 5, 9, 0.6) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
        }
        section[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.05) !important; }

        /* Ultra-Premium Glass Cards */
        .cyber-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.005) 100%);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 24px;
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.08);
            padding: 24px 28px;
            margin-bottom: 24px;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .cyber-card:hover {
            border-color: rgba(139, 92, 246, 0.3);
            box-shadow: 0 15px 50px -10px rgba(139, 92, 246, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.12);
            transform: translateY(-3px);
        }

        .cyber-card-slim {
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            padding: 16px 24px;
            margin-bottom: 20px;
        }

        /* Typography & Headings */
        .section-header { margin-bottom: 20px; }
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(90deg, #FFFFFF, #94A3B8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.2px;
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0;
        }
        .section-subtitle {
            font-size: 0.9rem;
            font-weight: 400;
            color: #64748B !important;
            margin: 6px 0 0 0;
            line-height: 1.6;
        }

        /* Hero Section Premium Treatment */
        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            letter-spacing: -1.5px;
            background: linear-gradient(135deg, #FFFFFF 0%, #C084FC 45%, #2DD4BF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 12px 0;
            line-height: 1.1;
            filter: drop-shadow(0 4px 20px rgba(192, 132, 252, 0.25));
        }
        .hero-subtitle {
            font-size: 1.2rem;
            font-weight: 600;
            color: #A78BFA !important;
            margin: 0 0 16px 0;
            letter-spacing: -0.2px;
        }
        .hero-desc {
            font-size: 1rem;
            color: #94A3B8 !important;
            margin: 0 0 20px 0;
            line-height: 1.6;
            font-weight: 300;
        }
        .hero-pill {
            background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #E2E8F0 !important;
            padding: 6px 16px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-right: 10px;
            box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
        }

        /* Sidebar Section Labels */
        .sidebar-section-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #475569 !important;
            margin: 24px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Streaming Badges (Glowing) */
        .stream-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 18px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
            backdrop-filter: blur(12px);
        }
        .stream-live {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34D399 !important;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
        }
        .stream-paused {
            background: rgba(100, 116, 139, 0.1);
            border: 1px solid rgba(100, 116, 139, 0.2);
            color: #94A3B8 !important;
        }
        .live-dot {
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8);
            animation: pulse-green 1.5s infinite cubic-bezier(0.4, 0, 0.6, 1);
        }
        @keyframes pulse-green {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Benchmark Compact Metric Card */
        .bench-metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .bench-metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .bench-metric-value {
            font-family: 'Fira Code', monospace;
            font-size: 1.3rem;
            font-weight: 700;
            color: #F8FAFC !important;
            margin-top: 4px;
        }

        /* Decision Banner Extravaganza */
        .decision-banner {
            border-radius: 20px;
            padding: 24px 28px;
            margin-bottom: 24px;
            backdrop-filter: blur(24px);
            box-shadow: 0 12px 40px -10px rgba(0, 0, 0, 0.6), inset 0 1px 2px rgba(255, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
            animation: slideInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .decision-banner::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1.5px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        }
        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .banner-accept {
            background: linear-gradient(145deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .banner-withhold {
            background: linear-gradient(145deg, rgba(244, 63, 94, 0.15) 0%, rgba(159, 18, 57, 0.05) 100%);
            border: 1px solid rgba(244, 63, 94, 0.4);
        }
        .banner-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
            text-transform: uppercase;
        }
        .banner-desc {
            font-size: 0.9rem;
            color: #CBD5E1 !important;
            margin-top: 6px;
            font-weight: 400;
        }

        /* Status Chips */
        .status-chip-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .status-chip-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748B !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-chip-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 30px;
            font-family: 'Fira Code', monospace;
            font-size: 0.95rem;
            font-weight: 700;
            white-space: nowrap !important;
            width: fit-content;
        }
        .chip-green { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #34D399 !important; }
        .chip-amber { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); color: #FBBF24 !important; }
        .chip-red { background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.4); color: #FB7185 !important; }

        /* Next-Gen KPI Tiles */
        .kpi-tile {
            background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: inset 0 1px 1px rgba(255,255,255,0.08);
        }
        .kpi-tile:hover {
            border-color: rgba(45, 212, 191, 0.4);
            transform: translateY(-5px);
            background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
            box-shadow: 0 12px 30px rgba(45, 212, 191, 0.1), inset 0 1px 1px rgba(255,255,255,0.15);
        }
        .kpi-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        .kpi-value {
            font-family: 'Fira Code', monospace;
            font-size: 2.2rem;
            font-weight: 700;
            color: #F8FAFC !important;
            line-height: 1;
            text-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .kpi-progress-track {
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 16px;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.5);
        }
        .kpi-progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* Callout */
        .insight-callout {
            background: linear-gradient(90deg, rgba(45, 212, 191, 0.1) 0%, rgba(45, 212, 191, 0.02) 100%);
            border-left: 4px solid #2DD4BF;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
            font-size: 0.9rem;
            color: #F8FAFC !important;
            font-weight: 400;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        /* Stunning Gradient Button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #8B5CF6 0%, #2DD4BF 100%) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 14px 28px !important;
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            letter-spacing: 0.5px !important;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 40px rgba(45, 212, 191, 0.4) !important;
            filter: brightness(1.1) !important;
        }

        /* Inputs */
        div[data-baseweb="select"] > div {
            background-color: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 12px !important;
            color: #F8FAFC !important;
        }
        div[role="radiogroup"] label { color: #E2E8F0 !important; font-weight: 500 !important; }
        div[data-testid="stSlider"] label { color: #94A3B8 !important; font-weight: 600 !important; }

        /* Footer */
        .neuro-footer {
            text-align: center;
            padding: 32px 0 20px 0;
            color: #475569 !important;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 40px;
            letter-spacing: 0.5px;
        }
    </style>
    """, unsafe_allow_html=True)
'''

lines = lines[:37] + [new_css + '\n'] + lines[449:]
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
