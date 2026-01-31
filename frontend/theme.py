import streamlit as st

THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Work+Sans:wght@400;500;600&display=swap');

    :root {
        --ink: #e5e7eb;
        --muted: #9ca3af;
        --brand: #22c55e;
        --brand-2: #38bdf8;
        --accent: #f59e0b;
        --bg: #0b0f14;
        --bg-2: #111827;
        --card: #111827;
        --border: #1f2937;
        --shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }

    html, body, [class*="css"]  {
        font-family: 'Work Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(900px 360px at 5% -10%, rgba(34, 197, 94, 0.18), transparent 60%),
            radial-gradient(900px 360px at 100% 0%, rgba(56, 189, 248, 0.16), transparent 60%),
            linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
    }

    h1, h2, h3, h4, h5, h6, p, span, li {
        color: var(--ink);
    }

    .hero {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.16), rgba(56, 189, 248, 0.16));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 28px 32px;
        box-shadow: var(--shadow);
        margin-bottom: 24px;
        animation: rise 0.6s ease-out;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        line-height: 1.1;
        margin: 0 0 6px 0;
        color: var(--ink);
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        margin: 0;
    }
    .hero-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .badge {
        background: rgba(34, 197, 94, 0.12);
        color: var(--brand);
        border: 1px solid rgba(34, 197, 94, 0.35);
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 999px;
        letter-spacing: 0.02em;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid var(--border);
        background: rgba(17, 24, 39, 0.8);
    }
    .status-pill .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #9ca3af;
    }
    .status-online .dot {
        background: #22c55e;
        box-shadow: 0 0 0 6px rgba(34, 197, 94, 0.12);
    }
    .status-offline .dot {
        background: #ef4444;
        box-shadow: 0 0 0 6px rgba(239, 68, 68, 0.12);
    }

    .section-card {
        background: rgba(17, 24, 39, 0.92);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: var(--shadow);
        animation: fadeIn 0.6s ease-out;
    }

    div[data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.92);
        border: 1px solid var(--border);
        padding: 16px;
        border-radius: 14px;
        box-shadow: var(--shadow);
    }
    div[data-testid="metric-container"] > div {
        color: var(--ink);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding-left: 18px;
        padding-right: 18px;
        background-color: rgba(17, 24, 39, 0.9);
        border: 1px solid var(--border);
        border-radius: 12px 12px 0 0;
        color: var(--ink);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f766e;
        color: #ffffff;
        border-color: #0f766e;
    }

    .stButton > button {
        background: #0f766e;
        color: #ffffff;
        border: 1px solid #0f766e;
        border-radius: 12px;
        padding: 0.4rem 0.9rem;
        font-weight: 600;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(17, 24, 39, 0.9);
        color: var(--ink);
        border: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 12, 18, 0.96);
        border-right: 1px solid var(--border);
    }

    @keyframes rise {
        from { transform: translateY(8px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
</style>
"""


def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_hero(title, subtitle, badge, status_text, status_class):
    hero_html = f"""
    <div class=\"hero\">
        <div class=\"hero-row\">
            <div class=\"badge\">{badge}</div>
            <div class=\"status-pill {status_class}\">
                <span class=\"dot\"></span>
                <span>{status_text}</span>
            </div>
        </div>
        <div class=\"hero-title\">{title}</div>
        <p class=\"hero-subtitle\">{subtitle}</p>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
