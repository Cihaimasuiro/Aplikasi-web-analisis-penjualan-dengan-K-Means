import streamlit as st


def apply_dark_mode():
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f172a;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #243044;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        .analytics-card {
            background: #162033;
            border: 1px solid #2f3b52;
            border-radius: 8px;
            padding: 16px;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: #e5e7eb;
        }
        .muted {
            color: #94a3b8;
        }
        .sidebar-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .sidebar-link {
            display: block;
            padding: 8px 10px;
            margin: 4px 0;
            border-radius: 8px;
            color: #dbeafe !important;
            text-decoration: none;
            background: #172238;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-title">Customer Analytics</div>', unsafe_allow_html=True)
    st.sidebar.caption("Segmentasi pelanggan berbasis K-Means")
    st.sidebar.markdown(
        """
        <a class="sidebar-link" href="/" target="_self">Dashboard</a>
        <a class="sidebar-link" href="/upload" target="_self">Upload Dataset</a>
        <a class="sidebar-link" href="/preprocessing" target="_self">Preprocessing</a>
        <a class="sidebar-link" href="/clustering" target="_self">Clustering</a>
        <a class="sidebar-link" href="/detail_clustering" target="_self">Detail Hasil</a>
        <a class="sidebar-link" href="/evaluation" target="_self">Evaluation</a>
        <a class="sidebar-link" href="/visualization" target="_self">Visualization</a>
        """,
        unsafe_allow_html=True,
    )


def setup_page():
    apply_dark_mode()
    render_sidebar()
