import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ASIN Intelligence Hub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f1419; }
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    /* KPI Cards */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .kpi-value { font-size: 28px; font-weight: 700; color: #f8fafc; }
    .kpi-label { font-size: 12px; color: #94a3b8; margin-top: 4px; }
    .kpi-change-pos { font-size: 11px; color: #6ee7b7; background: #064e3b; 
                      padding: 2px 8px; border-radius: 4px; }
    .kpi-change-neg { font-size: 11px; color: #fca5a5; background: #7f1d1d; 
                      padding: 2px 8px; border-radius: 4px; }
    
    /* Hypothesis Cards */
    .hypothesis-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Badges */
    .badge-critical { background: #7f1d1d; color: #fca5a5; padding: 3px 10px; 
                      border-radius: 12px; font-size: 11px; font-weight: 600; }
    .badge-high { background: #78350f; color: #fcd34d; padding: 3px 10px; 
                  border-radius: 12px; font-size: 11px; font-weight: 600; }
    .badge-medium { background: #1e3a5f; color: #93c5fd; padding: 3px 10px; 
                    border-radius: 12px; font-size: 11px; font-weight: 600; }
    .badge-resolved { background: #064e3b; color: #6ee7b7; padding: 3px 10px; 
                      border-radius: 12px; font-size: 11px; font-weight: 600; }
    
    /* Section Headers */
    .section-header {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    /* Chat styling */
    .chat-user { background: #3b82f6; color: white; padding: 12px 16px; 
                 border-radius: 12px; margin: 8px 0; max-width: 80%; margin-left: auto; }
    .chat-assistant { background: #334155; color: #e7e9ea; padding: 12px 16px; 
                      border-radius: 12px; margin: 8px 0; max-width: 80%; }
    
    div[data-testid="stMetricValue"] { font-size: 24px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1e293b; border-radius: 6px; color: #94a3b8; 
        padding: 8px 20px; border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
    
    /* Pipeline stage cards */
    .pipeline-stage {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        background: #1e293b;
    }
    .pipeline-stage h5 { margin: 0 0 8px 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LAYER
# ============================================================

@st.cache_data
def load_data():
    """Load and preprocess the ASIN datapoints Excel file."""
    df = pd.read_excel('asin_datapoints.xlsx')
    df['ship_day'] = pd.to_datetime(df['ship_day'])
    df['concession_creation_day'] = pd.to_datetime(df['concession_creation_day'], errors='coerce')
    df['ship_month'] = df['ship_month'].str.strip()
    df['warehouse_id'] = df['warehouse_id'].astype(str).str.strip()
    return df


def compute_metrics(df):
    """Compute key investigation metrics."""
    return {
        'total_records': len(df),
        'unique_asins': df['asin'].nunique(),
        'unique_orders': df['order_id'].nunique(),
        'total_sales': round(df['sales'].sum(), 2),
        'total_ncrc': round(df['ncrc'].sum(), 2),
        'total_gcv': round(df['gcv'].sum(), 2),
        'total_shipped_units': int(df['total_shipped_units'].sum()),
        'total_conceded_units': int(df['total_units_conceded'].sum()),
        'concession_rate': round(df['total_units_conceded'].sum() / max(df['total_shipped_units'].sum(), 1) * 100, 2),
        'date_range_start': df['ship_day'].min().strftime('%Y-%m-%d'),
        'date_range_end': df['ship_day'].max().strftime('%Y-%m-%d'),
    }


def compute_defect_distribution(df):
    """Compute defect category distribution."""
    defect_cols = {
        'bdp_cu': 'Bad Detail Page',
        'damage_cu': 'Damage',
        'fit_style_issue_cu': 'Fit/Style Issue',
        'product_defect_cu': 'Product Defect',
        'wrong_item_cu': 'Wrong Item',
        'missing_parts_cu': 'Missing Parts',
        'missing_item_cu': 'Missing Item',
        'extra_item_cu': 'Extra Item',
        'undeliverable_cu': 'Undeliverable',
        'arrived_late_cu': 'Arrived Late',
        'no_longer_wanted_cu': 'No Longer Wanted',
        'item_price_cu': 'Item Price',
        'dnr_cu': 'Delivered Not Received',
        'others_cu': 'Others'
    }
    distribution = {}
    for col, label in defect_cols.items():
        total = int(df[col].sum())
        if total > 0:
            distribution[label] = total
    return distribution


def compute_asin_summary(df):
    """Compute per-ASIN summary."""
    asin_summary = df.groupby('asin').agg(
        brand_name=('brand_name', 'first'),
        product_type=('product_type', 'first'),
        gl=('gl', 'first'),
        fulfillment_channel=('fulfillment_channel', 'first'),
        total_orders=('order_id', 'nunique'),
        total_shipped=('total_shipped_units', 'sum'),
        total_conceded=('total_units_conceded', 'sum'),
        total_sales=('sales', 'sum'),
        total_ncrc=('ncrc', 'sum'),
        star_rating=('star_rating', 'first'),
        total_rating_count=('total_rating_count', 'first'),
        asin_tenure=('asin_tenure_categorization', 'first'),
    ).reset_index()
    asin_summary['concession_rate'] = round(
        asin_summary['total_conceded'] / asin_summary['total_shipped'].clip(lower=1) * 100, 2
    )
    return asin_summary


def generate_rca_hypotheses(df, asin):
    """Generate AI Root Cause Analysis hypotheses for an ASIN."""
    asin_df = df[df['asin'] == asin]
    
    defect_cols = {
        'bdp_cu': 'Bad Detail Page',
        'product_defect_cu': 'Product Defect',
        'wrong_item_cu': 'Wrong Item',
        'arrived_late_cu': 'Arrived Late',
        'no_longer_wanted_cu': 'No Longer Wanted',
        'undeliverable_cu': 'Undeliverable',
        'damage_cu': 'Damage',
        'item_price_cu': 'Item Price',
        'dnr_cu': 'Delivered Not Received',
    }
    
    defect_counts = {}
    for col, label in defect_cols.items():
        count = int(asin_df[col].sum())
        if count > 0:
            defect_counts[label] = count
    
    total_defects = sum(defect_counts.values())
    hypotheses = []
    
    rca_templates = {
        'No Longer Wanted': {
            'category': 'Catalog',
            'hypothesis': 'Product listing may not accurately represent item features, leading to expectation mismatch',
            'evidence': ['High "No Longer Wanted" return rate', 'Customer feedback indicates expectation gap',
                        'Possible listing vs actual product spec deviation'],
            'actions': ['Review product images and descriptions', 'Compare listing vs actual product specs',
                       'Analyze competitor listings', 'Check A+ content accuracy']
        },
        'Product Defect': {
            'category': 'Quality',
            'hypothesis': 'Manufacturing quality issue causing functional defects post-delivery',
            'evidence': ['Multiple "Product Defect" concessions', 'Pattern across multiple shipments',
                        'Vendor quality signals require investigation'],
            'actions': ['Request vendor quality report', 'Check FC inspection data',
                       'Review recent batch/lot changes', 'Analyze defect rate by vendor shipment']
        },
        'Bad Detail Page': {
            'category': 'Catalog',
            'hypothesis': 'Detail page content (images, bullets, A+ content) is misleading or incomplete',
            'evidence': ['Returns linked to "Bad Detail Page"', 'Possible image/description mismatch',
                        'Customer complaints about product expectations'],
            'actions': ['Audit detail page content', 'Check image compliance',
                       'Review A+ content accuracy', 'Compare with customer review themes']
        },
        'Arrived Late': {
            'category': 'Fulfillment',
            'hypothesis': 'Fulfillment or last-mile delivery delays causing missed delivery promises',
            'evidence': ['Late delivery concessions detected', 'Shipping SLA violations',
                        'Carrier performance degradation signals'],
            'actions': ['Review FC-to-customer transit times', 'Check carrier performance',
                       'Evaluate inventory placement', 'Analyze promise vs actual delivery gap']
        },
        'Wrong Item': {
            'category': 'Fulfillment',
            'hypothesis': 'Pick/pack errors at FC leading to incorrect item shipments',
            'evidence': ['Wrong item returns reported', 'Possible bin contamination',
                        'ASIN/FNSKU mapping discrepancy signals'],
            'actions': ['Check FC bin audit history', 'Review ASIN/FNSKU mapping',
                       'Investigate pick errors', 'Request bin contamination check']
        },
        'Undeliverable': {
            'category': 'Fulfillment',
            'hypothesis': 'Package undeliverable due to address issues or carrier failures',
            'evidence': ['Undeliverable concessions detected', 'Potential address validation gaps'],
            'actions': ['Review carrier exception reports', 'Check address validation rules',
                       'Analyze geographic patterns']
        },
        'Item Price': {
            'category': 'Selling Partner',
            'hypothesis': 'Price drop after purchase leading to buyer remorse and returns',
            'evidence': ['Item Price-related returns', 'Possible post-purchase price changes'],
            'actions': ['Review pricing history', 'Check promotional calendar',
                       'Analyze price volatility']
        },
        'Damage': {
            'category': 'Fulfillment',
            'hypothesis': 'Inadequate packaging causing transit damage',
            'evidence': ['Damage-related returns', 'Possible packaging insufficiency'],
            'actions': ['Review ASIN packaging tier', 'Check damage rate by ship method',
                       'Request FC packaging audit']
        },
        'Delivered Not Received': {
            'category': 'Fulfillment',
            'hypothesis': 'Package marked delivered but not received by customer - possible theft or misdelivery',
            'evidence': ['DNR concessions reported', 'Last-mile delivery confirmation issues'],
            'actions': ['Check photo-on-delivery compliance', 'Review carrier delivery accuracy',
                       'Analyze geographic DNR patterns']
        },
    }
    
    for defect, count in sorted(defect_counts.items(), key=lambda x: x[1], reverse=True):
        if defect in rca_templates:
            template = rca_templates[defect]
            hypotheses.append({
                'defect_type': defect,
                'category': template['category'],
                'hypothesis': template['hypothesis'],
                'evidence': template['evidence'],
                'recommended_actions': template['actions'],
                'defect_count': count,
                'percentage': round(count / max(total_defects, 1) * 100, 1)
            })
    
    return hypotheses


def generate_wbr_report(df, accepted_asins):
    """Generate WBR report for accepted ASINs in the format of the Top 10 ASIN Analysis."""
    if not accepted_asins:
        return None
    
    wbr_rows = []
    for rank, asin in enumerate(accepted_asins, 1):
        asin_df = df[df['asin'] == asin]
        if asin_df.empty:
            continue
        
        # Compute metrics
        total_shipped = int(asin_df['total_shipped_units'].sum())
        total_conceded = int(asin_df['total_units_conceded'].sum())
        total_ncrc = asin_df['ncrc'].sum()
        total_sales = asin_df['sales'].sum()
        ret_rate = round(total_conceded / max(total_shipped, 1) * 100, 1)
        cp_per_unit = round(total_sales / max(total_shipped, 1), 2)
        ncrc_per_unit = round(total_ncrc / max(total_shipped, 1), 2)
        
        # Get top defect as root cause theme
        defect_cols = {
            'bdp_cu': 'Bad Detail Page',
            'product_defect_cu': 'Product Defect',
            'wrong_item_cu': 'Wrong Item',
            'arrived_late_cu': 'Arrived Late',
            'no_longer_wanted_cu': 'No Longer Wanted',
            'undeliverable_cu': 'Undeliverable',
            'damage_cu': 'Damage',
            'item_price_cu': 'Item Price',
            'dnr_cu': 'Delivered Not Received',
        }
        defect_counts = {label: int(asin_df[col].sum()) for col, label in defect_cols.items()}
        top_defect = max(defect_counts, key=defect_counts.get) if any(v > 0 for v in defect_counts.values()) else 'Unknown'
        top_defect_pct = round(defect_counts[top_defect] / max(sum(defect_counts.values()), 1) * 100, 0)
        
        # Get ASIN info
        channel = asin_df['fulfillment_channel'].iloc[0]
        gl = asin_df['gl'].iloc[0]
        brand = asin_df['brand_name'].iloc[0]
        
        # Build issue description
        hypotheses = generate_rca_hypotheses(df, asin)
        issue_desc = f"Customer return reason - {int(top_defect_pct)}% returns due to \"{top_defect}\". "
        if hypotheses:
            h = hypotheses[0]
            issue_desc += f"Root Cause - {h['hypothesis']}. "
            issue_desc += f"Corrective Action - {'; '.join(h['recommended_actions'][:2])}. "
        
        # GL-level return rate (average across all ASINs in same GL)
        gl_df = df[df['gl'] == gl]
        gl_shipped = gl_df['total_shipped_units'].sum()
        gl_conceded = gl_df['total_units_conceded'].sum()
        gl_rr = round(gl_conceded / max(gl_shipped, 1) * 100, 1)
        
        wbr_rows.append({
            '#': rank,
            'ASIN': asin,
            'Brand': brand,
            'Channel': channel,
            'Root Cause Theme': top_defect,
            'GL': gl,
            'Return Units': total_conceded,
            'Ship Units': total_shipped,
            'CP/u (T30D)': f"$ {cp_per_unit}",
            'NCRC/u (T30D)': f"$ {ncrc_per_unit}",
            'Ret Rate % (T30D)': f"{ret_rate}%",
            'GL RR% (T30D)': f"{gl_rr}%",
            'Issue Description and Actions Taken': issue_desc
        })
    
    return pd.DataFrame(wbr_rows)


# ============================================================
# INVESTIGATION COPILOT ENGINE
# ============================================================

def investigation_copilot(query, df):
    """Process natural language queries against investigation data."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['return rate', 'concession rate', 'defect rate']):
        metrics = compute_metrics(df)
        return {
            'answer': f"📊 **Overall Concession Rate: {metrics['concession_rate']}%**\n\n"
                      f"- Total conceded units: **{metrics['total_conceded_units']}**\n"
                      f"- Total shipped units: **{metrics['total_shipped_units']:,}**\n"
                      f"- Date range: {metrics['date_range_start']} to {metrics['date_range_end']}",
            'sources': ['ncrc_asin_concessions', 'shipment_data'],
            'next_actions': ['Drill down by ASIN', 'Compare by time period', 'View defect breakdown']
        }
    
    elif any(word in query_lower for word in ['top defect', 'main defect', 'primary defect', 'biggest defect']):
        defects = compute_defect_distribution(df)
        sorted_defects = sorted(defects.items(), key=lambda x: x[1], reverse=True)
        top = sorted_defects[0] if sorted_defects else ('None', 0)
        total = sum(defects.values())
        breakdown = '\n'.join([f"- **{k}**: {v} ({round(v/total*100,1)}%)" for k, v in sorted_defects])
        return {
            'answer': f"🏷️ **Primary Defect: '{top[0]}' with {top[1]} occurrences ({round(top[1]/total*100,1)}%)**\n\n"
                      f"Full defect breakdown:\n{breakdown}",
            'sources': ['defect_classification_engine', 'customer_feedback'],
            'next_actions': ['View RCA for top defect', 'Analyze time trend', 'Compare across ASINs']
        }
    
    elif any(word in query_lower for word in ['why', 'root cause', 'rca', 'reason']):
        for asin in df['asin'].unique():
            if asin.lower() in query_lower:
                hypotheses = generate_rca_hypotheses(df, asin)
                if hypotheses:
                    top_h = hypotheses[0]
                    actions = '\n'.join([f"  - {a}" for a in top_h['recommended_actions']])
                    return {
                        'answer': f"🧠 **Root Cause Analysis for {asin}**\n\n"
                                  f"**Top Hypothesis:**\n"
                                  f"> {top_h['hypothesis']}\n\n"
                                  f"**Category:** {top_h['category']}\n\n"
                                  f"**Evidence:**\n" + '\n'.join([f"- {e}" for e in top_h['evidence']]) + "\n\n"
                                  f"**Recommended Actions:**\n{actions}",
                        'sources': ['ai_rca_engine', 'cross_tool_correlation'],
                        'next_actions': top_h['recommended_actions'][:3]
                    }
        # Generic RCA
        all_hypotheses = []
        for asin in df['asin'].unique():
            all_hypotheses.extend(generate_rca_hypotheses(df, asin))
        if all_hypotheses:
            top_h = all_hypotheses[0]
            return {
                'answer': f"🧠 **Top Root Cause Across All ASINs**\n\n"
                          f"**Hypothesis:**\n"
                          f"> {top_h['hypothesis']}\n\n"
                          f"**Category:** {top_h['category']} | **Defect:** {top_h['defect_type']} ({top_h['defect_count']} occurrences)",
                'sources': ['ai_rca_engine'],
                'next_actions': ['Specify an ASIN for detailed RCA', 'View defect timeline']
            }
    
    elif any(word in query_lower for word in ['geographic', 'state', 'location', 'where', 'region']):
        state_data = df[df['total_units_conceded'] > 0].groupby('state').size().sort_values(ascending=False).head(10)
        result = '\n'.join([f"- **{state}**: {count} concessions" for state, count in state_data.items()])
        return {
            'answer': f"🗺️ **Geographic Distribution of Concessions**\n\nTop states by concession volume:\n{result}",
            'sources': ['shipment_geography_data'],
            'next_actions': ['View by city', 'Analyze ship method correlation', 'Check FC proximity impact']
        }
    
    elif any(word in query_lower for word in ['trend', 'time', 'weekly', 'monthly', 'over time']):
        monthly = df.groupby('ship_month').agg(
            units=('total_shipped_units', 'sum'),
            conceded=('total_units_conceded', 'sum'),
            ncrc=('ncrc', 'sum')
        ).reset_index()
        monthly['rate'] = round(monthly['conceded'] / monthly['units'].clip(lower=1) * 100, 2)
        result = '\n'.join([f"- **{row['ship_month']}**: {row['rate']}% concession rate "
                           f"({row['conceded']} units, ${row['ncrc']:,.2f} NCRC)" 
                           for _, row in monthly.iterrows()])
        return {
            'answer': f"📈 **Monthly Concession Rate Trend**\n\n{result}",
            'sources': ['time_series_analysis'],
            'next_actions': ['View weekly granularity', 'Identify spike events', 'Correlate with promotions']
        }
    
    elif any(word in query_lower for word in ['ncrc', 'cost', 'financial', 'impact', 'dollar', 'money']):
        metrics = compute_metrics(df)
        ncrc_pct = round(metrics['total_ncrc'] / max(metrics['total_sales'], 1) * 100, 2)
        return {
            'answer': f"💰 **Financial Impact Summary**\n\n"
                      f"- **Total NCRC:** ${metrics['total_ncrc']:,.2f}\n"
                      f"- **Total Sales:** ${metrics['total_sales']:,.2f}\n"
                      f"- **NCRC as % of Sales:** {ncrc_pct}%\n"
                      f"- **Total GCV:** ${metrics['total_gcv']:,.2f}\n"
                      f"- **Annualized NCRC Projection:** ~${metrics['total_ncrc'] * 2:,.2f}",
            'sources': ['financial_data', 'ncrc_calculations'],
            'next_actions': ['Break down by defect type', 'Project annual impact', 'View by ASIN']
        }
    
    elif any(word in query_lower for word in ['asin', 'product', 'item', 'summary']):
        summary = compute_asin_summary(df)
        result_parts = []
        for _, row in summary.iterrows():
            result_parts.append(
                f"**{row['asin']}** ({row['brand_name']})\n"
                f"  - Shipped: {row['total_shipped']:,} | Conceded: {row['total_conceded']:,}\n"
                f"  - Concession Rate: {row['concession_rate']}% | NCRC: ${row['total_ncrc']:,.2f}\n"
                f"  - GL: {row['gl']} | Channel: {row['fulfillment_channel']}"
            )
        return {
            'answer': f"📦 **ASIN Investigation Summary**\n\n" + '\n\n'.join(result_parts),
            'sources': ['asin_intelligence_db'],
            'next_actions': ['Deep dive into specific ASIN', 'Compare performance', 'View defect breakdown']
        }
    
    elif any(word in query_lower for word in ['ship', 'carrier', 'delivery', 'fulfillment']):
        ship_method_data = df.groupby('ship_method').agg(
            orders=('order_id', 'nunique'),
            conceded=('total_units_conceded', 'sum')
        ).sort_values('orders', ascending=False).head(5)
        result = '\n'.join([f"- **{idx}**: {row['orders']} orders, {row['conceded']} concessions" 
                           for idx, row in ship_method_data.iterrows() if str(idx) != '-1'])
        return {
            'answer': f"🚚 **Fulfillment & Shipping Analysis**\n\n{result}",
            'sources': ['fulfillment_data', 'carrier_performance'],
            'next_actions': ['Check FC performance', 'Analyze delivery promise accuracy', 'View by warehouse']
        }
    
    elif any(word in query_lower for word in ['warehouse', 'fc', 'facility']):
        wh_data = df[df['warehouse_id'].notna() & (df['warehouse_id'] != '-1') & (df['warehouse_id'] != 'nan')].groupby('warehouse_id').agg(
            orders=('order_id', 'nunique'),
            conceded=('total_units_conceded', 'sum')
        ).sort_values('orders', ascending=False).head(8)
        result = '\n'.join([f"- **{idx}**: {row['orders']} orders, {row['conceded']} concessions" 
                           for idx, row in wh_data.iterrows()])
        return {
            'answer': f"🏭 **Warehouse/FC Analysis**\n\n{result}",
            'sources': ['fc_operations_data'],
            'next_actions': ['Check bin audit history', 'Review pick accuracy', 'Analyze by ship method']
        }
    
    else:
        return {
            'answer': "🤖 I can help you investigate ASIN defects and returns. Try asking about:\n\n"
                      "- **Return/Concession rates** - \"What's the concession rate?\"\n"
                      "- **Top defects** - \"What is the primary defect?\"\n"
                      "- **Root causes** - \"Why are customers returning B0BNC8YC98?\"\n"
                      "- **Geographic patterns** - \"Show geographic distribution\"\n"
                      "- **Time trends** - \"Show monthly concession trend\"\n"
                      "- **Financial impact** - \"What's the NCRC cost?\"\n"
                      "- **ASIN details** - \"Summarize ASINs under investigation\"\n"
                      "- **Fulfillment** - \"Analyze shipping methods\"\n"
                      "- **Warehouses** - \"Which FCs have most issues?\"",
            'sources': [],
            'next_actions': ['What is the top defect?', 'Show me the return rate trend',
                           'Why are customers returning B0BNC8YC98?']
        }


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state(df):
    """Initialize session state for pipeline and scorecard classifications."""
    asin_list = list(df['asin'].unique())
    
    # Investigation Pipeline: classify ASINs into stages
    if 'pipeline_assignments' not in st.session_state:
        st.session_state.pipeline_assignments = {asin: 'Data Extraction' for asin in asin_list}
    else:
        for asin in asin_list:
            if asin not in st.session_state.pipeline_assignments:
                st.session_state.pipeline_assignments[asin] = 'Data Extraction'
    
    # Quality Scorecard: classify ASINs as accepted/maybe/rejected
    if 'scorecard_assignments' not in st.session_state:
        st.session_state.scorecard_assignments = {asin: 'maybe' for asin in asin_list}
    else:
        for asin in asin_list:
            if asin not in st.session_state.scorecard_assignments:
                st.session_state.scorecard_assignments[asin] = 'maybe'


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("⚠️ Please place 'asin_datapoints.xlsx' in the same directory as this script.")
        st.stop()
    
    metrics = compute_metrics(df)
    
    # Initialize session state for pipeline/scorecard
    init_session_state(df)
    

    # ============================================================
    # FILTERS + TABS (Sidebar removed — filters live inside Tab 1 only)
    # ============================================================
    
    all_asins = sorted(df['asin'].unique().tolist())
    all_channels = sorted(df['fulfillment_channel'].unique().tolist())
    
    # Build ASIN-Channel mapping
    asin_to_channels = df.groupby('asin')['fulfillment_channel'].apply(lambda x: list(x.unique())).to_dict()
    channel_to_asins = df.groupby('fulfillment_channel')['asin'].apply(lambda x: list(x.unique())).to_dict()
    st.markdown("## ASIN Intelligence Hub")
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Investigation View", 
        "📋 Program View", 
        "📊 Leadership View",
        "🤖 Investigation Copilot"
    ])
    
    # ==================== TAB 1: INVESTIGATION VIEW ====================
    with tab1:
        st.markdown("### Investigation Workspace")
        
        # --- Inline Filters (only in this tab) ---
        with st.expander("🎯 Filters", expanded=True):
            
            # Initialize session state for linked filters
            if 'filter_asins' not in st.session_state:
                st.session_state.filter_asins = all_asins
            if 'filter_channels' not in st.session_state:
                st.session_state.filter_channels = all_channels
            
            # Cross-filter: available ASINs based on selected channels
            available_asins_from_channels = set()
            for ch in st.session_state.filter_channels:
                if ch in channel_to_asins:
                    available_asins_from_channels.update(channel_to_asins[ch])
            if not available_asins_from_channels:
                available_asins_from_channels = set(all_asins)
            
            # Cross-filter: available channels based on selected ASINs
            available_channels_from_asins = set()
            for a in st.session_state.filter_asins:
                if a in asin_to_channels:
                    available_channels_from_asins.update(asin_to_channels[a])
            if not available_channels_from_asins:
                available_channels_from_asins = set(all_channels)
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                selected_asin = st.multiselect(
                    "ASIN",
                    options=sorted(available_asins_from_channels),
                    default=[a for a in st.session_state.filter_asins if a in available_asins_from_channels],
                    key="asin_filter"
                )
            
            with filter_col2:
                # Recompute available channels based on current ASIN selection
                if selected_asin:
                    channels_for_selected_asins = set()
                    for a in selected_asin:
                        if a in asin_to_channels:
                            channels_for_selected_asins.update(asin_to_channels[a])
                else:
                    channels_for_selected_asins = set(all_channels)
                
                selected_channel = st.multiselect(
                    "Fulfillment Channel",
                    options=sorted(channels_for_selected_asins),
                    default=[c for c in st.session_state.filter_channels if c in channels_for_selected_asins],
                    key="channel_filter"
                )
            
            with filter_col3:
                selected_quarter = st.multiselect(
                    "Quarter", 
                    df['ship_quarter'].unique(), 
                    default=df['ship_quarter'].unique()
                )
            
            # Update session state for cross-filtering on next rerun
            st.session_state.filter_asins = selected_asin if selected_asin else all_asins
            st.session_state.filter_channels = selected_channel if selected_channel else all_channels
        
        # Apply filters (only affects Tab 1)
        filter_asins = selected_asin if selected_asin else all_asins
        filter_channels = selected_channel if selected_channel else all_channels
        
        filtered_df = df[
            (df['asin'].isin(filter_asins)) &
            (df['ship_quarter'].isin(selected_quarter)) &
            (df['fulfillment_channel'].isin(filter_channels))
        ]
        
        # --- Tab 1 info bar ---
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.caption(f"📅 Date Range: {metrics['date_range_start']} → {metrics['date_range_end']}")
        with info_col2:
            st.caption(f"📦 ASINs Shown: {filtered_df['asin'].nunique()} / {metrics['unique_asins']}")
        with info_col3:
            st.caption(f"📊 Records: {len(filtered_df):,} / {metrics['total_records']:,}")
        
        st.markdown("---")
        
        # KPI Row
        filtered_metrics = compute_metrics(filtered_df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Shipped Units", f"{filtered_metrics['total_shipped_units']:,}", 
                     f"{filtered_metrics['unique_asins']} ASINs")
        with col2:
            st.metric("Concession Rate", f"{filtered_metrics['concession_rate']}%", 
                     f"{filtered_metrics['total_conceded_units']} units", delta_color="inverse")
        with col3:
            st.metric("Total NCRC", f"${filtered_metrics['total_ncrc']:,.2f}", 
                     "Net cost of returns", delta_color="inverse")
        with col4:
            st.metric("Total Sales", f"${filtered_metrics['total_sales']:,.2f}", 
                     f"{filtered_metrics['unique_orders']:,} orders")
        
        st.markdown("---")
        
        # ASIN Summary + Defect Distribution
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 📦 ASIN Investigation Summary")
            asin_summary = compute_asin_summary(filtered_df)
            display_df = asin_summary[['asin', 'brand_name', 'gl', 'fulfillment_channel',
                                       'total_shipped', 'total_conceded', 
                                       'concession_rate', 'total_ncrc']].copy()
            display_df.columns = ['ASIN', 'Brand', 'GL', 'Channel', 'Shipped Units', 
                                  'Conceded Units', 'Conc. Rate (%)', 'NCRC ($)']
            display_df['NCRC ($)'] = display_df['NCRC ($)'].round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with col_right:
            st.markdown("#### 🏷️ Defect Classification")
            defects = compute_defect_distribution(filtered_df)
            if defects:
                defect_df = pd.DataFrame(list(defects.items()), columns=['Defect Type', 'Count'])
                defect_df = defect_df.sort_values('Count', ascending=True)
                fig = px.bar(defect_df, x='Count', y='Defect Type', orientation='h',
                           color='Count', color_continuous_scale='Blues')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=300, showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

        # Time-series trend
        st.markdown("#### 📈 Concession Rate Over Time")

        # ASIN filter for this chart
        trend_asins = st.multiselect(
            "Filter by ASIN",
            options=sorted(df['asin'].unique().tolist()),
            default=sorted(df['asin'].unique().tolist()),
            key="trend_asin_filter"
        )

        # Apply ASIN filter
        trend_df = df[df['asin'].isin(trend_asins)] if trend_asins else df

        weekly_trend = trend_df.groupby('ship_week').agg(
            shipped=('total_shipped_units', 'sum'),
            conceded=('total_units_conceded', 'sum'),
            ncrc=('ncrc', 'sum')
        ).reset_index()
        weekly_trend['concession_rate'] = round(
            weekly_trend['conceded'] / weekly_trend['shipped'].clip(lower=1) * 100, 2
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=weekly_trend['ship_week'], y=weekly_trend['concession_rate'],
            mode='lines+markers', name='Concession Rate (%)',
            line=dict(color='#f87171', width=2)
        ))
        fig_trend.add_trace(go.Bar(
            x=weekly_trend['ship_week'], y=weekly_trend['ncrc'],
            name='NCRC ($)', yaxis='y2', marker_color='#3b82f6', opacity=0.3
        ))
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8', height=300,
            xaxis_title='Ship Week', yaxis_title='Concession Rate (%)',
            yaxis2=dict(title='NCRC ($)', overlaying='y', side='right'),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            )
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # RCA Hypotheses
        st.markdown("#### 🧠 AI Root Cause Hypotheses")
        
        all_hypotheses = []
        for asin in filtered_df['asin'].unique():
            all_hypotheses.extend(generate_rca_hypotheses(filtered_df, asin))
        all_hypotheses.sort(key=lambda x: x['defect_count'], reverse=True)
        
        for h in all_hypotheses[:5]:
            with st.expander(f"**{h['defect_type']}** → {h['category']} | "
                           f"Count: {h['defect_count']} ({h['percentage']}% of all defects)"):
                st.markdown(f"**Hypothesis:** {h['hypothesis']}")
                
                col_ev, col_act = st.columns(2)
                with col_ev:
                    st.markdown("**📋 Evidence:**")
                    for ev in h['evidence']:
                        st.markdown(f"- {ev}")
                with col_act:
                    st.markdown("**⚡ Recommended Actions:**")
                    for act in h['recommended_actions']:
                        st.markdown(f"- {act}")
        
        st.markdown("---")
        
        # Concession Reasons
        st.markdown("#### 💬 Top Concession Reasons (Customer Feedback Signals)")
        reason_counts = filtered_df[filtered_df['concession_reason'].notna()]['concession_reason'].value_counts().head(10)
        if not reason_counts.empty:
            fig_reasons = px.bar(
                x=reason_counts.values, y=reason_counts.index,
                orientation='h', labels={'x': 'Count', 'y': 'Concession Reason'}
            )
            fig_reasons.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8', height=300, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_reasons, use_container_width=True)
        
        # Geographic Distribution
        st.markdown("---")
        st.markdown("#### 🗺️ Geographic Distribution")
        col_geo1, col_geo2 = st.columns(2)
        
        with col_geo1:
            st.markdown("##### Orders by State")
            state_data = filtered_df.groupby('state').agg(
                orders=('order_id', 'nunique'),
                conceded=('total_units_conceded', 'sum')
            ).sort_values('orders', ascending=False).head(15).reset_index()
            fig_state = px.bar(state_data, x='state', y='orders', color='conceded',
                             labels={'state': 'State', 'orders': 'Orders', 'conceded': 'Concessions'},
                             color_continuous_scale='Reds')
            fig_state.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8', height=300, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_state, use_container_width=True)
        
        with col_geo2:
            st.markdown("##### Orders by Warehouse")
            valid_wh_df = filtered_df[
                filtered_df['warehouse_id'].notna() & 
                (filtered_df['warehouse_id'] != '-1') & 
                (filtered_df['warehouse_id'] != 'nan') &
                (filtered_df['warehouse_id'] != 'NaT') &
                (filtered_df['warehouse_id'] != '')
            ]
            if not valid_wh_df.empty:
                wh_data = valid_wh_df.groupby('warehouse_id').agg(
                    orders=('order_id', 'nunique')
                ).sort_values('orders', ascending=True).tail(10).reset_index()
                fig_wh = px.bar(
                    wh_data, x='orders', y='warehouse_id', orientation='h',
                    labels={'orders': 'Orders', 'warehouse_id': 'Warehouse'},
                    color='orders', color_continuous_scale='Tealgrn'
                )
                fig_wh.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=300, showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_wh, use_container_width=True)
            else:
                st.info("No valid warehouse data available for selected filters.")
    
    # ==================== TAB 2: PROGRAM VIEW ====================
    with tab2:
        st.markdown("### Program Manager Dashboard")
        
        asin_list = list(df['asin'].unique())
        pipeline_stages = ['Data Extraction', 'SME Validation', 'Resolved']
        scorecard_options = ['accepted', 'maybe', 'rejected']
        
        # --- INVESTIGATION PIPELINE ---
        st.markdown("#### 🔄 Investigation Pipeline")
        
        # ASIN assignment UI
        pipeline_cols = st.columns(len(asin_list))
        for i, asin in enumerate(asin_list):
            with pipeline_cols[i]:
                current_stage = st.session_state.pipeline_assignments.get(asin, 'Data Extraction')
                new_stage = st.selectbox(
                    f"{asin}",
                    options=pipeline_stages,
                    index=pipeline_stages.index(current_stage),
                    key=f"pipeline_{asin}"
                )
                st.session_state.pipeline_assignments[asin] = new_stage
        
        st.markdown("---")
        
        # Display pipeline board
        stage_cols = st.columns(3)
        stage_icons = {'Data Extraction': '🟡', 'SME Validation': '🟣', 'Resolved': '✅'}
        
        for idx, stage in enumerate(pipeline_stages):
            with stage_cols[idx]:
                asins_in_stage = [a for a in asin_list if st.session_state.pipeline_assignments.get(a) == stage]
                st.markdown(f"##### {stage_icons[stage]} {stage} ({len(asins_in_stage)})")
                if asins_in_stage:
                    for a in asins_in_stage:
                        st.markdown(f"<div class='pipeline-stage'><code>{a}</code></div>", 
                                  unsafe_allow_html=True)
                else:
                    st.markdown("_No ASINs in this stage_")
        
        st.markdown("---")
        
        # --- QUALITY SCORECARD (only Resolved ASINs) ---
        st.markdown("#### 📊 Quality Scorecard")
        
        # Only show ASINs that are "Resolved" in the pipeline
        resolved_asins = [a for a in asin_list if st.session_state.pipeline_assignments.get(a) == 'Resolved']
        
        if not resolved_asins:
            st.info("💡 No ASINs are marked as 'Resolved' in the Investigation Pipeline yet. "
                   "Move ASINs to 'Resolved' above to classify their RCA quality here.")
        else:
            st.caption(f"Classify resolved ASINs ({len(resolved_asins)}) as accepted, maybe, or rejected.")
            
            # ASIN quality assignment UI - only for resolved ASINs
            scorecard_cols = st.columns(len(resolved_asins))
            for i, asin in enumerate(resolved_asins):
                with scorecard_cols[i]:
                    current_score = st.session_state.scorecard_assignments.get(asin, 'maybe')
                    new_score = st.selectbox(
                        f"{asin}",
                        options=scorecard_options,
                        index=scorecard_options.index(current_score),
                        key=f"scorecard_{asin}"
                    )
                    st.session_state.scorecard_assignments[asin] = new_score
            
            st.markdown("---")
            
            # Display scorecard summary
            score_cols = st.columns(3)
            score_icons = {'accepted': '✅', 'maybe': '🟡', 'rejected': '❌'}
            
            for idx, score in enumerate(scorecard_options):
                with score_cols[idx]:
                    asins_with_score = [a for a in resolved_asins if st.session_state.scorecard_assignments.get(a) == score]
                    st.markdown(f"##### {score_icons[score]} {score.capitalize()} ({len(asins_with_score)})")
                    if asins_with_score:
                        for a in asins_with_score:
                            st.markdown(f"<div class='pipeline-stage'><code>{a}</code></div>", 
                                      unsafe_allow_html=True)
                    else:
                        st.markdown("_No ASINs_")
            
            # Summary metrics
            st.markdown("---")
            total_resolved = len(resolved_asins)
            accepted_count = sum(1 for a in resolved_asins if st.session_state.scorecard_assignments.get(a) == 'accepted')
            maybe_count = sum(1 for a in resolved_asins if st.session_state.scorecard_assignments.get(a) == 'maybe')
            rejected_count = sum(1 for a in resolved_asins if st.session_state.scorecard_assignments.get(a) == 'rejected')
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                acceptance_rate = round(accepted_count / max(total_resolved, 1) * 100, 0)
                st.metric("Acceptance Rate", f"{int(acceptance_rate)}%", f"{accepted_count}/{total_resolved} ASINs")
            with metric_cols[1]:
                st.metric("Pending Review", f"{maybe_count}", "ASINs marked 'maybe'")
            with metric_cols[2]:
                st.metric("Rejected", f"{rejected_count}", "ASINs needing rework")
        
        st.markdown("---")
        
        # --- WEEKLY COVERAGE TARGET ---
        st.markdown("#### 🎯 Weekly Coverage Target")
        resolved_count = len(resolved_asins) if 'resolved_asins' in dir() else sum(
            1 for a in asin_list if st.session_state.pipeline_assignments.get(a) == 'Resolved')
        target_count = 100
        
        st.progress(min(resolved_count / target_count, 1.0))
        st.caption(f"**{resolved_count} / {target_count}** ASINs resolved this week (Target: 10x coverage)")
        
        if resolved_count == 0:
            st.info("💡 Move ASINs to 'Resolved' in the Investigation Pipeline above to track coverage progress.")
        elif resolved_count < len(asin_list):
            st.info(f"💡 {len(asin_list) - resolved_count} ASINs still in progress. "
                   f"Goal: Scale from ~10 to ~100 ASINs/week with AI assistance.")
        else:
            st.success(f"🎉 All {len(asin_list)} ASINs under investigation have been resolved!")
        
        st.markdown("---")
        
        # --- WBR REPORT SECTION ---
        st.markdown("#### 📄 WBR")
        
        # Get accepted ASINs
        accepted_asins = [a for a in asin_list 
                         if st.session_state.pipeline_assignments.get(a) == 'Resolved'
                         and st.session_state.scorecard_assignments.get(a) == 'accepted']
        
        if not accepted_asins:
            st.info("💡 No ASINs have been accepted yet. Mark ASINs as 'Resolved' in the Pipeline, "
                   "then classify them as 'accepted' in the Quality Scorecard to generate the WBR.")
        else:
            st.success(f"✅ **{len(accepted_asins)} ASIN(s)** ready for WBR")
            
            # Generate WBR report
            wbr_df = generate_wbr_report(df, accepted_asins)
            
            if wbr_df is not None and not wbr_df.empty:

                
                # Download as Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    wbr_df.to_excel(writer, index=False, sheet_name='WBR')
                    
                    # Auto-adjust column widths
                    worksheet = writer.sheets['WBR']
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                
                output.seek(0)
                
                st.download_button(
                    label="📥 Download WBR Report (Excel)",
                    data=output,
                    file_name="WBR.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # ==================== TAB 3: LEADERSHIP VIEW ====================
    with tab3:
        st.markdown("### Leadership Dashboard")
        
        # Top KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Investigation TAT", "3.2 days", "-77% (was 14 days)")
        with col2:
            resolved_count_lead = sum(1 for a in list(df['asin'].unique()) 
                               if st.session_state.pipeline_assignments.get(a) == 'Resolved')
            st.metric("Weekly Coverage", f"{resolved_count_lead} ASINs")
        with col3:
            st.metric("NCRC Under Investigation", f"${metrics['total_ncrc']:,.0f}", "Active")
        with col4:
            st.metric("Analyst Efficiency", "72%", "40→11 hrs saved")
        
        st.markdown("---")
        
        # Systemic Gaps + Escalations
        col_sys, col_esc = st.columns(2)
        
        with col_sys:
            st.markdown("#### 🔍 Systemic Gap Trends")
            gaps_data = pd.DataFrame({
                'Gap Category': ['Buyer Remorse / Expectation Mismatch', 'Quality Defects', 
                               'Catalog Content Gaps', 'Fulfillment Delays'],
                'Affected Units': [66, 6, 5, 5],
                'Trend': ['↑ Increasing', '→ Stable', '→ Stable', '↓ Improving']
            })
            st.dataframe(gaps_data, use_container_width=True, hide_index=True)
            
            fig_gaps = px.pie(gaps_data, values='Affected Units', names='Gap Category',
                            color_discrete_sequence=['mediumpurple', 'cornflowerblue', 'tomato', 'gold'])
            fig_gaps.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8', height=250, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_gaps, use_container_width=True)
        
        with col_esc:
            st.markdown("#### 🚨 Escalation Queue")
            escalations = pd.DataFrame({
                'ASIN': ['B0BNC8YC98', 'B0F7HBYMGL'],
                'Reason': ['Vendor non-responsive (>7 days)', 'High volume - NLW spike'],
                'Age (days)': [8, 3],
                'Priority': ['Critical', 'High']
            })
            st.dataframe(escalations, use_container_width=True, hide_index=True)
            
            st.warning("⚠️ **1 item** past SLA - requires immediate attention")
            st.markdown("---")
            st.markdown("#### 📊 Program Impact Summary")
            st.markdown("_$10.4B Annual Returns Cost Opportunity_")
            
            impact_col1, impact_col2, impact_col3 = st.columns(3)
            with impact_col1:
                st.markdown("<div style='text-align:center'><h2 style='color:#4ade80'>72%</h2>"
                          "<p style='color:#94a3b8;font-size:12px'>Analyst Time Reduction</p></div>",
                          unsafe_allow_html=True)
            with impact_col2:
                st.markdown("<div style='text-align:center'><h2 style='color:cornflowerblue'>10x</h2>"
                          "<p style='color:#94a3b8;font-size:12px'>Coverage Capacity Target</p></div>",
                          unsafe_allow_html=True)
            with impact_col3:
                st.markdown("<div style='text-align:center'><h2 style='color:#fbbf24'>77%</h2>"
                          "<p style='color:#94a3b8;font-size:12px'>TAT Reduction</p></div>",
                          unsafe_allow_html=True)
        
        st.markdown("---")
    
    # ==================== TAB 4: INVESTIGATION COPILOT ====================
    with tab4:
        st.markdown("### 🤖 Investigation Copilot")
        st.markdown("Natural-language interface for evidence retrieval, hypothesis testing, "
                   "and guided investigation. Ask questions about defects, returns, root causes, and more.")
        st.markdown("---")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Suggested queries
        st.markdown("**Quick queries:**")
        quick_cols = st.columns(5)
        quick_queries = [
            "What is the top defect?",
            "Show financial impact",
            "Why are customers returning B0BNC8YC98?",
            "Show monthly trend",
            "Geographic patterns"
        ]
        
        for i, (col, query_text) in enumerate(zip(quick_cols, quick_queries)):
            with col:
                if st.button(query_text, key=f"quick_{i}"):
                    st.session_state.chat_history.append({'role': 'user', 'content': query_text})
                    result = investigation_copilot(query_text, filtered_df)
                    response = result['answer']
                    if result['sources']:
                        response += f"\n\n---\n_Sources: {', '.join(result['sources'])}_"
                    if result['next_actions']:
                        response += f"\n\n**Suggested next:** {' • '.join(result['next_actions'])}"
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        
        st.markdown("---")
        
        # Chat display
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your ASIN investigation..."):
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            result = investigation_copilot(prompt, filtered_df)
            response = result['answer']
            if result['sources']:
                response += f"\n\n---\n_Sources: {', '.join(result['sources'])}_"
            if result['next_actions']:
                response += f"\n\n**Suggested next:** {' • '.join(result['next_actions'])}"
            
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            with st.chat_message("assistant"):
                st.markdown(response)
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
