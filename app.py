import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from russian_ifrs_analyzer import RussianIFRSAnalyzer
import tempfile

st.set_page_config(
    page_title="Russian IFRS Analyzer v.2.0",
    page_icon="📊",
    layout="wide"
)

def create_comparison_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """Create a comparison chart comparing reported and comparative values."""
    fig = go.Figure()
    
    metric_date = df[df['Metric'] == metric]['Date'].iloc[0]
    comp_date = df[df['Metric'] == metric]['Comparative Date'].iloc[0]
    
    # Add bars for reported and comparative values
    values = df[df['Metric'] == metric]
    fig.add_trace(go.Bar(
        name=f'Reported ({metric_date})',
        x=[metric],
        y=[values['Value'].iloc[0]],
        text=[f'{values["Value"].iloc[0]:,.0f}M'],
        textposition='auto',
    ))
    
    if pd.notna(comp_date):  # Only add comparative if it exists
        fig.add_trace(go.Bar(
            name=f'Comparative ({comp_date})',
            x=[metric],
            y=[values['Comparative Value'].iloc[0]],
            text=[f'{values["Comparative Value"].iloc[0]:,.0f}M'],
            textposition='auto',
        ))
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()}',
        barmode='group',
        height=500,
        yaxis_title="Value (in millions)",
        template="plotly_white"
    )
    return fig

def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate key financial ratios."""
    ratios = []
    
    # Get the main reporting date
    report_date = df['Date'].iloc[0]
    
    # Get values for ratio calculations
    values = df.set_index('Metric')['Value']
    
    # Calculate ratios where possible
    if 'total_assets' in values and 'net_profit' in values and values['total_assets'] != 0:
        ratios.append({
            'Ratio': 'ROA',
            'Value': values['net_profit'] / values['total_assets'],
            'Date': report_date
        })
    
    if 'total_equity' in values and 'net_profit' in values and values['total_equity'] != 0:
        ratios.append({
            'Ratio': 'ROE',
            'Value': values['net_profit'] / values['total_equity'],
            'Date': report_date
        })
    
    if 'total_debt' in values and 'ebitda' in values and values['ebitda'] != 0:
        ratios.append({
            'Ratio': 'Debt/EBITDA',
            'Value': values['total_debt'] / values['ebitda'],
            'Date': report_date
        })
    
    return pd.DataFrame(ratios)

def main():
    st.title("🎯 R. IFRS Statement Analyzer v.2.29")
    
    st.markdown("""
    This app analyzes Russian IFRS financial statements and extracts key financial metrics.
    Upload your PDF file to get started.
    """)
    
    uploaded_file = st.file_uploader("Upload IFRS Statement (PDF)", type="pdf")
    
    if uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            progress_bar.progress(20)
            status_text.text("PDF uploaded successfully. Initializing analysis...")

            # Initialize analyzer and process statements
            analyzer = RussianIFRSAnalyzer(
                pdf_path=tmp_file_path,
                openai_api_key=st.secrets["openai_api_key"]
            )

            progress_bar.progress(40)
            status_text.text("Analyzing financial statements...")
            
            results_df = analyzer.analyze_statements()

            progress_bar.progress(80)
            status_text.text("Formatting results...")

            # Format results for display
            formatted_df = results_df.copy()
            formatted_df['Value'] = formatted_df['Value'].apply(RussianIFRSAnalyzer.format_value)
            formatted_df['Comparative Value'] = formatted_df['Comparative Value'].apply(RussianIFRSAnalyzer.format_value)
            
            # Format dates for display
            formatted_df['Date'] = pd.to_datetime(formatted_df['Date']).dt.strftime('%Y-%m-%d')
            formatted_df['Comparative Date'] = pd.to_datetime(formatted_df['Comparative Date']).dt.strftime('%Y-%m-%d')
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Visualizations", "📋 Key Ratios"])
            
            with tab1:
                st.subheader("Extracted Financial Metrics")
                st.dataframe(formatted_df, use_container_width=True)
                
                # Download button for results
                csv = formatted_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="ifrs_analysis_results.csv",
                    mime="text/csv",
                )
            
            with tab2:
                st.subheader("Financial Metrics Visualization")
                metric = st.selectbox(
                    "Select metric to visualize",
                    options=results_df['Metric'].unique()
                )
                fig = create_comparison_chart(results_df, metric)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Key Financial Ratios")
                ratios_df = calculate_ratios(results_df)
                
                if not ratios_df.empty:
                    # Format ratio values for display
                    formatted_ratios = ratios_df.copy()
                    formatted_ratios['Value'] = formatted_ratios['Value'].apply(lambda x: f"{x:.2f}x")
                    st.dataframe(formatted_ratios, use_container_width=True)
                else:
                    st.write("Unable to calculate ratios due to missing or invalid data")
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()