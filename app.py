import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from russian_financial_extractor import RussianFinancialExtractor
import tempfile
from russian_ifrs_analyzer import RussianIFRSAnalyzer

st.set_page_config(
    page_title="Russian IFRS Analyzer v.2.0",
    page_icon="📊",
    layout="wide"
)

def create_comparison_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """Create a comparison chart for a specific metric across periods."""
    fig = go.Figure()
    
    # Add a bar for each period
    for period_type in df['period_type'].unique():
        period_data = df[df['period_type'] == period_type]
        fig.add_trace(go.Bar(
            name=f'{period_type}',
            x=period_data['year'].astype(str),
            y=period_data[metric],
            text=period_data[metric].apply(lambda x: f'{x:,.0f}M'),
            textposition='auto',
        ))
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} by Period',
        barmode='group',
        height=500,
        xaxis_title="Year",
        yaxis_title="Value (in millions)",
        template="plotly_white"
    )
    return fig

def main():
    st.title("🎯 R. IFRS Statement Analyzer v.2.21")
    
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

            # Initialize analyzer with OpenAI API key from Streamlit secrets
            analyzer = RussianIFRSAnalyzer(
                pdf_path=tmp_file_path,
                openai_api_key=st.secrets["openai_api_key"]
            )
            
            results_df = analyzer.analyze_statements()

            # Format results for display
            formatted_df = results_df.copy()
            formatted_df['Value'] = formatted_df['Value'].apply(RussianIFRSAnalyzer.format_value)

            # Display in Streamlit
            st.dataframe(
                formatted_df.sort_values(['Date', 'Metric']),
                use_container_width=True
)

            #extracted_text = analyzer.extract_pdf_text()
            
            # Create a formatted version of the text with page numbers
            #formatted_text = ""
            #for i, page_text in enumerate(extracted_text, 1):
            #    formatted_text += f"\n=== Page {i} ===\n{page_text}\n"


            progress_bar.progress(40)
            status_text.text("Extracting financial data...")
            
            # Initialize and use the financial extractor
            extractor = RussianFinancialExtractor()
            results = extractor.process_document(extracted_text)
            
            progress_bar.progress(80)
            status_text.text("Formatting results...")
            
            # Convert results to DataFrame
            results_df = extractor.format_results(results)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Visualizations", "📋 Key Ratios"])
            
            with tab1:
                st.subheader("Extracted Financial Metrics")
                st.dataframe(results_df)
                
                # Download button for results
                csv = results_df.to_csv(index=False).encode('utf-8')
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
                    options=[col for col in results_df.columns if col not in ['period_type', 'year', 'end_date']]
                )
                fig = create_comparison_chart(results_df, metric)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Key Financial Ratios")
                for idx, row in results_df.iterrows():
                    period = f"{row['period_type']} {row['year']}"
                    st.write(f"### {period}")
                    
                    # Calculate and display ratios
                    if 'bs_total_assets' in row and row['bs_total_assets'] != 0:
                        st.write(f"ROA: {row.get('is_net_profit', 0) / row['bs_total_assets']:.2%}")
                    
                    if 'bs_total_equity' in row and row['bs_total_equity'] != 0:
                        st.write(f"ROE: {row.get('is_net_profit', 0) / row['bs_total_equity']:.2%}")
                    
                    if 'bs_current_liabilities' in row and row['bs_current_liabilities'] != 0:
                        current_ratio = row.get('bs_current_assets', 0) / row['bs_current_liabilities']
                        st.write(f"Current Ratio: {current_ratio:.2f}")
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()