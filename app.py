import streamlit as st
import pandas as pd
from russian_ifrs_analyzer import RussianIFRSAnalyzer
import tempfile
import plotly.graph_objects as go


st.set_page_config(
    page_title="Russian IFRS Analyzer",
    page_icon="📊",
    layout="wide"
)

def create_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create a comparison bar chart for current and previous year figures."""
    fig = go.Figure(data=[
        go.Bar(name='Current Year', x=df['Metric'], y=df['Current Year'].fillna(0)),
        go.Bar(name='Previous Year', x=df['Metric'], y=df['Previous Year'].fillna(0))
    ])
    
    fig.update_layout(
        title='Financial Metrics Comparison',
        barmode='group',
        height=500,
        xaxis_title="Metrics",
        yaxis_title="Value (in millions)",
        template="plotly_white"
    )
    return fig

def main():
    st.title("🎯 R. IFRS Statement Analyzer v.1.4")
    
    st.markdown("""
    This app analyzes Russian IFRS financial statements and extracts key financial metrics.
    Upload your PDF file to get started.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload IFRS Statement (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # Show progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Update progress
            progress_bar.progress(20)
            status_text.text("PDF uploaded successfully. Initializing analysis...")
            
            # Initialize analyzer with OpenAI API key from Streamlit secrets
            analyzer = RussianIFRSAnalyzer(
                pdf_path=tmp_file_path,
                openai_api_key=st.secrets["openai_api_key"]
            )
            
            # Update progress
            progress_bar.progress(40)
            status_text.text("Extracting text from PDF...")
            
            # Extract text
            extracted_text = analyzer.extract_pdf_text()
            
            # Create a formatted version of the text with page numbers
            formatted_text = ""
            for i, page_text in enumerate(extracted_text, 1):
                formatted_text += f"\n=== Page {i} ===\n{page_text}\n"
            
            # Download buttons side by side
            col1, col2 = st.columns(2)
            
            with col1:
                # Download button for extracted text
                st.download_button(
                    label="Download Extracted Text",
                    data=formatted_text.encode('utf-8'),
                    file_name="extracted_text.txt",
                    mime="text/plain",
                )


            # Extract text
            analyzer.extract_pdf_text()
            progress_bar.progress(60)
            status_text.text("Creating vector store...")
            
            # Create vector store
            analyzer.create_vector_store()
            progress_bar.progress(80)
            status_text.text("Analyzing statements...")
            
            # Analyze statements
            results_df = analyzer.analyze_statements()
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            

            with col2:
                # Download button for results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="ifrs_analysis_results.csv",
                    mime="text/csv",
                )

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Results Table", "📈 Visualization", "📋 Key Ratios"])
            
            with tab1:
                st.subheader("Extracted Financial Metrics")
                
                # Format the results
                formatted_df = results_df.copy()
                formatted_df['Current Year'] = formatted_df['Current Year'].apply(
                    lambda x: f"{x:,.2f}M" if pd.notnull(x) else "N/A"
                )
                formatted_df['Previous Year'] = formatted_df['Previous Year'].apply(
                    lambda x: f"{x:,.2f}M" if pd.notnull(x) else "N/A"
                )
                
                st.dataframe(
                    formatted_df.style.set_properties(**{
                        'background-color': 'lightgrey',
                        'color': 'black'
                    }),
                    use_container_width=True
                )
                
                
                

            with tab2:
                st.subheader("Visual Comparison")
                # Create and display comparison chart
                fig = create_comparison_chart(results_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Key Financial Ratios")
                
                # Calculate key ratios
                current_ratios = {}
                previous_ratios = {}
                
                # Get raw values for calculations
                def get_value(metric, year='current'):
                    row = results_df[results_df['Metric'] == metric]
                    if not row.empty:
                        return row[f'{year.title()} Year'].iloc[0]
                    return None
                
                # Calculate ratios for both years
                for year, ratios in [('current', current_ratios), ('previous', previous_ratios)]:
                    ebitda = get_value('ebitda', year)
                    total_debt = get_value('total_debt', year)
                    interest_expense = get_value('interest_expense', year)
                    
                    if all(v is not None for v in [total_debt, ebitda]):
                        ratios['Debt/EBITDA'] = total_debt / ebitda if ebitda != 0 else None
                    
                    if all(v is not None for v in [ebitda, interest_expense]):
                        ratios['Interest Coverage'] = ebitda / interest_expense if interest_expense != 0 else None
                
                # Display ratios
                ratios_df = pd.DataFrame({
                    'Ratio': list(current_ratios.keys()),
                    'Current Year': list(current_ratios.values()),
                    'Previous Year': list(previous_ratios.values())
                })
                
                # Format ratios
                ratios_df['Current Year'] = ratios_df['Current Year'].apply(
                    lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A"
                )
                ratios_df['Previous Year'] = ratios_df['Previous Year'].apply(
                    lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A"
                )
                
                st.dataframe(ratios_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()