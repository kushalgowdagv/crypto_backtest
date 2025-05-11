import os
import logging
from backtest_lib.analysis.analysis import Analysis

# Get logger
logger = logging.getLogger('backtest_lib')




class Report:
    """
    Generate HTML and PDF reports for backtest results
    """
    
    def __init__(self, analysis: Analysis):
        """
        Initialize the reporting module
        
        Parameters:
        -----------
        analysis : Analysis
            Analysis instance
        """
        self.analysis = analysis
        self.report_data = None
    
    def generate_html_report(self, output_file: str, title: str = None) -> str:
        """
        Generate HTML report
        
        Parameters:
        -----------
        output_file : str
            Output file path
        title : str, optional
            Report title
            
        Returns:
        --------
        str
            Path to the HTML file
        """
        # Get report data
        if self.report_data is None:
            self.report_data = self.analysis.generate_performance_report()
        
        if title is None:
            title = f"Backtest Report: {self.analysis.strategy_name}"
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .metric-card {{
                    flex: 1;
                    min-width: 200px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .chart-container {{
                    margin: 30px 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {self.report_data['timestamp']}</p>
            
            <h2>Performance Metrics</h2>
            <div class="metrics-container">
        """
        
        # Add performance metrics
        metrics = self.report_data['metrics']
        
        # Format Total Return
        total_return = metrics.get('total_return', 0) * 100
        total_return_class = 'positive' if total_return >= 0 else 'negative'
        html_content += f"""
                <div class="metric-card">
                    <h3>Total Return</h3>
                    <div class="metric-value {total_return_class}">{total_return:.2f}%</div>
                </div>
        """
        
        # Format Annual Return
        annual_return = metrics.get('annual_return', 0) * 100
        annual_return_class = 'positive' if annual_return >= 0 else 'negative'
        html_content += f"""
                <div class="metric-card">
                    <h3>Annual Return</h3>
                    <div class="metric-value {annual_return_class}">{annual_return:.2f}%</div>
                </div>
        """
        
        # Format Sharpe Ratio
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        sharpe_class = 'positive' if sharpe_ratio >= 1 else 'negative'
        html_content += f"""
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <div class="metric-value {sharpe_class}">{sharpe_ratio:.2f}</div>
                </div>
        """
        
        # Format Max Drawdown
        max_drawdown = metrics.get('max_drawdown', 0) * 100
        html_content += f"""
                <div class="metric-card">
                    <h3>Max Drawdown</h3>
                    <div class="metric-value negative">{max_drawdown:.2f}%</div>
                </div>
        """
        
        # Format Win Rate
        win_rate = metrics.get('win_rate', 0) * 100
        win_rate_class = 'positive' if win_rate >= 50 else 'negative'
        html_content += f"""
                <div class="metric-card">
                    <h3>Win Rate</h3>
                    <div class="metric-value {win_rate_class}">{win_rate:.2f}%</div>
                </div>
        """
        
        # Format Number of Trades
        num_trades = metrics.get('num_trades', 0)
        html_content += f"""
                <div class="metric-card">
                    <h3>Number of Trades</h3>
                    <div class="metric-value">{num_trades}</div>
                </div>
            </div>
        """
        
        # Add charts
        html_content += """
            <h2>Performance Charts</h2>
        """
        
        for fig_path in self.report_data['figures']:
            fig_name = os.path.basename(fig_path)
            fig_display_name = fig_name.split('_')[-1].split('.')[0].replace('_', ' ').title()
            
            html_content += f"""
            <div class="chart-container">
                <h3>{fig_display_name}</h3>
                <img src="{fig_path}" alt="{fig_display_name}">
            </div>
            """
        
        # Add trade table
        html_content += """
            <h2>Trades</h2>
            <table>
                <tr>
                    <th>Trade ID</th>
                    <th>Entry Date</th>
                    <th>Exit Date</th>
                    <th>Position Type</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>PnL (%)</th>
                </tr>
        """
        
        # Add trade rows
        for trade in self.report_data['trades']:
            trade_id = trade.get('trade_id', '')
            entry_date = trade.get('entry_date', '')
            exit_date = trade.get('exit_date', 'Open')
            position_type = trade.get('position_type', '')
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            pnl = trade.get('pnl', 0) * 100
            pnl_class = 'positive' if pnl >= 0 else 'negative'
            
            html_content += f"""
                <tr>
                    <td>{trade_id}</td>
                    <td>{entry_date}</td>
                    <td>{exit_date}</td>
                    <td>{position_type}</td>
                    <td>${entry_price:.2f}</td>
                    <td>${exit_price:.2f if exit_price else ''}</td>
                    <td class="{pnl_class}">{pnl:.2f}% if pnl else ''</td>
                </tr>
            """
        
        # Close table and HTML
        html_content += """
            </table>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_file}")
        
        return output_file
    
    def generate_pdf_report(self, output_file: str, title: str = None) -> str:
        """
        Generate PDF report
        
        Parameters:
        -----------
        output_file : str
            Output file path
        title : str, optional
            Report title
            
        Returns:
        --------
        str
            Path to the PDF file
        """
        try:
            # First generate HTML report
            html_file = output_file.replace('.pdf', '.html')
            self.generate_html_report(html_file, title)
            
            # Convert HTML to PDF using wkhtmltopdf
            from weasyprint import HTML
            
            HTML(html_file).write_pdf(output_file)
            
            logger.info(f"PDF report saved to {output_file}")
            return output_file
            
        except ImportError:
            logger.error("weasyprint not installed. Please install it to generate PDF reports.")
            raise ImportError("weasyprint not installed. Please install it to generate PDF reports.")
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise