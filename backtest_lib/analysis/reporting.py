# import os
# import logging
# from backtest_lib.analysis.analysis import Analysis

# # Get logger
# logger = logging.getLogger('backtest_lib')




# class Report:
#     """
#     Generate HTML and PDF reports for backtest results
#     """
    
#     def __init__(self, analysis: Analysis):
#         """
#         Initialize the reporting module
        
#         Parameters:
#         -----------
#         analysis : Analysis
#             Analysis instance
#         """
#         self.analysis = analysis
#         self.report_data = None
    
#     def generate_html_report(self, output_file: str, title: str = None) -> str:
#         """
#         Generate HTML report
        
#         Parameters:
#         -----------
#         output_file : str
#             Output file path
#         title : str, optional
#             Report title
            
#         Returns:
#         --------
#         str
#             Path to the HTML file
#         """
#         # Get report data
#         if self.report_data is None:
#             self.report_data = self.analysis.generate_performance_report()
        
#         if title is None:
#             title = f"Backtest Report: {self.analysis.strategy_name}"
        
#         # Create HTML content
#         html_content = f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>{title}</title>
#             <style>
#                 body {{
#                     font-family: Arial, sans-serif;
#                     line-height: 1.6;
#                     max-width: 1200px;
#                     margin: 0 auto;
#                     padding: 20px;
#                 }}
#                 h1, h2, h3 {{
#                     color: #333;
#                 }}
#                 table {{
#                     border-collapse: collapse;
#                     width: 100%;
#                     margin-bottom: 20px;
#                 }}
#                 th, td {{
#                     border: 1px solid #ddd;
#                     padding: 8px;
#                     text-align: left;
#                 }}
#                 th {{
#                     background-color: #f2f2f2;
#                 }}
#                 tr:nth-child(even) {{
#                     background-color: #f9f9f9;
#                 }}
#                 .metrics-container {{
#                     display: flex;
#                     flex-wrap: wrap;
#                     gap: 20px;
#                 }}
#                 .metric-card {{
#                     flex: 1;
#                     min-width: 200px;
#                     border: 1px solid #ddd;
#                     border-radius: 5px;
#                     padding: 15px;
#                     box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#                 }}
#                 .metric-value {{
#                     font-size: 24px;
#                     font-weight: bold;
#                     margin: 10px 0;
#                 }}
#                 .positive {{
#                     color: green;
#                 }}
#                 .negative {{
#                     color: red;
#                 }}
#                 .chart-container {{
#                     margin: 30px 0;
#                 }}
#                 img {{
#                     max-width: 100%;
#                     height: auto;
#                 }}
#             </style>
#         </head>
#         <body>
#             <h1>{title}</h1>
#             <p>Generated on: {self.report_data['timestamp']}</p>
            
#             <h2>Performance Metrics</h2>
#             <div class="metrics-container">
#         """
        
#         # Add performance metrics
#         metrics = self.report_data['metrics']
        
#         # Format Total Return
#         total_return = metrics.get('total_return', 0) * 100
#         total_return_class = 'positive' if total_return >= 0 else 'negative'
#         html_content += f"""
#                 <div class="metric-card">
#                     <h3>Total Return</h3>
#                     <div class="metric-value {total_return_class}">{total_return:.2f}%</div>
#                 </div>
#         """
        
#         # Format Annual Return
#         annual_return = metrics.get('annual_return', 0) * 100
#         annual_return_class = 'positive' if annual_return >= 0 else 'negative'
#         html_content += f"""
#                 <div class="metric-card">
#                     <h3>Annual Return</h3>
#                     <div class="metric-value {annual_return_class}">{annual_return:.2f}%</div>
#                 </div>
#         """
        
#         # Format Sharpe Ratio
#         sharpe_ratio = metrics.get('sharpe_ratio', 0)
#         sharpe_class = 'positive' if sharpe_ratio >= 1 else 'negative'
#         html_content += f"""
#                 <div class="metric-card">
#                     <h3>Sharpe Ratio</h3>
#                     <div class="metric-value {sharpe_class}">{sharpe_ratio:.2f}</div>
#                 </div>
#         """
        
#         # Format Max Drawdown
#         max_drawdown = metrics.get('max_drawdown', 0) * 100
#         html_content += f"""
#                 <div class="metric-card">
#                     <h3>Max Drawdown</h3>
#                     <div class="metric-value negative">{max_drawdown:.2f}%</div>
#                 </div>
#         """
        
#         # Format Win Rate
#         win_rate = metrics.get('win_rate', 0) * 100
#         win_rate_class = 'positive' if win_rate >= 50 else 'negative'
#         html_content += f"""
#                 <div class="metric-card">
#                     <h3>Win Rate</h3>
#                     <div class="metric-value {win_rate_class}">{win_rate:.2f}%</div>
#                 </div>
#         """
        
#         # Format Number of Trades
#         num_trades = metrics.get('num_trades', 0)
#         html_content += f"""
#                 <div class="metric-card">
#                     <h3>Number of Trades</h3>
#                     <div class="metric-value">{num_trades}</div>
#                 </div>
#             </div>
#         """
        
#         # Add charts
#         html_content += """
#             <h2>Performance Charts</h2>
#         """
        
#         for fig_path in self.report_data['figures']:
#             fig_name = os.path.basename(fig_path)
#             fig_display_name = fig_name.split('_')[-1].split('.')[0].replace('_', ' ').title()
            
#             html_content += f"""
#             <div class="chart-container">
#                 <h3>{fig_display_name}</h3>
#                 <img src="{fig_path}" alt="{fig_display_name}">
#             </div>
#             """
        
#         # Add trade table
#         html_content += """
#             <h2>Trades</h2>
#             <table>
#                 <tr>
#                     <th>Trade ID</th>
#                     <th>Entry Date</th>
#                     <th>Exit Date</th>
#                     <th>Position Type</th>
#                     <th>Entry Price</th>
#                     <th>Exit Price</th>
#                     <th>PnL (%)</th>
#                 </tr>
#         """
        
#         # Add trade rows
#         for trade in self.report_data['trades']:
#             trade_id = trade.get('trade_id', '')
#             entry_date = trade.get('entry_date', '')
#             exit_date = trade.get('exit_date', 'Open')
#             position_type = trade.get('position_type', '')
#             entry_price = trade.get('entry_price', 0)
#             exit_price = trade.get('exit_price', 0)
#             pnl = trade.get('pnl', 0) * 100
#             pnl_class = 'positive' if pnl >= 0 else 'negative'
            
#             html_content += f"""
#                 <tr>
#                     <td>{trade_id}</td>
#                     <td>{entry_date}</td>
#                     <td>{exit_date}</td>
#                     <td>{position_type}</td>
#                     <td>${entry_price:.2f}</td>
#                     <td>${exit_price:.2f if exit_price else ''}</td>
#                     <td class="{pnl_class}">{pnl:.2f}% if pnl else ''</td>
#                 </tr>
#             """
        
#         # Close table and HTML
#         html_content += """
#             </table>
#         </body>
#         </html>
#         """
        
#         # Write HTML to file
#         with open(output_file, 'w') as f:
#             f.write(html_content)
        
#         logger.info(f"HTML report saved to {output_file}")
        
#         return output_file
    
#     def generate_pdf_report(self, output_file: str, title: str = None) -> str:
#         """
#         Generate PDF report
        
#         Parameters:
#         -----------
#         output_file : str
#             Output file path
#         title : str, optional
#             Report title
            
#         Returns:
#         --------
#         str
#             Path to the PDF file
#         """
#         try:
#             # First generate HTML report
#             html_file = output_file.replace('.pdf', '.html')
#             self.generate_html_report(html_file, title)
            
#             # Convert HTML to PDF using wkhtmltopdf
#             from weasyprint import HTML
            
#             HTML(html_file).write_pdf(output_file)
            
#             logger.info(f"PDF report saved to {output_file}")
#             return output_file
            
#         except ImportError:
#             logger.error("weasyprint not installed. Please install it to generate PDF reports.")
#             raise ImportError("weasyprint not installed. Please install it to generate PDF reports.")
        
#         except Exception as e:
#             logger.error(f"Error generating PDF report: {e}")
#             raise

import os
import logging
from datetime import timedelta
import pandas as pd
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
                    color: #333;
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
                    margin-bottom: 30px;
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
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .metrics-table {{
                    margin: 30px 0;
                }}
                .metrics-table h3 {{
                    margin-bottom: 10px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .overview-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .overview-card {{
                    flex: 1;
                    min-width: 300px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {self.report_data['timestamp']}</p>
            
            <div class="section">
                <h2>Backtest Overview</h2>
                <div class="overview-container">
        """
        
        # Add backtest overview section
        metrics = self.report_data['metrics']
        start_date = metrics.get('start_date', '')
        end_date = metrics.get('end_date', '')
        period_days = metrics.get('period_days', 0)
        start_value = metrics.get('start_value', 0)
        end_value = metrics.get('end_value', 0)
        
        html_content += f"""
                    <div class="overview-card">
                        <h3>Time Period</h3>
                        <p><strong>Start Date:</strong> {start_date}</p>
                        <p><strong>End Date:</strong> {end_date}</p>
                        <p><strong>Duration:</strong> {period_days} days</p>
                    </div>
                    
                    <div class="overview-card">
                        <h3>Portfolio Value</h3>
                        <p><strong>Initial Capital:</strong> ${start_value:.2f}</p>
                        <p><strong>Final Value:</strong> ${end_value:.2f}</p>
                        <p><strong>Absolute Change:</strong> ${end_value - start_value:.2f}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Key Performance Metrics</h2>
                <div class="metrics-container">
        """
        
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
        
        # Format Profit Factor
        profit_factor = metrics.get('profit_factor', 0)
        profit_factor_class = 'positive' if profit_factor > 1 else 'negative'
        profit_factor_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
        html_content += f"""
                    <div class="metric-card">
                        <h3>Profit Factor</h3>
                        <div class="metric-value {profit_factor_class}">{profit_factor_display}</div>
                    </div>
                </div>
            </div>
        """
        
        # Add detailed metrics tables
        html_content += """
            <div class="section">
                <h2>Detailed Metrics</h2>
                
                <div class="metrics-table">
                    <h3>Return Metrics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add return metrics
        html_content += f"""
                        <tr>
                            <td>Total Return</td>
                            <td>{metrics.get('total_return', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Annual Return</td>
                            <td>{metrics.get('annual_return', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Volatility (Annualized)</td>
                            <td>{metrics.get('volatility', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Sortino Ratio</td>
                            <td>{metrics.get('sortino_ratio', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Calmar Ratio</td>
                            <td>{metrics.get('calmar_ratio', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td>Omega Ratio</td>
                            <td>{metrics.get('omega_ratio', 0):.2f}</td>
                        </tr>
        """
        
        html_content += """
                    </table>
                </div>
                
                <div class="metrics-table">
                    <h3>Drawdown Metrics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add drawdown metrics
        max_dd_duration = metrics.get('max_drawdown_duration', timedelta(0))
        max_dd_days = max_dd_duration.days if isinstance(max_dd_duration, timedelta) else 'N/A'
        recovered = metrics.get('drawdown_recovered', False)
        
        html_content += f"""
                        <tr>
                            <td>Maximum Drawdown</td>
                            <td>{metrics.get('max_drawdown', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Maximum Drawdown Duration</td>
                            <td>{max_dd_days} days</td>
                        </tr>
                        <tr>
                            <td>Drawdown Recovered</td>
                            <td>{"Yes" if recovered else "No"}</td>
                        </tr>
        """
        
        html_content += """
                    </table>
                </div>
                
                <div class="metrics-table">
                    <h3>Trade Metrics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add trade metrics
        html_content += f"""
                        <tr>
                            <td>Total Trades</td>
                            <td>{metrics.get('total_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td>Total Closed Trades</td>
                            <td>{metrics.get('total_closed_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td>Total Open Trades</td>
                            <td>{metrics.get('total_open_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td>Winning Trades</td>
                            <td>{metrics.get('winning_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td>Losing Trades</td>
                            <td>{metrics.get('losing_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td>Win Rate</td>
                            <td>{metrics.get('win_rate', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Best Trade</td>
                            <td>{metrics.get('best_trade', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Worst Trade</td>
                            <td>{metrics.get('worst_trade', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Average Trade</td>
                            <td>{metrics.get('avg_trade', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Average Winning Trade</td>
                            <td>{metrics.get('avg_winning_trade', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Average Losing Trade</td>
                            <td>{metrics.get('avg_losing_trade', 0) * 100:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Profit Factor</td>
                            <td>{profit_factor_display}</td>
                        </tr>
                        <tr>
                            <td>Expectancy</td>
                            <td>{metrics.get('expectancy', 0) * 100:.2f}%</td>
                        </tr>
        """
        
        # Add trade duration metrics if available
        avg_trade_duration = metrics.get('avg_trade_duration', None)
        if avg_trade_duration is not None:
            avg_trade_hours = avg_trade_duration.total_seconds() / 3600 if isinstance(avg_trade_duration, timedelta) else 'N/A'
            html_content += f"""
                        <tr>
                            <td>Average Trade Duration</td>
                            <td>{avg_trade_hours:.2f} hours</td>
                        </tr>
            """
        
        avg_winning_duration = metrics.get('avg_winning_trade_duration', None)
        if avg_winning_duration is not None:
            avg_winning_hours = avg_winning_duration.total_seconds() / 3600 if isinstance(avg_winning_duration, timedelta) else 'N/A'
            html_content += f"""
                        <tr>
                            <td>Average Winning Trade Duration</td>
                            <td>{avg_winning_hours:.2f} hours</td>
                        </tr>
            """
        
        avg_losing_duration = metrics.get('avg_losing_trade_duration', None)
        if avg_losing_duration is not None:
            avg_losing_hours = avg_losing_duration.total_seconds() / 3600 if isinstance(avg_losing_duration, timedelta) else 'N/A'
            html_content += f"""
                        <tr>
                            <td>Average Losing Trade Duration</td>
                            <td>{avg_losing_hours:.2f} hours</td>
                        </tr>
            """
        
        # Add commission/fee information
        html_content += f"""
                        <tr>
                            <td>Commission Rate</td>
                            <td>{metrics.get('commission', 0) * 100:.3f}%</td>
                        </tr>
                        <tr>
                            <td>Total Fees Paid</td>
                            <td>${metrics.get('total_fees_paid', 0):.2f}</td>
                        </tr>
        """
        
        html_content += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Charts</h2>
        """
        
        # Add charts
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
            </div>
            
            <div class="section">
                <h2>Trades</h2>
                <table>
                    <tr>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Position Type</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>PnL (%)</th>
                        <th>Duration</th>
                    </tr>
        """
        
        # Add trade rows
        for trade in self.report_data['trades']:
            entry_date = trade.get('entry_date', '')
            exit_date = trade.get('exit_date', 'Open')
            position_type = trade.get('position_type', '')
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            pnl = trade.get('pnl', 0) * 100 if trade.get('pnl') is not None else None
            pnl_class = 'positive' if pnl is not None and pnl >= 0 else 'negative'
            duration = trade.get('duration', 'N/A')
            
            html_content += f"""
                    <tr>
                        <td>{entry_date}</td>
                        <td>{exit_date if exit_date is not None else 'Open'}</td>
                        <td>{position_type}</td>
                        <td>${entry_price:.2f}</td>
                        <td>${exit_price:.2f if exit_price is not None else ''}</td>
                        <td class="{pnl_class}">{f'{pnl:.2f}%' if pnl is not None else ''}</td>
                        <td>{duration if duration != 'N/A' else ''}</td>
                    </tr>
            """
        
        # Close table and HTML
        html_content += """
                </table>
            </div>
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
            
            # Convert HTML to PDF using weasyprint
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