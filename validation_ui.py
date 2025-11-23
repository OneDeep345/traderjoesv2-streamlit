"""
Validation UI Component for Crypto Futures Trading Bot
Provides real-time validation monitoring and analysis interface
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QTextEdit, QGroupBox, QSplitter,
    QHeaderView, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QPushButton, QProgressBar, QComboBox, QCheckBox, QSpinBox
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QDateTime, pyqtSlot,
    QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QColor, QBrush, QFont, QPainter, QPen, QPixmap
)
import pyqtgraph as pg
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Optional

from validation_engine import (
    ValidationEngine, ValidationStatus, ValidationCategory,
    ValidationResult, TradeValidation
)


class ValidationStatusWidget(QWidget):
    """Widget showing overall validation status"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Health Score
        health_layout = QHBoxLayout()
        health_layout.addWidget(QLabel("System Health:"))
        
        self.health_bar = QProgressBar()
        self.health_bar.setMaximum(100)
        self.health_bar.setTextVisible(True)
        self.health_bar.setFormat("%v%")
        health_layout.addWidget(self.health_bar)
        
        self.health_label = QLabel("Excellent")
        self.health_label.setStyleSheet("font-weight: bold; color: #00FF88;")
        health_layout.addWidget(self.health_label)
        
        layout.addLayout(health_layout)
        
        # Validation counts
        counts_layout = QHBoxLayout()
        
        self.passed_label = self.create_count_label("Passed", "#00FF88")
        self.warning_label = self.create_count_label("Warnings", "#FFB800")
        self.failed_label = self.create_count_label("Failed", "#FF3366")
        self.critical_label = self.create_count_label("Critical", "#FF0000")
        
        counts_layout.addWidget(self.passed_label[0])
        counts_layout.addWidget(self.warning_label[0])
        counts_layout.addWidget(self.failed_label[0])
        counts_layout.addWidget(self.critical_label[0])
        
        layout.addLayout(counts_layout)
        
        # Execution metrics
        metrics_group = QGroupBox("Execution Metrics")
        metrics_layout = QVBoxLayout()
        
        # Slippage
        slippage_layout = QHBoxLayout()
        slippage_layout.addWidget(QLabel("Avg Slippage:"))
        self.slippage_label = QLabel("0.00%")
        self.slippage_label.setStyleSheet("font-weight: bold;")
        slippage_layout.addWidget(self.slippage_label)
        slippage_layout.addStretch()
        metrics_layout.addLayout(slippage_layout)
        
        # Price accuracy
        accuracy_layout = QHBoxLayout()
        accuracy_layout.addWidget(QLabel("Price Accuracy:"))
        self.accuracy_label = QLabel("100.00%")
        self.accuracy_label.setStyleSheet("font-weight: bold; color: #00FF88;")
        accuracy_layout.addWidget(self.accuracy_label)
        accuracy_layout.addStretch()
        metrics_layout.addLayout(accuracy_layout)
        
        # Total fees
        fees_layout = QHBoxLayout()
        fees_layout.addWidget(QLabel("Total Fees:"))
        self.fees_label = QLabel("$0.00")
        self.fees_label.setStyleSheet("font-weight: bold;")
        fees_layout.addWidget(self.fees_label)
        fees_layout.addStretch()
        metrics_layout.addLayout(fees_layout)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        self.setLayout(layout)
    
    def create_count_label(self, text: str, color: str):
        """Create a count label with color"""
        container = QWidget()
        layout = QVBoxLayout()
        
        count = QLabel("0")
        count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #8892B0;")
        
        layout.addWidget(count)
        layout.addWidget(label)
        container.setLayout(layout)
        
        return container, count
    
    def update_status(self, summary: Dict):
        """Update status display with validation summary"""
        # Update health score
        health = summary.get('overall_health', 100)
        self.health_bar.setValue(int(health))
        
        if health >= 90:
            self.health_label.setText("Excellent")
            self.health_label.setStyleSheet("font-weight: bold; color: #00FF88;")
            self.health_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #00FF88;
                }
            """)
        elif health >= 70:
            self.health_label.setText("Good")
            self.health_label.setStyleSheet("font-weight: bold; color: #00D4FF;")
            self.health_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #00D4FF;
                }
            """)
        elif health >= 50:
            self.health_label.setText("Warning")
            self.health_label.setStyleSheet("font-weight: bold; color: #FFB800;")
            self.health_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #FFB800;
                }
            """)
        else:
            self.health_label.setText("Critical")
            self.health_label.setStyleSheet("font-weight: bold; color: #FF3366;")
            self.health_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #FF3366;
                }
            """)
        
        # Update counts
        if 'by_category' in summary:
            total_passed = sum(cat.get('passed', 0) for cat in summary['by_category'].values())
            total_warnings = sum(cat.get('warnings', 0) for cat in summary['by_category'].values())
            total_failed = sum(cat.get('failed', 0) for cat in summary['by_category'].values())
            total_critical = sum(cat.get('critical', 0) for cat in summary['by_category'].values())
            
            self.passed_label[1].setText(str(total_passed))
            self.warning_label[1].setText(str(total_warnings))
            self.failed_label[1].setText(str(total_failed))
            self.critical_label[1].setText(str(total_critical))
        
        # Update execution metrics
        if 'execution_metrics' in summary:
            metrics = summary['execution_metrics']
            
            slippage = metrics.get('avg_slippage', 0) * 100
            self.slippage_label.setText(f"{slippage:.3f}%")
            if slippage < 0.05:
                self.slippage_label.setStyleSheet("font-weight: bold; color: #00FF88;")
            elif slippage < 0.1:
                self.slippage_label.setStyleSheet("font-weight: bold; color: #FFB800;")
            else:
                self.slippage_label.setStyleSheet("font-weight: bold; color: #FF3366;")
            
            accuracy = metrics.get('price_accuracy', 100)
            self.accuracy_label.setText(f"{accuracy:.2f}%")
            if accuracy >= 99:
                self.accuracy_label.setStyleSheet("font-weight: bold; color: #00FF88;")
            elif accuracy >= 95:
                self.accuracy_label.setStyleSheet("font-weight: bold; color: #FFB800;")
            else:
                self.accuracy_label.setStyleSheet("font-weight: bold; color: #FF3366;")
            
            fees = metrics.get('total_fees', 0)
            self.fees_label.setText(f"${fees:.2f}")


class ValidationChartWidget(QWidget):
    """Charts for validation metrics over time"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_data()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Tab widget for different charts
        self.chart_tabs = QTabWidget()
        
        # Validation rate chart
        self.validation_chart = pg.PlotWidget()
        self.validation_chart.setLabel('left', 'Count', color='#00D4FF')
        self.validation_chart.setLabel('bottom', 'Time', color='#00D4FF')
        self.validation_chart.setTitle('Validation Results', color='#00D4FF')
        self.validation_chart.setBackground('#151A3A')
        self.validation_chart.showGrid(x=True, y=True, alpha=0.1)
        
        self.chart_tabs.addTab(self.validation_chart, "Validation Rate")
        
        # Slippage chart
        self.slippage_chart = pg.PlotWidget()
        self.slippage_chart.setLabel('left', 'Slippage (%)', color='#00D4FF')
        self.slippage_chart.setLabel('bottom', 'Trade #', color='#00D4FF')
        self.slippage_chart.setTitle('Execution Slippage', color='#00D4FF')
        self.slippage_chart.setBackground('#151A3A')
        self.slippage_chart.showGrid(x=True, y=True, alpha=0.1)
        
        self.chart_tabs.addTab(self.slippage_chart, "Slippage")
        
        # Price deviation chart
        self.deviation_chart = pg.PlotWidget()
        self.deviation_chart.setLabel('left', 'Deviation (%)', color='#00D4FF')
        self.deviation_chart.setLabel('bottom', 'Time', color='#00D4FF')
        self.deviation_chart.setTitle('Price Execution Accuracy', color='#00D4FF')
        self.deviation_chart.setBackground('#151A3A')
        self.deviation_chart.showGrid(x=True, y=True, alpha=0.1)
        
        self.chart_tabs.addTab(self.deviation_chart, "Price Accuracy")
        
        layout.addWidget(self.chart_tabs)
        self.setLayout(layout)
    
    def init_data(self):
        """Initialize data storage"""
        self.time_data = deque(maxlen=100)
        self.passed_data = deque(maxlen=100)
        self.warning_data = deque(maxlen=100)
        self.failed_data = deque(maxlen=100)
        
        self.slippage_data = deque(maxlen=100)
        self.deviation_data = deque(maxlen=100)
        
        # Create plot items
        self.passed_line = self.validation_chart.plot(pen=pg.mkPen('#00FF88', width=2))
        self.warning_line = self.validation_chart.plot(pen=pg.mkPen('#FFB800', width=2))
        self.failed_line = self.validation_chart.plot(pen=pg.mkPen('#FF3366', width=2))
        
        self.slippage_line = self.slippage_chart.plot(pen=pg.mkPen('#00D4FF', width=2))
        self.slippage_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen('#00D4FF'), 
                                                   brush=pg.mkBrush('#00D4FF'))
        self.slippage_chart.addItem(self.slippage_scatter)
        
        self.deviation_line = self.deviation_chart.plot(pen=pg.mkPen('#00D4FF', width=2))
    
    def update_validation_chart(self, passed: int, warnings: int, failed: int):
        """Update validation rate chart"""
        current_time = len(self.time_data)
        self.time_data.append(current_time)
        self.passed_data.append(passed)
        self.warning_data.append(warnings)
        self.failed_data.append(failed)
        
        self.passed_line.setData(list(self.time_data), list(self.passed_data))
        self.warning_line.setData(list(self.time_data), list(self.warning_data))
        self.failed_line.setData(list(self.time_data), list(self.failed_data))
    
    def add_slippage_point(self, slippage: float):
        """Add slippage data point"""
        trade_num = len(self.slippage_data)
        self.slippage_data.append(slippage * 100)  # Convert to percentage
        
        x_data = list(range(len(self.slippage_data)))
        y_data = list(self.slippage_data)
        
        self.slippage_line.setData(x_data, y_data)
        self.slippage_scatter.setData(x_data, y_data)
    
    def add_deviation_point(self, deviation: float):
        """Add price deviation data point"""
        self.deviation_data.append(deviation * 100)  # Convert to percentage
        
        x_data = list(range(len(self.deviation_data)))
        y_data = list(self.deviation_data)
        
        self.deviation_line.setData(x_data, y_data)


class ValidationLogWidget(QTableWidget):
    """Table showing detailed validation logs"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Set columns
        columns = ['Time', 'Category', 'Status', 'Check', 'Expected', 
                  'Actual', 'Deviation', 'Message']
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        
        # Configure table
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSortingEnabled(True)
        
    def add_validation(self, result: ValidationResult):
        """Add validation result to log"""
        row = self.rowCount()
        self.insertRow(row)
        
        # Time
        time_item = QTableWidgetItem(result.timestamp.strftime("%H:%M:%S"))
        self.setItem(row, 0, time_item)
        
        # Category
        category_item = QTableWidgetItem(result.category.value)
        self.setItem(row, 1, category_item)
        
        # Status
        status_item = QTableWidgetItem(result.status.value.upper())
        
        # Color code by status
        if result.status == ValidationStatus.PASSED:
            status_item.setForeground(QBrush(QColor('#00FF88')))
        elif result.status == ValidationStatus.WARNING:
            status_item.setForeground(QBrush(QColor('#FFB800')))
        elif result.status == ValidationStatus.FAILED:
            status_item.setForeground(QBrush(QColor('#FF3366')))
        elif result.status == ValidationStatus.CRITICAL:
            status_item.setForeground(QBrush(QColor('#FF0000')))
            status_item.setFont(QFont("Arial", weight=QFont.Weight.Bold))
        
        self.setItem(row, 2, status_item)
        
        # Check name
        self.setItem(row, 3, QTableWidgetItem(result.check_name))
        
        # Expected value
        expected = str(result.expected_value)[:50]  # Truncate long values
        self.setItem(row, 4, QTableWidgetItem(expected))
        
        # Actual value
        actual = str(result.actual_value)[:50]
        self.setItem(row, 5, QTableWidgetItem(actual))
        
        # Deviation
        deviation_item = QTableWidgetItem(f"{result.deviation:.4f}")
        if result.deviation > 0.01:
            deviation_item.setForeground(QBrush(QColor('#FF3366')))
        elif result.deviation > 0.001:
            deviation_item.setForeground(QBrush(QColor('#FFB800')))
        self.setItem(row, 6, deviation_item)
        
        # Message
        self.setItem(row, 7, QTableWidgetItem(result.message))
        
        # Keep only last 1000 rows
        if self.rowCount() > 1000:
            self.removeRow(0)


class TradeValidationWidget(QWidget):
    """Widget showing validation details for each trade"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.trade_validations = {}
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Trade selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Trade:"))
        
        self.trade_combo = QComboBox()
        self.trade_combo.currentTextChanged.connect(self.on_trade_selected)
        selector_layout.addWidget(self.trade_combo)
        
        selector_layout.addStretch()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_trades)
        selector_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(selector_layout)
        
        # Trade validation details
        self.validation_tree = QTreeWidget()
        self.validation_tree.setHeaderLabels(['Check', 'Status', 'Details'])
        layout.addWidget(self.validation_tree)
        
        # Summary
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(100)
        layout.addWidget(self.summary_text)
        
        self.setLayout(layout)
    
    def add_trade_validation(self, trade_validation: TradeValidation):
        """Add a trade validation to the widget"""
        trade_id = trade_validation.trade_id
        self.trade_validations[trade_id] = trade_validation
        
        # Add to combo box
        display_text = f"{trade_id[:8]}... - {trade_validation.symbol} ({trade_validation.strategy})"
        self.trade_combo.addItem(display_text, trade_id)
        
        # Auto-select if it's the first one
        if self.trade_combo.count() == 1:
            self.on_trade_selected(display_text)
    
    def on_trade_selected(self, text: str):
        """Handle trade selection"""
        if not text:
            return
        
        # Get trade ID from combo box data
        index = self.trade_combo.currentIndex()
        if index < 0:
            return
        
        trade_id = self.trade_combo.itemData(index)
        if not trade_id or trade_id not in self.trade_validations:
            return
        
        # Display validation details
        validation = self.trade_validations[trade_id]
        self.display_validation(validation)
    
    def display_validation(self, validation: TradeValidation):
        """Display validation details in tree"""
        self.validation_tree.clear()
        
        # Group checks by category
        checks_by_category = defaultdict(list)
        for check in validation.validation_checks:
            checks_by_category[check.category.value].append(check)
        
        # Add to tree
        for category, checks in checks_by_category.items():
            category_item = QTreeWidgetItem([category.replace('_', ' ').title(), '', ''])
            
            # Determine category status
            has_critical = any(c.status == ValidationStatus.CRITICAL for c in checks)
            has_failed = any(c.status == ValidationStatus.FAILED for c in checks)
            has_warning = any(c.status == ValidationStatus.WARNING for c in checks)
            
            if has_critical:
                category_item.setForeground(1, QBrush(QColor('#FF0000')))
                category_item.setText(1, 'CRITICAL')
            elif has_failed:
                category_item.setForeground(1, QBrush(QColor('#FF3366')))
                category_item.setText(1, 'FAILED')
            elif has_warning:
                category_item.setForeground(1, QBrush(QColor('#FFB800')))
                category_item.setText(1, 'WARNING')
            else:
                category_item.setForeground(1, QBrush(QColor('#00FF88')))
                category_item.setText(1, 'PASSED')
            
            # Add individual checks
            for check in checks:
                check_item = QTreeWidgetItem([
                    check.check_name,
                    check.status.value.upper(),
                    check.message
                ])
                
                # Color code
                if check.status == ValidationStatus.PASSED:
                    check_item.setForeground(1, QBrush(QColor('#00FF88')))
                elif check.status == ValidationStatus.WARNING:
                    check_item.setForeground(1, QBrush(QColor('#FFB800')))
                elif check.status == ValidationStatus.FAILED:
                    check_item.setForeground(1, QBrush(QColor('#FF3366')))
                elif check.status == ValidationStatus.CRITICAL:
                    check_item.setForeground(1, QBrush(QColor('#FF0000')))
                
                category_item.addChild(check_item)
            
            self.validation_tree.addTopLevelItem(category_item)
            category_item.setExpanded(True)
        
        # Update summary
        self.update_summary(validation)
    
    def update_summary(self, validation: TradeValidation):
        """Update summary text"""
        summary = f"Trade ID: {validation.trade_id}\n"
        summary += f"Symbol: {validation.symbol}\n"
        summary += f"Strategy: {validation.strategy}\n"
        summary += f"Overall Status: {validation.overall_status.value.upper()}\n"
        summary += f"Confidence Score: {validation.confidence_score:.2%}\n"
        summary += f"Total Checks: {len(validation.validation_checks)}"
        
        self.summary_text.setText(summary)
    
    def refresh_trades(self):
        """Refresh trade list"""
        # This would be connected to the main validation engine
        pass


class ValidationControlWidget(QWidget):
    """Control panel for validation settings"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Validation settings
        settings_group = QGroupBox("Validation Settings")
        settings_layout = QVBoxLayout()
        
        # Enable/disable categories
        self.strategy_check = QCheckBox("Strategy Execution")
        self.strategy_check.setChecked(True)
        settings_layout.addWidget(self.strategy_check)
        
        self.price_check = QCheckBox("Price Matching")
        self.price_check.setChecked(True)
        settings_layout.addWidget(self.price_check)
        
        self.data_check = QCheckBox("Data Integrity")
        self.data_check.setChecked(True)
        settings_layout.addWidget(self.data_check)
        
        self.risk_check = QCheckBox("Risk Compliance")
        self.risk_check.setChecked(True)
        settings_layout.addWidget(self.risk_check)
        
        self.ml_check = QCheckBox("ML Predictions")
        self.ml_check.setChecked(True)
        settings_layout.addWidget(self.ml_check)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Execution simulation settings
        sim_group = QGroupBox("Execution Simulation")
        sim_layout = QVBoxLayout()
        
        # Slippage multiplier
        slippage_layout = QHBoxLayout()
        slippage_layout.addWidget(QLabel("Slippage Multiplier:"))
        self.slippage_spin = QSpinBox()
        self.slippage_spin.setRange(50, 200)
        self.slippage_spin.setValue(100)
        self.slippage_spin.setSuffix("%")
        slippage_layout.addWidget(self.slippage_spin)
        sim_layout.addLayout(slippage_layout)
        
        # Fee structure
        fee_layout = QHBoxLayout()
        fee_layout.addWidget(QLabel("Fee Structure:"))
        self.fee_combo = QComboBox()
        self.fee_combo.addItems(["Binance Standard", "VIP 1", "VIP 2", "Custom"])
        fee_layout.addWidget(self.fee_combo)
        sim_layout.addLayout(fee_layout)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.export_btn = QPushButton("Export Validation Report")
        actions_layout.addWidget(self.export_btn)
        
        self.clear_btn = QPushButton("Clear Validation History")
        actions_layout.addWidget(self.clear_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def get_settings(self) -> Dict:
        """Get current validation settings"""
        return {
            'categories': {
                'strategy_execution': self.strategy_check.isChecked(),
                'price_matching': self.price_check.isChecked(),
                'data_integrity': self.data_check.isChecked(),
                'risk_compliance': self.risk_check.isChecked(),
                'ml_prediction': self.ml_check.isChecked()
            },
            'simulation': {
                'slippage_multiplier': self.slippage_spin.value() / 100,
                'fee_structure': self.fee_combo.currentText()
            }
        }


class ValidationTab(QWidget):
    """Main validation tab for the trading bot UI"""
    
    def __init__(self):
        super().__init__()
        self.validation_engine = ValidationEngine()
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Validation & Integrity Monitor")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00D4FF;")
        layout.addWidget(title)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Status and charts
        left_panel = QSplitter(Qt.Orientation.Vertical)
        
        # Status widget
        self.status_widget = ValidationStatusWidget()
        left_panel.addWidget(self.status_widget)
        
        # Charts
        self.chart_widget = ValidationChartWidget()
        left_panel.addWidget(self.chart_widget)
        
        left_panel.setSizes([200, 400])
        main_splitter.addWidget(left_panel)
        
        # Center panel - Logs and trade validations
        center_panel = QSplitter(Qt.Orientation.Vertical)
        
        # Validation log
        log_group = QGroupBox("Validation Log")
        log_layout = QVBoxLayout()
        self.log_widget = ValidationLogWidget()
        log_layout.addWidget(self.log_widget)
        log_group.setLayout(log_layout)
        center_panel.addWidget(log_group)
        
        # Trade validations
        trade_group = QGroupBox("Trade Validations")
        trade_layout = QVBoxLayout()
        self.trade_widget = TradeValidationWidget()
        trade_layout.addWidget(self.trade_widget)
        trade_group.setLayout(trade_layout)
        center_panel.addWidget(trade_group)
        
        center_panel.setSizes([350, 350])
        main_splitter.addWidget(center_panel)
        
        # Right panel - Controls
        self.control_widget = ValidationControlWidget()
        self.control_widget.setMaximumWidth(250)
        main_splitter.addWidget(self.control_widget)
        
        main_splitter.setSizes([500, 700, 250])
        layout.addWidget(main_splitter)
        
        self.setLayout(layout)
        
        # Connect signals
        self.control_widget.export_btn.clicked.connect(self.export_report)
        self.control_widget.clear_btn.clicked.connect(self.clear_history)
    
    def setup_timers(self):
        """Setup update timers"""
        # Update timer for charts
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(2000)  # Update every 2 seconds
    
    def validate_trade(self, trade: Dict, signal: Dict, 
                       market_data, order_book: Dict = None):
        """Validate a new trade"""
        # Perform validation
        trade_validation = self.validation_engine.validate_trade_entry(
            trade, signal, market_data, order_book
        )
        
        # Add to UI
        self.trade_widget.add_trade_validation(trade_validation)
        
        # Add individual checks to log
        for check in trade_validation.validation_checks:
            self.log_widget.add_validation(check)
        
        # Update charts
        if 'slippage' in trade:
            self.chart_widget.add_slippage_point(trade['slippage'])
        
        if 'price_deviation' in trade:
            self.chart_widget.add_deviation_point(trade['price_deviation'])
        
        return trade_validation
    
    def update_display(self):
        """Update validation display"""
        # Get summary from validation engine
        summary = self.validation_engine.get_validation_summary()
        
        # Update status widget
        self.status_widget.update_status(summary)
        
        # Update validation chart with recent counts
        if 'by_category' in summary:
            recent_passed = sum(cat.get('passed', 0) for cat in summary['by_category'].values())
            recent_warnings = sum(cat.get('warnings', 0) for cat in summary['by_category'].values())
            recent_failed = sum(cat.get('failed', 0) for cat in summary['by_category'].values())
            
            self.chart_widget.update_validation_chart(
                recent_passed, recent_warnings, recent_failed
            )
    
    def export_report(self):
        """Export validation report"""
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Validation Report",
            f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': self.validation_engine.get_validation_summary(),
                'trade_validations': {
                    trade_id: {
                        'symbol': val.symbol,
                        'strategy': val.strategy,
                        'status': val.overall_status.value,
                        'checks': [check.to_dict() for check in val.validation_checks]
                    }
                    for trade_id, val in self.validation_engine.trade_validations.items()
                },
                'settings': self.control_widget.get_settings()
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4)
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export Complete", 
                                  f"Validation report exported to:\n{filename}")
    
    def clear_history(self):
        """Clear validation history"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all validation history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear engine history
            self.validation_engine.validation_history.clear()
            self.validation_engine.trade_validations.clear()
            self.validation_engine.statistics.clear()
            
            # Clear UI
            self.log_widget.setRowCount(0)
            self.trade_widget.validation_tree.clear()
            self.trade_widget.trade_validations.clear()
            self.trade_widget.trade_combo.clear()
            
            # Reset charts
            self.chart_widget.init_data()
