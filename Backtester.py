import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import datetime
from fpdf import FPDF  # For PDF generation
import os  # For managing temporary plot files

# --- 1. Trade Class ---
class Trade:
    def __init__(self, entry_date, entry_price, quantity, trade_type):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.trade_type = trade_type  # 'long' or 'short'
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0
        self.duration = 0
        self.transaction_cost = 0  # Total transaction cost for the trade (entry + exit)

    def close(self, exit_date, exit_price, exit_cost):
        self.exit_date = exit_date
        self.exit_price = exit_price
        if self.trade_type == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
        self.pnl -= exit_cost  # Deduct exit cost from PnL
        self.transaction_cost += exit_cost  # Add exit cost to total
        self.duration = (self.exit_date - self.entry_date).days


# --- 2. TransactionCosts Class ---
class TransactionCosts:
    def __init__(self, broker_type='Zerodha', trade_type='delivery'):
        self.broker = broker_type
        self.trade_type = trade_type

        # Zerodha specific charges (as per 2023-2024 rates, subject to change)
        self.brokerage_rates = {
            'delivery': 0,  # Free delivery
            'intraday': {'max': 20, 'percentage': 0.0003}  # Rs 20 or 0.03% whichever is lower
        }
        self.stt_rates = {  # Securities Transaction Tax
            'delivery_buy': 0.001,
            'delivery_sell': 0.001,
            'intraday_buy': 0.00025,
            'intraday_sell': 0.00025
        }
        self.transaction_charges = 0.0000345  # NSE (EQ) 0.00345% of turnover
        self.gst_rate = 0.18  # 18% of brokerage + transaction charges
        self.sebi_turnover_fee = 0.0000005  # 0.00005% of turnover
        self.stamp_duty_rates = {
            'delivery_buy': 0.00015,  # 0.015%
            'intraday_buy': 0.00003  # 0.003%
        }

    def calculate_cost(self, value, transaction_type):
        """
        Calculates total transaction costs for a trade.
        :param value: Notional value of the trade (price * quantity).
        :param transaction_type: 'buy' or 'sell'.
        :return: Total cost for the trade.
        """
        brokerage = 0
        stt = 0
        transaction_charges_val = 0
        gst = 0
        sebi_fee = 0
        stamp_duty = 0

        # Brokerage
        if self.trade_type == 'intraday':
            brokerage = min(self.brokerage_rates['intraday']['max'], value * self.brokerage_rates['intraday']['percentage'])
        # For delivery, brokerage is 0

        # STT (Securities Transaction Tax)
        if self.trade_type == 'delivery':
            stt = value * self.stt_rates[f'delivery_{transaction_type}']
        elif self.trade_type == 'intraday':
            stt = value * self.stt_rates[f'intraday_{transaction_type}']

        # Transaction Charges
        transaction_charges_val = value * self.transaction_charges

        # GST
        gst = (brokerage + transaction_charges_val) * self.gst_rate

        # SEBI Turnover Fee
        sebi_fee = value * self.sebi_turnover_fee

        # Stamp Duty (only on buy side)
        if transaction_type == 'buy':
            if self.trade_type == 'delivery':
                stamp_duty = value * self.stamp_duty_rates['delivery_buy']
            elif self.trade_type == 'intraday':
                stamp_duty = value * self.stamp_duty_rates['intraday_buy']

        total_cost = brokerage + stt + transaction_charges_val + gst + sebi_fee + stamp_duty
        return total_cost


# --- 3. Portfolio Class ---
class Portfolio:
    def __init__(self, initial_capital, max_capital_allocation_per_trade, transaction_costs_calculator):
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital  # New: Tracks liquid cash
        self.invested_capital = 0  # New: Tracks market value of active position (long or short)
        self.total_portfolio_value = initial_capital  # New: cash_balance + invested_capital (market value)

        self.max_capital_allocation_per_trade = max_capital_allocation_per_trade  # % of total_portfolio_value
        self.transaction_costs_calculator = transaction_costs_calculator

        self.equity_curve = pd.Series(dtype=float)
        self.daily_returns = pd.Series(dtype=float)  # New: To store daily returns
        self.trades = []
        self.active_trade = None
        self.positions = {'Asset': 0}  # Tracks quantity of asset held

    def _update_equity_for_day(self, date, current_price):
        """
        Updates the portfolio's equity curve for the current day.
        Also updates invested_capital and total_portfolio_value.
        """
        if self.active_trade:
            # Update market value of the invested capital
            if self.active_trade.trade_type == 'long':
                self.invested_capital = self.active_trade.quantity * current_price
            else:  # short
                # For short, invested_capital is negative, representing the current liability
                self.invested_capital = -self.active_trade.quantity * current_price
        else:
            self.invested_capital = 0  # No active position

        # Calculate total portfolio value (cash + mark-to-market value of positions)
        previous_total_portfolio_value = self.total_portfolio_value  # Store for daily return calculation
        self.total_portfolio_value = self.cash_balance + self.invested_capital

        # Update equity curve
        self.equity_curve.loc[date] = self.total_portfolio_value

        # Calculate and store daily return
        if not self.equity_curve.empty and len(self.equity_curve) > 1:
            current_day_value = self.equity_curve.iloc[-1]
            previous_day_value = self.equity_curve.iloc[-2]
            if previous_day_value != 0:  # Avoid division by zero
                self.daily_returns.loc[date] = (current_day_value - previous_day_value) / previous_day_value
            else:
                self.daily_returns.loc[date] = 0  # Or NaN, depending on desired behavior
        else:
            self.daily_returns.loc[date] = 0  # First day has no previous return

    def _close_active_trade(self, date, exit_price, exit_cost):
        """
        Helper method to close the active trade, calculate P&L, and update capital.
        """
        if self.active_trade:
            self.active_trade.close(date, exit_price, exit_cost)
            self.trades.append(self.active_trade)

            # Adjust cash_balance based on realized P&L
            if self.active_trade.trade_type == 'long':
                self.cash_balance += (self.active_trade.quantity * exit_price) - exit_cost
            else:  # short
                self.cash_balance += self.active_trade.pnl  # Net profit/loss affecting cash

            self.positions['Asset'] = 0  # No longer holding asset
            self.active_trade = None  # Clear active trade
            self.invested_capital = 0  # No longer invested

    def execute_trade(self, date, price, signal, trading_halt=False):
        """
        Executes a trade (buy/sell) based on a signal, ensuring only one active trade.
        Closes existing trade before opening a new one if a reversal signal is received.
        Shorting is restricted for 'delivery' trade type.
        """
        trade_executed = False

        # If a trading halt is active due to drawdown, prevent new trades
        if trading_halt and signal != 0:
            return False

        # Handle closing an existing trade if a reversal signal or other close condition is met
        if self.active_trade:
            should_close = False
            # If active trade is long and new signal is Sell (-1)
            if self.active_trade.trade_type == 'long' and signal == -1:
                should_close = True
            # If active trade is short and new signal is Buy (1)
            elif self.active_trade.trade_type == 'short' and signal == 1:
                should_close = True
            # If force close (signal 0 from Backtester due to max_holding_period or end of data)
            elif signal == 0:
                 should_close = True

            if should_close:
                closing_transaction_type = 'sell' if self.active_trade.trade_type == 'long' else 'buy'
                exit_value = self.active_trade.quantity * price
                exit_cost = self.transaction_costs_calculator.calculate_cost(exit_value, closing_transaction_type)
                self._close_active_trade(date, price, exit_cost)
                trade_executed = True  # A trade (closure) was executed

        # Execute new trade if no active trade and signal is not 0
        if not self.active_trade and signal != 0:
            # Calculate maximum capital to allocate for this trade based on total_portfolio_value
            max_allowed_capital_for_trade = self.total_portfolio_value * self.max_capital_allocation_per_trade

            if signal == 1:  # Buy (Long)
                # Determine max quantity based on allocated capital
                approx_quantity = int(max_allowed_capital_for_trade / price)
                if approx_quantity == 0:
                    return False  # Cannot even afford 1 share with allocated capital

                buy_value_approx = approx_quantity * price
                entry_cost_approx = self.transaction_costs_calculator.calculate_cost(buy_value_approx, 'buy')

                # Calculate exact quantity based on actual cash balance
                quantity_to_buy = 0
                for q in range(approx_quantity, 0, -1):
                    current_buy_value = q * price
                    current_entry_cost = self.transaction_costs_calculator.calculate_cost(current_buy_value, 'buy')
                    if (current_buy_value + current_entry_cost) <= self.cash_balance:
                        quantity_to_buy = q
                        break

                if quantity_to_buy > 0:
                    final_buy_value = quantity_to_buy * price
                    final_entry_cost = self.transaction_costs_calculator.calculate_cost(final_buy_value, 'buy')

                    self.cash_balance -= (final_buy_value + final_entry_cost)
                    self.active_trade = Trade(date, price, quantity_to_buy, 'long')
                    self.active_trade.transaction_cost += final_entry_cost  # Store entry cost
                    self.positions['Asset'] = quantity_to_buy
                    self.invested_capital = final_buy_value  # Set invested capital to the market value of the position
                    trade_executed = True

            elif signal == -1 and self.transaction_costs_calculator.trade_type != 'delivery':  # Only allow shorting if NOT delivery
                # For shorting, capital allocation limits the notional value of the short.
                approx_quantity = int(max_allowed_capital_for_trade / price)
                if approx_quantity == 0:
                    return False  # Cannot short 1 share with allocated capital

                quantity_to_short = approx_quantity  # No cash limit for shorting, only notional value
                final_short_value = quantity_to_short * price
                final_entry_cost = self.transaction_costs_calculator.calculate_cost(final_short_value, 'sell')

                if quantity_to_short > 0:
                    self.cash_balance += (final_short_value - final_entry_cost)  # Cash increases from short sale
                    self.active_trade = Trade(date, price, quantity_to_short, 'short')
                    self.active_trade.transaction_cost += final_entry_cost  # Store entry cost
                    self.positions['Asset'] = -quantity_to_short  # Negative for short position
                    self.invested_capital = -final_short_value  # Negative for short position's notional value
                    trade_executed = True

        return trade_executed

    def finalize_portfolio(self, final_date, final_price):
        """
        Closes any remaining active trade at the final price of the backtest.
        """
        if self.active_trade:
            closing_transaction_type = 'sell' if self.active_trade.trade_type == 'long' else 'buy'
            exit_value = self.active_trade.quantity * final_price
            exit_cost = self.transaction_costs_calculator.calculate_cost(exit_value, closing_transaction_type)
            self._close_active_trade(final_date, final_price, exit_cost)

        # Ensure final equity curve point is set correctly (after final trade closure)
        self._update_equity_for_day(final_date, final_price)


# --- 4. Backtester Class ---
class Backtester:
    def __init__(self, initial_capital, broker_type, trade_type, slippage_bps=1.0,
                 max_holding_period=None, max_capital_allocation_per_trade=1.0,
                 max_drawdown_limit_pct=None,  # New: Drawdown risk limit (e.g., 0.1 for 10%)
                 benchmark_file_path=None  # New: Path to benchmark data
                 ):
        self.initial_capital = initial_capital
        self.broker_type = broker_type
        self.trade_type = trade_type
        self.slippage_bps = slippage_bps
        self.max_holding_period = max_holding_period
        self.max_capital_allocation_per_trade = max_capital_allocation_per_trade
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.benchmark_file_path = benchmark_file_path
        self.benchmark_daily_returns = None
        self.strategy_instance = None

        self.transaction_costs_calculator = TransactionCosts(broker_type, trade_type)
        self.portfolio = Portfolio(initial_capital, max_capital_allocation_per_trade, self.transaction_costs_calculator)

    def _apply_slippage(self, price, is_buy):
        """Applies slippage to the price."""
        slippage_amount = price * (self.slippage_bps / 10000)
        if is_buy:
            return price + slippage_amount
        else:
            return price - slippage_amount
                            
    
    def load_data(self, file_path, start_date, end_date):
        """
        Loads and prepares historical data.
        Handles encoding issues, trims column names, and merges benchmark data if provided.
        """
        try:
            # Load and clean main data
            data = pd.read_csv(file_path, encoding='utf-8-sig')
            data.columns = data.columns.str.strip()
    
            if 'Date' not in data.columns:
                raise ValueError(f"'Date' column not found in file: {file_path}")
    
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.set_index('Date')
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
            # Filter by date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
    
            # Filter weekends (Saturday = 5, Sunday = 6)
            data = data[data.index.dayofweek < 5]
    
            # --- Handle Benchmark Data ---
            if self.benchmark_file_path:
                benchmark_data = pd.read_csv(self.benchmark_file_path, encoding='utf-8-sig')
                benchmark_data.columns = benchmark_data.columns.str.strip()
    
                if 'Date' not in benchmark_data.columns:
                    raise ValueError(f"'Date' column not found in benchmark file: {self.benchmark_file_path}")
    
                #benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'], errors='coerce')
                benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'], format='%Y-%m-%d', errors='coerce')
                benchmark_data = benchmark_data.set_index('Date')
                benchmark_data = benchmark_data[['Close']].dropna()
                benchmark_data.rename(columns={'Close': 'Benchmark_Close'}, inplace=True)
    
                # Filter benchmark data
                benchmark_data = benchmark_data[(benchmark_data.index >= start_date) & (benchmark_data.index <= end_date)]
                benchmark_data = benchmark_data[benchmark_data.index.dayofweek < 5]
    
                # Merge with main data
                data = pd.merge(data, benchmark_data, left_index=True, right_index=True, how='left')
                data['Benchmark_Close'] = data['Benchmark_Close'].ffill().bfill()
    
                if 'Benchmark_Close' not in data.columns:
                    raise ValueError("Benchmark_Close column not found after merging. Check benchmark file.")
    
            return data
    
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading or processing data: {e}")
            return None



    def run_backtest(self, strategy_instance, data):
        if data is None or data.empty:
            print("No data to run backtest.")
            return None, None

        # Ensure strategy instance has access to data if it needs it
        strategy_instance.data = data  # Pass current data to strategy for signal generation
        self.strategy_instance = strategy_instance  # Store for later use

        signals = strategy_instance.generate_signals()
        # Ensure signals index aligns with the data index used in backtest loop
        signals = signals[signals.index.isin(data.index)]
        # Reindex signals to data's index, filling NaNs with 0 (no signal)
        signals = signals.reindex(data.index, fill_value=0)

        # Reset portfolio state for a new backtest run
        self.portfolio = Portfolio(self.initial_capital, self.max_capital_allocation_per_trade, self.transaction_costs_calculator)
        self.portfolio.equity_curve.loc[data.index[0]] = self.initial_capital
        self.portfolio.daily_returns.loc[data.index[0]] = 0  # First day's return is 0

        max_equity = self.initial_capital  # For drawdown tracking
        trading_halt_active = False  # Flag for drawdown limit

        for i, (date, row) in enumerate(data.iterrows()):
            close_price = row['Close']
            current_signal = signals.loc[date]

            # 1. Update Portfolio for the day (including daily returns)
            self.portfolio._update_equity_for_day(date, close_price)

            # 2. Check and Apply Drawdown-Based Risk Limit
            if self.max_drawdown_limit_pct is not None:
                current_equity = self.portfolio.total_portfolio_value
                max_equity = max(max_equity, current_equity)  # Update peak equity
                current_drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0

                if current_drawdown > self.max_drawdown_limit_pct and not trading_halt_active:
                    trading_halt_active = True
                    # Optionally, force close current position if drawdown limit is breached
                    if self.portfolio.active_trade:
                        exit_price = self._apply_slippage(row['Close'], is_buy=(self.portfolio.active_trade.trade_type == 'short'))
                        # Force close, not affected by halt flag
                        self.portfolio.execute_trade(date, exit_price, 0, trading_halt=False)
                # Re-enable trading if drawdown recovers significantly (e.g., to half the limit)
                elif current_drawdown < self.max_drawdown_limit_pct * 0.5 and trading_halt_active:
                    trading_halt_active = False

            # 3. Trading Logic (influenced by drawdown halt)
            if self.portfolio.active_trade:
                current_holding_period = (date - self.portfolio.active_trade.entry_date).days

                should_close_trade = False
                # If active trade is long and new signal is Sell (-1)
                if self.portfolio.active_trade.trade_type == 'long' and current_signal == -1:
                    should_close_trade = True
                # If active trade is short and new signal is Buy (1)
                elif self.portfolio.active_trade.trade_type == 'short' and current_signal == 1:
                    should_close_trade = True

                # If no new signal (0) and it's the last day of data (force close)
                elif current_signal == 0 and i == len(data) - 1:
                    should_close_trade = True

                # Holding period exceeds max_holding_period
                if self.max_holding_period is not None and current_holding_period >= self.max_holding_period:
                    should_close_trade = True

                if should_close_trade:
                    exit_price = self._apply_slippage(row['Close'], is_buy=(self.portfolio.active_trade.trade_type == 'short'))
                    # Pass trading_halt_active flag to execute_trade for new trades after closure
                    self.portfolio.execute_trade(date, exit_price, 0, trading_halt=trading_halt_active)

                    # After explicit closure, if a new signal exists AND trading is not halted, try to open a new trade
                    if current_signal != 0 and not trading_halt_active:
                        entry_price = self._apply_slippage(row['Close'], is_buy=(current_signal == 1))
                        self.portfolio.execute_trade(date, entry_price, current_signal, trading_halt=trading_halt_active)

            # No active trade, consider opening a new position based on the signal and if trading is not halted
            elif current_signal != 0 and not trading_halt_active:
                entry_price = self._apply_slippage(row['Close'], is_buy=(current_signal == 1))
                self.portfolio.execute_trade(date, entry_price, current_signal, trading_halt=trading_halt_active)

        # Finalize any remaining open trades at the very end of the backtest period
        final_date = data.index[-1]
        final_price = data['Close'].iloc[-1]
        self.portfolio.finalize_portfolio(final_date, final_price)

        self.portfolio.equity_curve = self.portfolio.equity_curve.sort_index()
        self.portfolio.daily_returns = self.portfolio.daily_returns.sort_index()

        # Store benchmark returns if loaded
        if 'Benchmark_Close' in data.columns:
            benchmark_returns = data['Benchmark_Close'].pct_change().dropna()
            # Align benchmark returns with strategy equity curve index for analysis
            self.benchmark_daily_returns = benchmark_returns.reindex(self.portfolio.daily_returns.index).fillna(0)
            self.benchmark_daily_returns = self.benchmark_daily_returns.sort_index()
        else:
            self.benchmark_daily_returns = None

        return self.portfolio.equity_curve, self.portfolio.trades

    def analyze_results(self):
        # Ensure equity_curve and daily_returns are not empty
        if self.portfolio.equity_curve.empty:
            print("No equity curve data to analyze.")
            return {}  # Return empty dict if no data

        total_return_pct = (self.portfolio.equity_curve.iloc[-1] / self.initial_capital - 1) * 100

        # Calculate max drawdown
        peak = self.portfolio.equity_curve.cummax()
        drawdown = (self.portfolio.equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

        # Max drawdown duration
        max_drawdown_end_date = drawdown.idxmin() if not drawdown.empty else None
        max_drawdown_start_date = (peak.loc[:max_drawdown_end_date]).idxmax() if max_drawdown_end_date else None
        max_drawdown_duration = (max_drawdown_end_date - max_drawdown_start_date).days if max_drawdown_start_date and max_drawdown_end_date and max_drawdown_start_date != max_drawdown_end_date else 0

        # Calculate Sharpe Ratio and Sortino Ratio
        # Ensure daily_returns are available and not all zero/NaN
        if not self.portfolio.daily_returns.empty and self.portfolio.daily_returns.std() != 0:
            avg_daily_return = self.portfolio.daily_returns.mean()
            std_daily_return = self.portfolio.daily_returns.std()
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)  # Annualized

            # For Sortino Ratio, calculate downside deviation
            downside_returns = self.portfolio.daily_returns[self.portfolio.daily_returns < 0]
            if not downside_returns.empty and downside_returns.std() != 0:
                downside_std = downside_returns.std()
                sortino_ratio = (avg_daily_return / downside_std) * np.sqrt(252)
            else:
                sortino_ratio = np.nan  # No downside returns or std is 0
        else:
            sharpe_ratio = np.nan
            sortino_ratio = np.nan

        # Calmar Ratio
        annualized_return = (1 + total_return_pct / 100)**(252 / len(self.portfolio.equity_curve)) - 1 if len(self.portfolio.equity_curve) > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown / 100) if max_drawdown != 0 else np.nan

        # Trade statistics
        winning_trades = [t for t in self.portfolio.trades if t.pnl > 0]
        losing_trades = [t for t in self.portfolio.trades if t.pnl < 0]
        num_trades = len(self.portfolio.trades)
        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)
        win_rate = (num_winning_trades / num_trades * 100) if num_trades > 0 else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if num_winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if num_losing_trades > 0 else 0

        # --- Benchmark Comparison Metrics ---
        benchmark_total_return_pct = np.nan
        alpha = np.nan
        beta = np.nan

        if self.benchmark_daily_returns is not None and not self.benchmark_daily_returns.empty and not self.portfolio.daily_returns.empty:
            # Align indices of strategy and benchmark returns
            # Use .dropna() to ensure only days with both strategy and benchmark returns are included
            aligned_returns = pd.DataFrame({
                'Strategy': self.portfolio.daily_returns,
                'Benchmark': self.benchmark_daily_returns
            }).dropna()

            if not aligned_returns.empty and aligned_returns['Benchmark'].std() != 0:
                # Calculate benchmark total return
                benchmark_total_return = (1 + aligned_returns['Benchmark']).prod() - 1
                benchmark_total_return_pct = benchmark_total_return * 100

                # Calculate Beta
                covariance = aligned_returns['Strategy'].cov(aligned_returns['Benchmark'])
                variance_benchmark = aligned_returns['Benchmark'].var()
                if variance_benchmark != 0:
                    beta = covariance / variance_benchmark

                # Calculate Alpha (Jensen's Alpha for simplicity, assuming risk-free rate is 0)
                # Alpha = Strategy_Return - Beta * Benchmark_Return
                # Using annualized returns for calculation
                strategy_ann_return = (1 + aligned_returns['Strategy']).prod()**(252/len(aligned_returns)) - 1 if len(aligned_returns) > 0 else 0
                benchmark_ann_return = (1 + aligned_returns['Benchmark']).prod()**(252/len(aligned_returns)) - 1 if len(aligned_returns) > 0 else 0
                alpha = strategy_ann_return - (beta * benchmark_ann_return if not np.isnan(beta) else 0)

        results = {
            "Total Return (%)": total_return_pct,
            "Max Drawdown (%)": max_drawdown,
            "Max Drawdown Duration (Days)": max_drawdown_duration,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            "Number of Trades": num_trades,
            "Winning Trades": num_winning_trades,
            "Losing Trades": num_losing_trades,
            "Win Rate (%)": win_rate,
            "Average Win (INR)": avg_win,
            "Average Loss (INR)": avg_loss,
            "Benchmark Total Return (%)": benchmark_total_return_pct,
            "Alpha": alpha,
            "Beta": beta
        }
        return results

    # Modified plot_results to allow saving
    def plot_results(self, data, results, title="Backtest Results", save_path=None):
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        # Plot 1: Price and Trading Signals
        axes[0].plot(data['Close'], label='Close Price', color='blue', alpha=0.7)
        if 'SMA200' in data.columns:  # Assuming strategy adds this
            axes[0].plot(data['SMA200'], label='SMA200', color='orange', linestyle='--', alpha=0.7)
        # Re-generate signals to plot if they were calculated in the strategy instance
        if hasattr(self, 'strategy_instance') and hasattr(self.strategy_instance, 'generate_signals'):
             signals_to_plot = self.strategy_instance.generate_signals(data)
             buy_signals = signals_to_plot[signals_to_plot == 1]
             sell_signals = signals_to_plot[signals_to_plot == -1]
             if not buy_signals.empty:
                 axes[0].scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'],
                                 marker='^', color='green', s=100, label='Buy Signal', alpha=0.8)
             if not sell_signals.empty:
                 axes[0].scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'],
                                 marker='v', color='red', s=100, label='Sell Signal', alpha=0.8)

        axes[0].set_title(f'Price with Trading Signals for {title}')
        axes[0].set_ylabel('Price (INR)')
        axes[0].legend()
        axes[0].grid(True)

        # Plot 2: Equity Curve vs. Benchmark (if available)
        axes[1].plot(self.portfolio.equity_curve, label='Strategy Equity Curve', color='purple')
        if self.benchmark_daily_returns is not None and not self.benchmark_daily_returns.empty:
            # Calculate benchmark equity curve starting from initial capital
            benchmark_equity_curve = (1 + self.benchmark_daily_returns).cumprod() * self.initial_capital
            # Align benchmark_equity_curve index to match strategy's first date if necessary
            if self.portfolio.equity_curve.index[0] not in benchmark_equity_curve.index:
                benchmark_equity_curve.loc[self.portfolio.equity_curve.index[0]] = self.initial_capital
            benchmark_equity_curve = benchmark_equity_curve.sort_index()
            # Ensure it matches the backtest date range exactly
            benchmark_equity_curve = benchmark_equity_curve[self.portfolio.equity_curve.index.min():self.portfolio.equity_curve.index.max()]

            axes[1].plot(benchmark_equity_curve, label='Benchmark Equity Curve', color='gray', linestyle='--')
        axes[1].set_title(f'Equity Curve for {title}')
        axes[1].set_ylabel('Portfolio Value (INR)')
        axes[1].legend()
        axes[1].grid(True)

        # Plot 3: Drawdown
        peak = self.portfolio.equity_curve.cummax()
        drawdown = (self.portfolio.equity_curve - peak) / peak * 100
        axes[2].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        axes[2].plot(drawdown.index, drawdown, color='red', alpha=0.7)
        axes[2].set_title(f'Drawdown (%) for {title}')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)
        axes[2].axhline(0, color='gray', linestyle='--')  # Zero drawdown line

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free up memory
        else:
            plt.show()

        # Display results in a readable format for console output
        print(f"\n--- Backtest Results for {title} ---")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if "%" in key or "Ratio" in key:
                    print(f"{key}: {value:.2f}{'%' if '%' in key else ''}")
                else:
                    print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}") 