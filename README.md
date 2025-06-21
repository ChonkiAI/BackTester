# Python Financial Trading Strategy Backtester

## 1. Project Overview

This is a comprehensive and extensible event-driven backtesting framework written in Python. It is designed to test quantitative trading strategies against historical stock data, providing detailed performance analysis and realistic simulations of trading costs. The engine can process multiple strategies across numerous assets in batches, generating aggregated performance reports to identify top-performing strategies.

---

## 2. Features

* **Event-Driven Engine**: Processes data day-by-day, providing a realistic simulation environment.
* **Strategy Library**: Comes with over 100 pre-built trading strategies based on common technical indicators from the `pandas-ta` library.
* **Performance Metrics**: Calculates a wide array of metrics, including:
    * Total Return (%)
    * Max Drawdown (%) & Duration
    * Sharpe Ratio, Sortino Ratio, Calmar Ratio
    * Win/Loss Rate, Average Win/Loss
    * Alpha and Beta for benchmark comparison.
* **Realistic Cost Simulation**: Models trading costs including:
    * Brokerage (configurable, with a Zerodha model included)
    * Securities Transaction Tax (STT)
    * Transaction Charges, GST, SEBI Fees, and Stamp Duty.
* **Risk Management**: Includes features to control risk, such as:
    * `max_drawdown_limit_pct`: Halts trading if a drawdown threshold is breached.
    * `max_holding_period`: Automatically closes trades held for too long.
    * `max_capital_allocation_per_trade`: Controls position sizing.
* **Automated Reporting**:
    * Generates detailed Excel reports for each stock's strategy performance.
    * Creates a final PDF and Excel report ranking all strategies across all tested stocks.
    * Logs and summarizes any strategies that fail during execution.
* **Data Visualization**: Plots equity curves, drawdown periods, and buy/sell signals on price charts using Matplotlib.

---

## 3. Technologies Used

* **Core**: Python 3.8+
* **Libraries**:
    * **Jupyter Notebook**: For interactive development and driving the backtests.
    * **Pandas**: For data manipulation and analysis.
    * **NumPy**: For numerical operations.
    * **Matplotlib**: For plotting results.
    * **Pandas-TA**: For a vast library of technical analysis indicators.
    * **fpdf2**: For generating PDF reports.
    * **openpyxl**: For formatting Excel reports.
    * **TA-Lib**: Required for some advanced candlestick pattern strategies.

---

## 4. Project Structure

A recommended structure for organizing the project:
├── backtester/
│   ├── init.py
│   ├── backtester.py          # Core backtesting engine
│   └── strategies.py          # All strategy class definitions
├── data/
│   └── example_stock_data.csv # Your historical stock data
├── benchmarks/
│   └── example_benchmark_data.csv # Benchmark index data (e.g., Nifty 50)
├── results/
│   └── ...                    # Output files are saved here
├── main.py                    # Main script to run the backtests
├── README.md                  # This file
├── requirements.txt           # Project dependencies
├── .gitignore                 # Files to ignore in git
└── Dockerfile                 # For containerized setup
---

## 5. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ChonkiAI/BackTester.git
    cd BackTester
    ```

2.  **Install TA-Lib (Prerequisite):**
    Some strategies depend on the TA-Lib C library. It must be installed before the Python wrapper.

    * **Windows**: Download `TA-Lib-0.4.0-msvc.zip` from [lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and unzip it to `C:\ta-lib`.
    * **macOS**: `brew install ta-lib`
    * **Linux**: Download the `ta-lib-0.4.0-src.tar.gz`, extract, and run `./configure --prefix=/usr`, `make`, and `sudo make install`.

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **(Optional) Using Docker:**
    If you have Docker installed, you can build and run the environment in a container, which handles all dependencies automatically.
    ```bash
    # Build the docker image
    docker build -t py-backtester .

    # Run the container with your local data and results folders mounted
    docker run -it -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results py-backtester
    ```

---

## 6. Example Usage

1.  **Prepare Your Data**:
    * Place your historical stock data CSV files in the `data/` folder. Each file should represent one stock and contain `Date`, `Open`, `High`, `Low`, `Close`, `Volume` columns.
    * Place your benchmark index data (e.g., Nifty 50) in the `benchmarks/` folder.

2.  **Configure the Backtest**:
    Open the `main.py` file and configure the parameters:
    ```python
    # main.py

    # --- CONFIGURATION ---
    data_folder_path = 'data/'
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    benchmark_file_path = 'benchmarks/example_benchmark_data.csv'

    backtester_params = {
        'initial_capital': 100000,
        'broker_type': 'Zerodha',
        'trade_type': 'delivery',
        'slippage_bps': 1.0,
        'max_holding_period': 30,
        'max_capital_allocation_per_trade': 0.1,
        'max_drawdown_limit_pct': 0.15,
        'benchmark_file_path': benchmark_file_path
    }
    # ... rest of the script
    ```

3.  **Run the Backtest**:
    Execute the main script from your terminal:
    ```bash
    python main.py
    ```

4.  **Check the Results**:
    The script will create a timestamped folder inside the `results/` directory containing:
    * An Excel file for each stock, detailing the performance of every strategy.
    * `strategy_ranking.pdf` and `strategy_ranking.xlsx` files that aggregate and rank all strategies across all stocks.
    * `failed_strategies_summary.xlsx` if any strategies encountered errors.
