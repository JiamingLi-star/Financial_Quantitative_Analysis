# Financial Quantitative Analysis 📈

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)

A self‑contained Python script that:

* downloads historical price data from **Yahoo Finance** with `yfinance`
* computes technical indicators (5‑ and 30‑day moving averages, volatility, returns)
* detects **golden/death crosses** and generates trading signals
* back‑tests a simple long‑only strategy (buy on golden cross, sell on death cross)
* evaluates performance – total & annual return, max draw‑down, Sharpe ratio
* predicts the next golden/death cross with a **Random Forest** model  
* plots a 4‑in‑1 diagnostic figure (price, moving averages, signals, return curve)

> **Disclaimer** – This project is for educational purposes only and **is not investment advice**.

---

## Demo

```console
$ python Financial_Quantitative_Analysis.py
---Stock Prediction and Recommendation System---
Please enter stock code (e.g., AAPL for Apple): AAPL
Please enter start date (YYYY-MM-DD): 2015-01-01
Please enter initial capital (positive number): 10000
Random-Forest MAE on hold-out: 9.23 days
Predicted Next Golden Cross: 2025-05-28
Predicted Next Death Cross : 2025-07-06
Recommendation             : Buy
Total Return      : 148.32%
Annualized Return : 12.01%
Maximum Draw-down  : -17.89%
Sharpe Ratio       : 1.23
```

Running the script also pops up a Matplotlib window with the diagnostic chart.

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-github-name>/Financial_Quantitative_Analysis.git
   cd Financial_Quantitative_Analysis
   ```

2. **Create a virtual environment** (recommended) and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   `requirements.txt`

   ```
   yfinance>=0.2
   pandas>=2.0
   numpy>=1.23
   matplotlib>=3.8
   scikit-learn>=1.4
   ```

---

## Usage

```
python Financial_Quantitative_Analysis.py
```

The program is fully interactive:

| Prompt                               | Meaning                                      | Example          |
|--------------------------------------|----------------------------------------------|------------------|
| `stock code`                         | Yahoo Finance ticker                         | `AAPL`, `MSFT`   |
| `start date`                         | First day of back‑test (YYYY‑MM‑DD)          | `2010‑01‑01`     |
| `initial capital`                    | Cash balance in USD                          | `10000`          |

Type **y** to analyse another ticker, or **n** to exit.

---

## Project Structure

```
├─ Financial_Quantitative_Analysis.py   ← main script
├─ README.md                            ← this file
├─ requirements.txt                     ← Python packages
└─ .gitignore                           ← optional, prevents cache / venv uploads
```

---

## Roadmap

* Add alternative indicators (RSI, MACD, Bollinger Bands)
* Plug in walk‑forward cross‑validation for the Random Forest
* Export results to HTML or CSV
* Turn the CLI into a Streamlit web app

Contributions are welcome! Open an issue to discuss ideas or submit a pull request.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
