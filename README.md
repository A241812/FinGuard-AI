# FinGuard AI – Hackathon MVP

## Run locally (Windows / VS Code)

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Run:

```powershell
python -m streamlit run app.py
```

The browser should open at `http://localhost:8501`.

## Included
- Synthetic banking transaction generator
- Random Forest fraud classifier
- Isolation Forest anomaly detection
- AML indicator rules and transaction network analysis
- Combined risk scoring and human-readable explanations
- Dashboard, model evaluation and CSV export

## Important
This is a hackathon demonstration using synthetic data. It is not a production banking system, does not connect to payment rails, and must not be used to make real customer decisions.
