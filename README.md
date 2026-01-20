# Aadhaar Pulse
## Child Update Intelligence Platform

*Fusion-driven, AI-first decision support for proactive, equitable Aadhaar service delivery*

---
## IMAGE
<img width="2294" height="1274" alt="Screenshot 2026-01-20 193622" src="https://github.com/user-attachments/assets/7c780591-2b4d-4b04-b350-a6254a570d1d" />
<img width="2282" height="1271" alt="Screenshot 2026-01-20 192916" src="https://github.com/user-attachments/assets/4bbf1a68-b1c7-406c-afff-3e6892742158" />
<img width="2297" height="1267" alt="Screenshot 2026-01-20 192932" src="https://github.com/user-attachments/assets/d46b2dd8-16d3-4a21-8163-0689b97c06f4" />
<img width="2299" height="1267" alt="Screenshot 2026-01-20 192942" src="https://github.com/user-attachments/assets/8e09eab5-9296-494d-9862-63ef146495db" />
<img width="2298" height="1271" alt="Screenshot 2026-01-20 192953" src="https://github.com/user-attachments/assets/00ac1f9d-61bf-4d7e-8a55-26830e922c53" />
<img width="2298" height="1273" alt="Screenshot 2026-01-20 193050" src="https://github.com/user-attachments/assets/ec69791c-34a5-434e-b0d0-ba0de6a72ce2" />
<img width="2299" height="1273" alt="Screenshot 2026-01-20 193105" src="https://github.com/user-attachments/assets/89069f7f-a480-4a52-ad9f-ef701835fd5a" />
<img width="2323" height="1271" alt="Screenshot 2026-01-20 193235" src="https://github.com/user-attachments/assets/aa15bef3-3e0b-4952-a771-488f84cee295" />
<img width="2325" height="1272" alt="Screenshot 2026-01-20 193255" src="https://github.com/user-attachments/assets/7d91e1df-b14d-4bbf-9f59-f1b6ca199d48" />
<img width="2322" height="1273" alt="Screenshot 2026-01-20 193305" src="https://github.com/user-attachments/assets/9fe3ce9a-bdb9-4d6c-b6f5-f8ed8d344b88" />
<img width="2299" height="1275" alt="Screenshot 2026-01-20 193328" src="https://github.com/user-attachments/assets/97ea689b-6e49-4772-8b57-c33c737997d1" />
<img width="2287" height="1269" alt="Screenshot 2026-01-20 193340" src="https://github.com/user-attachments/assets/19fa8f64-bf19-48af-a9b1-7b35d289edbf" />
<img width="2278" height="1265" alt="Screenshot 2026-01-20 193555" src="https://github.com/user-attachments/assets/88cbd793-dfe9-469d-8f51-cdb103b8b6b1" />
<img width="2294" height="1274" alt="Screenshot 2026-01-20 193622" src="https://github.com/user-attachments/assets/db4dc37f-9a48-4157-97ee-d5cde164cc5e" />


## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repo-url>
   cd aadhaar-pulse
   ```

2. **Create Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate   # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   - Create a `.env` file in the root directory (see `.env.example`).
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   - Get key from: [Google AI Studio](https://aistudio.google.com/apikey)

5. **Run Dashboard**
   ```bash
   streamlit run app/dashboard.py
   ```

**Demo Credentials:**
- Analyst: `analyst` / `analyst123` (full access)
- Viewer: `viewer` / `viewer123` (read-only)

---

## 📁 Project Structure

```
aadhaar-pulse/
├── app/                    # Streamlit dashboards
│   ├── dashboard.py       # Main unified dashboard
│   ├── components/        # Components (incl. chatbot_view.py)
│   └── utils/             # Utilities (auth, data_loader)
├── api/                    # FastAPI endpoints
│   └── main.py            # /chat, /forecast, /recommend
├── src/                    # Core analytics modules
│   ├── chatbot.py         # Gemini AI Chatbot Engine
│   ├── agg_etl.py         # ETL pipeline
│   ├── privacy_guard.py   # K-anonymity enforcement
│   ├── features.py        # Feature engineering
│   ├── simulator.py       # Monte Carlo policy simulator
│   └── ...
├── tests/                  # Test Suite
│   ├── test_chatbot.py
│   ├── test_api.py
│   └── ...
├── config/                 # Intervention definitions
├── outputs/                # Generated files
└── data/                   # Raw and processed data
```

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **AI Assistant** | **(New)** Chat with your data using Gemini 2.5 Flash. Ask about priorities, forecasts, and interventions. |
| **Bottleneck Fusion** | 5 diagnostic types: Operational, Demographic Surge, Capacity Strain, Inclusion Gap, Anomaly |
| **District Comparison** | Multi-district side-by-side analysis, trend classification, and peer benchmarking |
| **Demand Forecasting** | LightGBM with hierarchical reconciliation, SMAPE <70% |
| **Policy Simulator** | Monte Carlo simulation with 90% confidence intervals |
| **Privacy-First** | k=10 anonymity, SHA-256 hashing, differential privacy exports |
| **RBAC** | Analyst (full) / Viewer (masked) role-based access |

---

## 🧪 Running Tests

The project includes a comprehensive test suite using `pytest`.

```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/test_chatbot.py
python -m pytest tests/test_api.py

# Run with verbose output
python -m pytest tests/ -v
```

---

## 🛠️ Run Commands

```bash
# Main dashboard (with Chatbot)
streamlit run app/dashboard.py

# FastAPI Backend
uvicorn api.main:app --reload

# Run full pipeline
python src/agg_etl.py
python src/features.py  
python src/forecast_lightgbm.py

# Run Test Suite
python -m pytest tests/
```

---

## 📊 Dashboards

| Tab | Features |
|-----|----------|
| 📍 Hotspot Map | Priority visualization (Heatmap/Scatter), state summaries |
| 🔍 District Analysis | SHAP explanations, action recommendations |
| 📊 Compare Districts | Side-by-side comparison, radar charts, trend analysis |
| 🎮 Policy Simulator | Intervention testing with Monte Carlo & 90% CIs |
| 📊 Overview | Child metrics, bottleneck distribution, top 10 list |
| 📈 Pilot Monitor | Treatment vs Control trends, action tracker |
| 🔧 System Health | Data drift (PSI), MAPE trends, alerts |
| 🤖 AI Assistant | **(New)** Interactive Q&A, priority summaries, quick insights |

---

## 🔒 Privacy & Security

- **K-anonymity:** k=10 threshold, all values <10 suppressed
- **Hashing:** SHA-256 with cryptographic salt
- **Differential Privacy:** Laplace/Gaussian noise for exports (ε=1.0)
- **Safe Chat:** AI safety settings enabled for government data context

---

## 📖 Documentation

- [Pilot Charter](docs/pilot_charter.md)
- [SLA & Alerts](docs/SLA.md)
- [Privacy Checklist](docs/privacy_checklist.md)
- [Runbooks](docs/runbooks/)

---

## 🏆 Built for UIDAI Hackathon 2025

**Judge Pitch:**
> *"Fusion-driven bottleneck diagnosis, demand forecasting, and policy simulation—now with GenAI-powered conversational insights. Integrated, privacy-aware, and production-ready."*
