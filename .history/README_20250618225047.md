# QuantumWeaver: Zero-Point Energy Neural Network Platform

QuantumWeaver is a cutting-edge platform for Zero-Point Energy (ZPE) neural network research, training, and visualization. It features a robust backend for job management and data analysis, and a beautiful, modern frontend for interactive dashboards, visualizations, and model configuration.

## Features
- **ZPE Neural Network Training**: Launch, monitor, and analyze advanced ZPE and HS-QNN jobs.
- **Real-Time Dashboard**: Always up-to-date with the latest job results and metrics.
- **Beautiful Visualizations**: Interactive Bloch Sphere, ZPE Particle Sim, and more, with neon/glassmorphism design.
- **Mini Advisor**: Floating, draggable, and resizable HS-QNN advisor for quick parameter tweaks.
- **Responsive & Accessible UI**: Mobile-friendly, dark/light mode, and a11y best practices.
- **Log Management**: Merge, validate, and analyze training logs with robust scripts.
- **Developer Experience**: TypeScript, Tailwind, ESLint, Prettier, and Python type hints.

## Quick Start

1. **Install dependencies**
   ```bash
   npm install
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the app**
   ```bash
   npm run dev
   # In another terminal (for backend)
   python app.py
   ```

3. **Open in browser**
   Visit [http://localhost:9002](http://localhost:9002)

## Development
- **Frontend**: Next.js, React, Tailwind, shadcn/ui, p5.js for visualizations.
- **Backend**: Python, FastAPI, PyTorch, pandas, joblib.
- **Linting/Formatting**: ESLint, Prettier, and PEP8/Black for Python.
- **Testing**: (TODO) Add Jest/Vitest for frontend, pytest for backend.

## Contribution
- Fork and clone the repo.
- Use feature branches and submit pull requests.
- Run `npm run lint` and `npm run typecheck` before committing.
- Add/expand tests for new features.

## License
MIT
