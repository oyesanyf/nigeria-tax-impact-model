# Contributing to Nigeria Tax Policy Impact Model

Thank you for your interest in contributing to the Nigeria Tax Policy Impact Model! We welcome contributions from economists, data scientists, and developers.

## 🛣️ How to Contribute

1.  **Fork the Repo**: Create your own copy of the project.
2.  **Create a Branch**: `git checkout -b feature/amazing-feature`
3.  **Make Changes**: Implement your feature or fix.
4.  **Test**: Run `python tax_model.py --skip-fetch --simulations 100` to ensure no regressions.
5.  **Commit**: `git commit -m 'Add some amazing feature'`
6.  **Push**: `git push origin feature/amazing-feature`
7.  **Open a Pull Request**: Submit your changes for review.

## 🧪 Development Guidelines

### Coding Standards
- Follow PEP 8 for Python code.
- Use meaningful variable names (e.g., `sim_oil_price` instead of `x`).
- Document all new functions with docstrings.

### Data Privacy
- Never commit raw datasets containing sensitive or non-public government information.
- Use the provided `process_dataset.py` to handle data cleaning and normalization.

### Model Validation
- If adding a new feature to the model, provide the $R^2$ score and a brief explanation of why the feature is economically relevant.

## 🐞 Reporting Bugs
If you find a bug, please open an issue with:
- A clear description of the problem.
- Steps to reproduce.
- Your Python version and OS.

## 💡 Feature Requests
We are always looking to improve! If you have ideas for new policy scenarios or better data sources, please open an issue or start a discussion.

---
*Happy Modeling!*
