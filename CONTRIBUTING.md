# Contribuindo

## Setup rápido
```bash
git clone https://github.com/LBSL98/Paper-Grouper.git
cd Paper-Grouper
poetry install --with dev
poetry run pre-commit install
poetry run python -m paper_grouper.ui.main_window
