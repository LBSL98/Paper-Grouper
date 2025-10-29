# 🧠 Paper Grouper

![CI](https://github.com/LBSL98/Paper-Grouper/actions/workflows/ci.yml/badge.svg)

**Paper Grouper** é uma ferramenta interativa para **organizar e agrupar automaticamente artigos científicos** em uma pasta local, gerando:
- **Clusters temáticos** (baseados em embeddings semânticos)
- **Sugestões de leitura prioritária**
- **Mapa visual de similaridade entre artigos**
- **Renomeação automática** com base em metadados
- **Exportação organizada** em subpastas

Interface gráfica feita em **PySide6 (Qt)**, com backend modular e extensível em **Python 3.11+**.

---

## 🚀 Instalação e Configuração

### 1. Pré-requisitos

- **Python 3.11+**
- **Poetry** (gerenciador de dependências)
  Instale com:
  ```bash
  pipx install poetry
````

* (Opcional) **GitHub CLI** para contribuir:

  ```bash
  sudo apt install gh
  ```

---

### 2. Clonando o repositório

```bash
cd ~
git clone https://github.com/LBSL98/Paper-Grouper.git
cd Paper-Grouper
```

---

### 3. Criando o ambiente virtual e instalando dependências

```bash
poetry install --with dev
```

Isso criará automaticamente o ambiente virtual (`.venv/`) com todas as dependências:

* **Core**: `sentence-transformers`, `scikit-learn`, `networkx`, `python-louvain`, `pyside6`
* **Dev tools**: `pytest`, `black`, `ruff`, `mypy`, `pre-commit`

---

### 4. Ativando o ambiente

```bash
poetry shell
```

ou, se quiser rodar um comando direto:

```bash
poetry run python -m paper_grouper.ui.main_window
```

---

## 💡 Uso Básico

1. Execute o programa:

   ```bash
   poetry run python -m paper_grouper.ui.main_window
   ```
2. Escolha uma pasta contendo artigos `.pdf`.
3. Clique em **Executar (Manual)**.
4. Veja o relatório textual e o **mapa de similaridade** entre artigos.
5. Os resultados e os arquivos agrupados serão salvos automaticamente em uma nova pasta:

   ```
   ~/pdfs_grouped/
   ```

---

## 🧩 Estrutura do Projeto

```
paper_grouper/
├── core/           # Lógica principal (embedding, clustering, scoring)
├── io/             # Entrada/saída de arquivos e relatórios
├── ui/             # Interface PySide6 (MainWindow)
├── app_controller.py
├── app_entry.py
tests/
.github/workflows/  # CI com pytest, ruff, black
```

---

## 🧠 Como contribuir

1. Faça um fork do projeto.
2. Crie um branch:

   ```bash
   git checkout -b feat/nome-da-feature
   ```
3. Faça suas modificações.
4. Garanta que o código está formatado:

   ```bash
   poetry run pre-commit run --all-files
   ```
5. Execute os testes:

   ```bash
   poetry run pytest -q
   ```
6. Faça o commit e envie:

   ```bash
   git commit -m "feat: breve descrição da mudança"
   git push origin feat/nome-da-feature
   ```
7. Abra um **Pull Request** para `main`.

---

## 🧪 Testes e Qualidade

O projeto usa:

* **pytest** → testes automatizados
* **ruff** → linting rápido
* **black** → formatação consistente
* **pre-commit hooks** → evita código sujo em commits

Você pode rodar todos localmente com:

```bash
poetry run pytest
poetry run ruff check .
poetry run black --check .
```

---

## 📸 Captura de tela

*(Adicione uma imagem da interface quando desejar)*

```
[ Painel esquerdo: Relatório textual ]
[ Painel direito: Grafo de similaridade ]
```

---

## 📚 Roadmap

* [x] Interface manual com agrupamento e visualização
* [ ] Threading para execução sem travar a UI
* [ ] Extração avançada de metadados (título/abstract)
* [ ] Geração automática de `.bib` e planilha de resumo
* [ ] Modo CLI sem interface
* [ ] Exportação interativa (JSON, CSV, LaTeX)

---

## ⚖️ Licença

Distribuído sob a licença **MIT**.
Consulte o arquivo [LICENSE](LICENSE) para mais informações.

---

## 👥 Equipe & Créditos

**Autor:** Matheus Augusto de Paula Oliveira
**Versão atual:** `v0.1.0`
**Instituição:** UTFPR – Engenharia de Computação
**Contato:** [github.com/LBSL98](https://github.com/LBSL98)

---
