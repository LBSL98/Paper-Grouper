from __future__ import annotations

import traceback
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from paper_grouper import app_controller


class MainWindow(QMainWindow):
    """
    Janela principal do Paper Grouper.

    Metas de usabilidade:
    - Fluxo mental do usuário: "1) escolho dados, 2) ajusto estratégia, 3) executo, 4) vejo resultado".
    - Evitar jargão pesado. Onde tem termo técnico (ex: resolução Louvain), explicar.
    - Mostrar resultado como relatório de leitura, não só log técnico.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paper Grouper — Agrupamento Automático de Artigos")
        self.setMinimumSize(1000, 650)

        #
        # === ROOT STRUCTURE (main_widget / main_layout) ===
        #
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        #
        # === TOP AREA: diretórios + opções gerais ===
        #
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_widget.setLayout(top_layout)

        # Pasta de entrada
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Selecione a pasta com os PDFs de entrada")
        btn_input = QPushButton("Escolher…")
        btn_input.clicked.connect(self._choose_input_dir)

        input_box = QGroupBox("Pasta de entrada")
        input_box_layout = QHBoxLayout()
        input_box_layout.addWidget(self.input_edit, stretch=1)
        input_box_layout.addWidget(btn_input)
        input_box.setLayout(input_box_layout)

        # Pasta de saída
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(
            "Pasta de saída (opcional). Se vazio, será criada *_grouped"
        )
        btn_output = QPushButton("Escolher…")
        btn_output.clicked.connect(self._choose_output_dir)

        output_box = QGroupBox("Pasta de saída")
        output_box_layout = QHBoxLayout()
        output_box_layout.addWidget(self.output_edit, stretch=1)
        output_box_layout.addWidget(btn_output)
        output_box.setLayout(output_box_layout)

        # Opções gerais
        self.rename_checkbox = QCheckBox("Renomear PDFs usando o título detectado")
        self.rename_checkbox.setChecked(True)

        general_box = QGroupBox("Opções gerais")
        general_layout = QVBoxLayout()
        general_layout.addWidget(self.rename_checkbox)
        general_box.setLayout(general_layout)

        # Monta a barra superior
        top_layout.addWidget(input_box, stretch=2)
        top_layout.addWidget(output_box, stretch=2)
        top_layout.addWidget(general_box, stretch=1)

        #
        # === ABA MANUAL ===
        #
        manual_tab = QWidget()
        manual_layout = QVBoxLayout()
        manual_tab.setLayout(manual_layout)

        manual_form_box = QGroupBox("Configuração Manual")
        manual_form_layout = QFormLayout()

        self.k_spin = QSpinBox()
        self.k_spin.setMinimum(1)
        self.k_spin.setMaximum(100)
        self.k_spin.setValue(8)
        self.k_spin.setToolTip(
            "Número de vizinhos mais próximos usados no grafo de similaridade."
        )

        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setDecimals(2)
        self.resolution_spin.setSingleStep(0.1)
        self.resolution_spin.setMinimum(0.1)
        self.resolution_spin.setMaximum(5.0)
        self.resolution_spin.setValue(1.0)
        self.resolution_spin.setToolTip(
            "Resolução do algoritmo de comunidade (Louvain). "
            "Maior => mais clusters menores."
        )

        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setMinimum(1)
        self.min_cluster_spin.setMaximum(50)
        self.min_cluster_spin.setValue(4)
        self.min_cluster_spin.setToolTip(
            "Clusters menores que isso serão anexados ao cluster mais próximo."
        )

        manual_form_layout.addRow("k-NN (vizinhos por nó):", self.k_spin)
        manual_form_layout.addRow("Resolução Louvain:", self.resolution_spin)
        manual_form_layout.addRow("Tamanho mínimo de cluster:", self.min_cluster_spin)

        manual_form_box.setLayout(manual_form_layout)

        self.btn_run_manual = QPushButton("Executar (Manual)")
        self.btn_run_manual.setStyleSheet("font-weight: bold;")
        self.btn_run_manual.clicked.connect(self._run_manual_clicked)

        manual_layout.addWidget(manual_form_box)
        manual_layout.addWidget(self.btn_run_manual, alignment=Qt.AlignRight)
        manual_layout.addStretch()

        #
        # === ABA AUTOMÁTICO (AUTO-TUNE) ===
        #
        auto_tab = QWidget()
        auto_layout = QVBoxLayout()
        auto_tab.setLayout(auto_layout)

        auto_form_box = QGroupBox("Configuração Automática (Auto-Tune)")
        auto_form_layout = QFormLayout()

        self.k_values_edit = QLineEdit()
        self.k_values_edit.setPlaceholderText("ex: 5,8,10,12")
        self.k_values_edit.setText("5,8,10,12")

        self.resolutions_edit = QLineEdit()
        self.resolutions_edit.setPlaceholderText("ex: 0.8,1.0,1.2")
        self.resolutions_edit.setText("0.8,1.0,1.2")

        self.min_cluster_values_edit = QLineEdit()
        self.min_cluster_values_edit.setPlaceholderText("ex: 3,4,5")
        self.min_cluster_values_edit.setText("3,4,5")

        self.workers_spin = QSpinBox()
        self.workers_spin.setMinimum(1)
        self.workers_spin.setMaximum(32)
        self.workers_spin.setValue(4)
        self.workers_spin.setToolTip(
            "Quantas configurações testar em paralelo no modo automático."
        )

        auto_form_layout.addRow(
            "Lista de k (separado por vírgula):", self.k_values_edit
        )
        auto_form_layout.addRow("Lista de resoluções Louvain:", self.resolutions_edit)
        auto_form_layout.addRow(
            "Lista de tamanhos mínimos de cluster:", self.min_cluster_values_edit
        )
        auto_form_layout.addRow("Trabalhadores paralelos:", self.workers_spin)

        auto_form_box.setLayout(auto_form_layout)

        self.btn_run_auto = QPushButton("Executar (Auto)")
        self.btn_run_auto.setStyleSheet("font-weight: bold;")
        self.btn_run_auto.clicked.connect(self._run_auto_clicked)

        auto_layout.addWidget(auto_form_box)
        auto_layout.addWidget(self.btn_run_auto, alignment=Qt.AlignRight)
        auto_layout.addStretch()

        #
        # === TABS ===
        #
        tabs = QTabWidget()
        tabs.addTab(manual_tab, "Manual")
        tabs.addTab(auto_tab, "Automático")

        #
        # === ÁREA DE RESULTADOS: DUAS COLUNAS LADO A LADO ===
        #

        # Caixa da esquerda: relatório em texto
        report_box = QGroupBox("Relatório")
        report_layout = QVBoxLayout()
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText(
            "Aqui você vai ver:\n"
            "- Caminho da pasta de saída\n"
            "- Qualidade do agrupamento\n"
            "- Sugestão de leitura por cluster\n"
            "- Detalhamento dos clusters\n"
        )
        report_layout.addWidget(self.result_view)
        report_box.setLayout(report_layout)

        # Caixa da direita: visualização do grafo
        graph_box = QGroupBox("Mapa de Similaridade (clusters)")
        graph_layout = QVBoxLayout()

        self.graph_label = QLabel("O grafo aparecerá aqui após a execução.")
        self.graph_label.setAlignment(Qt.AlignCenter)

        self.graph_scroll = QScrollArea()
        self.graph_scroll.setWidgetResizable(True)
        self.graph_scroll.setWidget(self.graph_label)

        graph_layout.addWidget(self.graph_scroll)
        graph_box.setLayout(graph_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(report_box)
        splitter.addWidget(graph_box)
        splitter.setSizes([600, 400])
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #cccccc;
            }
            """
        )

        #
        # === MONTAGEM FINAL NO main_layout ===
        #
        main_layout.addWidget(top_widget)
        main_layout.addWidget(tabs)
        main_layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Helpers de UI
    # ------------------------------------------------------------------

    def _choose_input_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecione a pasta com os PDFs de entrada",
            "",
        )
        if path:
            self.input_edit.setText(path)

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Selecione a pasta onde os clusters serão salvos",
            "",
        )
        if path:
            self.output_edit.setText(path)

    def _append_result(self, text: str):
        """Mostra texto no painel de resultado de um jeito cumulativo."""
        self.result_view.append(text)
        cursor = self.result_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.result_view.setTextCursor(cursor)
        self.result_view.ensureCursorVisible()

    def _clear_result(self):
        self.result_view.clear()
        self.graph_label.setText("O grafo aparecerá aqui após a execução.")
        self.graph_label.setPixmap(QPixmap())

    # ------------------------------------------------------------------
    # Execução MANUAL
    # ------------------------------------------------------------------

    def _run_manual_clicked(self):
        self._clear_result()
        self._append_result("Executando agrupamento manual...\n")

        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip() or None
        rename_flag = self.rename_checkbox.isChecked()

        k = self.k_spin.value()
        resolution = self.resolution_spin.value()
        min_cluster = self.min_cluster_spin.value()

        if not input_dir:
            self._append_result("ERRO: Selecione a pasta de entrada.\n")
            return

        try:
            result = app_controller.run_manual(
                input_dir=input_dir,
                output_dir=output_dir,
                k=k,
                resolution=resolution,
                min_cluster_size=min_cluster,
                rename_with_title=rename_flag,
            )
            self._render_result(result, mode="manual")
        except Exception:
            self._append_result("FALHOU:\n" + traceback.format_exc())

    # ------------------------------------------------------------------
    # Execução AUTO (autotune)
    # ------------------------------------------------------------------

    def _parse_int_list(self, raw: str) -> List[int]:
        vals = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if chunk:
                vals.append(int(chunk))
        return vals

    def _parse_float_list(self, raw: str) -> List[float]:
        vals = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if chunk:
                vals.append(float(chunk))
        return vals

    def _run_auto_clicked(self):
        self._clear_result()
        self._append_result("Executando agrupamento automático (auto-tune)...\n")

        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip() or None
        rename_flag = self.rename_checkbox.isChecked()

        if not input_dir:
            self._append_result("ERRO: Selecione a pasta de entrada.\n")
            return

        try:
            k_values = self._parse_int_list(self.k_values_edit.text())
            resolutions = self._parse_float_list(self.resolutions_edit.text())
            min_cluster_values = self._parse_int_list(
                self.min_cluster_values_edit.text()
            )
            workers = self.workers_spin.value()

            result = app_controller.run_auto(
                input_dir=input_dir,
                output_dir=output_dir,
                k_values=k_values,
                resolutions=resolutions,
                min_cluster_sizes=min_cluster_values,
                max_workers=workers,
                rename_with_title=rename_flag,
            )
            self._render_result(result, mode="auto")
        except Exception:
            self._append_result("FALHOU:\n" + traceback.format_exc())

    # ------------------------------------------------------------------
    # Renderização do resultado na interface
    # ------------------------------------------------------------------

    def _render_result(self, result_dict: dict, mode: str):
        """
        Mostra um relatório amigável:
        - pasta criada
        - melhor config (no auto)
        - score_final
        - clusters detectados
        - top artigos de cada cluster
        - e carrega o grafo no painel da direita
        """
        self._clear_result()

        out_dir = result_dict.get("output_root", "<?>")
        self._append_result(f"Saída gerada em:\n  {out_dir}\n")

        summary = result_dict.get("summary", {})
        score_final = summary.get("score_final", None)
        n_clusters = summary.get("n_clusters", None)

        self._append_result("\nQualidade do agrupamento:")
        self._append_result(f"- Score final: {score_final}")
        self._append_result(f"- Nº de clusters: {n_clusters}")
        self._append_result(
            f"- Fração maior cluster: {summary.get('max_cluster_fraction')}"
        )
        self._append_result(f"- Modularity: {summary.get('modularity')}")
        self._append_result(f"- Balance score: {summary.get('balance_score')}")

        if mode == "auto":
            best_cfg = result_dict.get("best_cfg", {})
            self._append_result("\nMelhor configuração encontrada (auto-tune):")
            self._append_result(f"  k = {best_cfg.get('k')}")
            self._append_result(f"  resolução = {best_cfg.get('resolution')}")
            self._append_result(
                f"  min_cluster_size = {best_cfg.get('min_cluster_size')}"
            )

        clustering = result_dict.get("clustering")
        articles_by_id = result_dict.get("articles", {})

        if clustering:
            # sugestões de leitura
            self._append_result("\nSugestão de leitura inicial (por cluster):")
            for cid, members in clustering.clusters.items():
                label = clustering.cluster_labels.get(cid, f"cluster_{cid}")
                ranked = sorted(
                    members,
                    key=lambda aid: clustering.centrality.get(aid, 0.0),
                    reverse=True,
                )
                self._append_result(
                    f"\n▶ {label}  (Cluster {cid}, {len(members)} artigos)"
                )
                for aid in ranked[:2]:
                    art = articles_by_id.get(aid)
                    if not art:
                        continue
                    yr = art.year if art.year is not None else "s/ano"
                    self._append_result(f"   • {art.title} ({yr})")

            # listagem detalhada
            self._append_result("\nDetalhamento dos clusters:")
            for cid, members in clustering.clusters.items():
                label = clustering.cluster_labels.get(cid, f"cluster_{cid}")
                self._append_result(
                    f"\n▶ Cluster {cid} :: {label} (size={len(members)})"
                )
                ranked = sorted(
                    members,
                    key=lambda aid: clustering.centrality.get(aid, 0.0),
                    reverse=True,
                )
                for aid in ranked[:5]:
                    art = articles_by_id.get(aid)
                    if not art:
                        continue
                    yr = art.year if art.year is not None else "s/ano"
                    self._append_result(f"  - {art.title} ({yr}) [{aid}]")

        # carregar imagem do grafo
        graph_path = result_dict.get("graph_png")
        if graph_path:
            pix = QPixmap(graph_path)
            if not pix.isNull():
                scaled = pix.scaledToWidth(600, Qt.SmoothTransformation)
                self.graph_label.setPixmap(scaled)
                self.graph_label.setText("")
            else:
                self.graph_label.setText("Não consegui carregar a imagem do grafo.")
        else:
            self.graph_label.setText("Nenhuma imagem de grafo gerada.")

        # dump técnico (debug)
        self._append_result("\n---\nDetalhes completos (debug):")
        import pprint

        self._append_result(pprint.pformat(result_dict, width=100))


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
