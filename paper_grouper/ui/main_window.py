from __future__ import annotations

import contextlib
import traceback
from pathlib import Path
from typing import Any, Callable

import shiboken6
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
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


# -----------------------------------------------------------------------------
# Worker genérico para rodar em QThread
# -----------------------------------------------------------------------------
class Worker(QObject):
    finished = Signal(dict)
    error = Signal(str)
    progressed = Signal(str)

    def __init__(self, fn: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self._fn = fn

    @Slot()
    def run(self) -> None:
        try:
            # dica: se o _fn suportar callbacks, injete um callback que cheque interrupção
            res = self._fn()
            self.finished.emit(res or {})
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)
        # sem else: sempre cairá no finally
        finally:
            pass  # nada a fazer aqui além da garantia de emitir sinais


# -----------------------------------------------------------------------------
# Janela principal
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def _safe_thread_running(self) -> bool:
        t = getattr(self, "_thread", None)
        try:
            return (t is not None) and shiboken6.isValid(t) and t.isRunning()
        except Exception:
            return False

    def _cleanup_qobjects(self) -> None:
        # encerra e solta referencias com segurança
        with contextlib.suppress(Exception):
            t = getattr(self, "_thread", None)
            if t is not None and shiboken6.isValid(t):
                t.quit()
                t.wait(1500)
        with contextlib.suppress(Exception):
            w = getattr(self, "_worker", None)
            if w is not None and shiboken6.isValid(w):
                w.deleteLater()
        self._thread = None
        self._worker = None

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Paper Grouper")
        self.setMinimumSize(1050, 640)

        self._last_graph_path: Path | None = None
        self._thread: QThread | None = None

        root = QWidget(self)
        self.setCentralWidget(root)

        # Splitter principal: esquerda (form) | direita (preview)
        splitter = QSplitter(Qt.Horizontal, self)

        # -----------------------
        # Esquerda: formulário com scroll
        # -----------------------
        left_form = QFormLayout()
        left_form.setLabelAlignment(Qt.AlignRight)

        # Pasta de entrada / saída
        self.input_edit = QLineEdit()
        btn_in = QPushButton("Selecionar...")
        btn_in.clicked.connect(self._pick_input_dir)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Opcional (será criada ao lado da entrada)")
        btn_out = QPushButton("Selecionar...")
        btn_out.clicked.connect(self._pick_output_dir)

        row_in = QHBoxLayout()
        row_in.addWidget(self.input_edit, 1)
        row_in.addWidget(btn_in)

        row_out = QHBoxLayout()
        row_out.addWidget(self.output_edit, 1)
        row_out.addWidget(btn_out)

        left_form.addRow(QLabel("<b>Pasta de entrada:</b>"), self._wrap_row(row_in))
        left_form.addRow(QLabel("Pasta de saída (opcional):"), self._wrap_row(row_out))

        # Parâmetros MANUAIS
        params_group = QGroupBox("Parâmetros do agrupamento")
        params_layout = QFormLayout()

        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 999)
        self.k_spin.setValue(8)

        self.res_spin = QDoubleSpinBox()
        self.res_spin.setDecimals(3)
        self.res_spin.setRange(0.01, 10.0)
        self.res_spin.setSingleStep(0.05)
        self.res_spin.setValue(1.000)

        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setRange(1, 999)
        self.min_cluster_spin.setValue(4)

        self.rename_chk = QCheckBox("Renomear PDFs pelo título extraído")
        self.rename_chk.setChecked(True)

        params_layout.addRow("k (k-NN):", self.k_spin)
        params_layout.addRow("Resolução (Louvain):", self.res_spin)
        params_layout.addRow("Tamanho mínimo do cluster:", self.min_cluster_spin)
        params_layout.addRow(self.rename_chk)
        params_group.setLayout(params_layout)

        # Parâmetros para AUTO-TUNE (listas)
        autot_group = QGroupBox("Auto-tune (valores separados por vírgula)")
        autot_layout = QFormLayout()

        self.k_list_edit = QLineEdit("3,5,8,12")
        self.res_list_edit = QLineEdit("0.8,1.0,1.2")
        self.min_list_edit = QLineEdit("3,4,5")

        autot_layout.addRow("k (lista):", self.k_list_edit)
        autot_layout.addRow("Resolução (lista):", self.res_list_edit)
        autot_layout.addRow("Min cluster (lista):", self.min_list_edit)
        autot_group.setLayout(autot_layout)

        # Botões
        btn_manual = QPushButton("Executar (Manual)")
        btn_manual.clicked.connect(self._on_run_manual)

        btn_auto = QPushButton("Executar (Auto-tune)")
        btn_auto.clicked.connect(self._on_run_autotune)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(btn_manual)
        buttons_row.addWidget(btn_auto)

        left_container = QWidget()
        left_v = QVBoxLayout(left_container)
        left_v.addLayout(left_form)
        left_v.addWidget(params_group)
        left_v.addWidget(autot_group)
        left_v.addLayout(buttons_row)
        left_v.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_container)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(420)

        splitter.addWidget(left_scroll)

        # -----------------------
        # Direita: preview + abas de texto
        # -----------------------
        right_widget = QWidget()
        right_v = QVBoxLayout(right_widget)
        right_v.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("Pré-visualização do grafo aparecerá aqui")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #111; color: #BBB;")
        self.preview_label.setMinimumHeight(380)

        tabs = QTabWidget()
        self.summary_edit = QTextEdit()
        self.summary_edit.setReadOnly(True)
        self.details_edit = QTextEdit()
        self.details_edit.setReadOnly(True)
        tabs.addTab(self.summary_edit, "Resumo")
        tabs.addTab(self.details_edit, "Detalhes")

        right_v.addWidget(self.preview_label, 3)
        right_v.addWidget(tabs, 2)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Layout principal
        outer = QVBoxLayout()
        outer.addWidget(splitter)
        root.setLayout(outer)

        # Status bar
        self.statusBar().showMessage("Pronto.")

    def _thread_is_valid(self) -> bool:
        t = getattr(self, "_thread", None)
        try:
            return t is not None and shiboken6.isValid(t) and t.isRunning()
        except RuntimeError:
            return False

    def _cleanup_thread(self) -> None:
        # Desconecta e coleta; tolera objetos já destruídos
        t = getattr(self, "_thread", None)
        w = getattr(self, "_worker", None)
        if w and shiboken6.isValid(w):
            with contextlib.suppress(Exception):
                w.finished.disconnect()
            with contextlib.suppress(Exception):
                w.error.disconnect()

        with contextlib.suppress(Exception):
            if t and shiboken6.isValid(t):
                t.quit()
                t.wait(2000)

    # -----------------------
    # Utilitários de UI
    # -----------------------
    @staticmethod
    def _wrap_row(layout: QHBoxLayout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _pick_input_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Escolha a pasta com PDFs")
        if d:
            self.input_edit.setText(d)

    def _pick_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Escolha a pasta de saída")
        if d:
            self.output_edit.setText(d)

    def _append_summary(self, text: str) -> None:
        self.summary_edit.moveCursor(QTextCursor.End)
        self.summary_edit.insertPlainText(text)
        self.summary_edit.moveCursor(QTextCursor.End)

    def _append_details(self, text: str) -> None:
        self.details_edit.moveCursor(QTextCursor.End)
        self.details_edit.insertPlainText(text)
        self.details_edit.moveCursor(QTextCursor.End)

    def _clear_outputs(self) -> None:
        self.summary_edit.clear()
        self.details_edit.clear()
        self.preview_label.setText("Pré-visualização do grafo aparecerá aqui")
        self._last_graph_path = None

    def _set_graph_preview(self, path: Path) -> None:
        self._last_graph_path = path
        pm = QPixmap(str(path))
        if pm.isNull():
            self.preview_label.setText(f"Não foi possível carregar a imagem:\n{path}")
            return
        self._scale_preview(pm)

    def resizeEvent(self, e) -> None:  # type: ignore[override]
        super().resizeEvent(e)
        if self._last_graph_path and self._last_graph_path.exists():
            pm = QPixmap(str(self._last_graph_path))
            if not pm.isNull():
                self._scale_preview(pm)

    def _scale_preview(self, pix: QPixmap) -> None:
        lbl = self.preview_label
        s = lbl.size()
        scaled = pix.scaled(s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(scaled)

    # -----------------------
    # Execuções
    # -----------------------
    def _on_run_manual(self) -> None:
        input_dir = self.input_edit.text().strip()
        if not input_dir:
            self.statusBar().showMessage("Selecione a pasta de entrada.")
            return

        output_dir = self.output_edit.text().strip() or None
        k = int(self.k_spin.value())
        res = float(self.res_spin.value())
        min_sz = int(self.min_cluster_spin.value())
        rename = self.rename_chk.isChecked()

        self._clear_outputs()
        self._append_summary("Executando (manual)...\n")
        self.statusBar().showMessage("Executando (manual)...")

        def job() -> dict[str, Any]:
            return app_controller.run_manual(
                input_dir=input_dir,
                output_dir=output_dir,
                k=k,
                resolution=res,
                min_cluster_size=min_sz,
                rename=rename,
                use_light=True,  # embeddings leves por padrão
            )

        self._start_worker(job)

    def _parse_int_list(self, raw: str) -> list[int]:
        vals: list[int] = []
        for chunk in raw.split(","):
            t = chunk.strip()
            if not t:
                continue
            vals.append(int(t))
        return vals

    def _parse_float_list(self, raw: str) -> list[float]:
        vals: list[float] = []
        for chunk in raw.split(","):
            t = chunk.strip()
            if not t:
                continue
            vals.append(float(t))
        return vals

    def _on_run_autotune(self) -> None:
        input_dir = self.input_edit.text().strip()
        if not input_dir:
            self.statusBar().showMessage("Selecione a pasta de entrada.")
            return

        output_dir = self.output_edit.text().strip() or None
        rename = self.rename_chk.isChecked()

        try:
            k_vals = self._parse_int_list(self.k_list_edit.text())
            r_vals = self._parse_float_list(self.res_list_edit.text())
            m_vals = self._parse_int_list(self.min_list_edit.text())
        except ValueError:
            self.statusBar().showMessage("Listas inválidas no Auto-tune.")
            return

        if not k_vals or not r_vals or not m_vals:
            self.statusBar().showMessage("Forneça listas não vazias para o Auto-tune.")
            return

        self._clear_outputs()
        self._append_summary("Executando (auto-tune)...\n")
        self.statusBar().showMessage("Executando (auto-tune)...")

        def job() -> dict[str, Any]:
            return app_controller.run_autotune(
                input_dir=input_dir,
                output_dir=output_dir,
                k_values=k_vals,
                resolution_values=r_vals,
                min_cluster_values=m_vals,
                rename=rename,
                use_light=True,
                max_workers=0,  # auto
            )

        self._start_worker(job)

    # -----------------------
    # Thread helper
    # -----------------------
    def _start_worker(self, job: Callable[[], dict[str, Any]]) -> None:
        # encerra execução anterior, se houver
        if self._safe_thread_running():
            self._thread.requestInterruption()
            self._thread.quit()
            self._thread.wait(5_000)  # até 5s

        self._thread = QThread(self)
        self._worker = Worker(job)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_failed)
        self._worker.finished.connect(self._cleanup_qobjects)
        self._worker.error.connect(self._cleanup_qobjects)
        self._worker.error.connect(self._thread.quit)
        self._worker.error.connect(self._worker.deleteLater)
        self._worker.progressed.connect(lambda s: self._append_summary(s + "\n"))

        # limpeza correta
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.finished.connect(self._cleanup_thread)
        self._worker.error.connect(self._cleanup_thread)

        self._thread.start()

    # -----------------------
    # Handlers de retorno
    # -----------------------
    def _on_worker_finished(self, result: dict[str, Any]) -> None:
        self.statusBar().showMessage("Concluído.")
        try:
            # Resumo
            summary = result.get("summary", {})
            lines: list[str] = []
            if summary:
                lines.append("Qualidade do agrupamento:")
                lines.append(f"- Score final: {summary.get('score_final')}")
                lines.append(f"- Nº de clusters: {summary.get('n_clusters')}")
                lines.append(
                    f"- Fração maior cluster: {summary.get('max_cluster_fraction')}"
                )
                lines.append(f"- Modularity: {summary.get('modularity')}")
                lines.append(f"- Balance score: {summary.get('balance_score')}")
                lines.append("")
            out_dir = result.get("output_root")
            if out_dir:
                lines.append(f"Saída gerada em:\n  {out_dir}\n")

            self._append_summary("\n".join(lines) + "\n")

            # Detalhes (debug) — imprime o dicionário quase todo
            debug_keys = [
                "articles",
                "clustering",
                "autotune_trials",
                "output_root",
                "graph_png",
                "summary",
            ]
            dbg = {k: result.get(k) for k in debug_keys}
            self._append_details("Detalhes completos (debug):\n")
            self._append_details(repr(dbg) + "\n")

            # Preview do grafo
            graph_png = result.get("graph_png")
            if graph_png:
                self._set_graph_preview(Path(graph_png))
            else:
                self._append_summary("Aviso: nenhuma imagem de grafo foi fornecida.\n")

        except Exception:
            self._append_summary("Falha ao processar resultado.\n")
            self._append_details(traceback.format_exc())

    def _on_worker_failed(self, message: str) -> None:
        self.statusBar().showMessage("Falha.")
        self._append_summary("Erro durante a execução:\n")
        self._append_summary(message + "\n")

    def closeEvent(self, event):
        # tenta encerrar trabalhador/threads ativos sem explodir
        self._cleanup_qobjects()
        super().closeEvent(event)


def main() -> None:
    import sys

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
