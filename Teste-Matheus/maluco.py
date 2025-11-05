import sys
import os
import logging
import traceback
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import shutil
import contextlib

# Silenciador de stderr em nível de file descriptor (cobre mensagens C-level)
class _FdStderrSilencer:
    def __enter__(self):
        import sys, os
        self._fd = sys.stderr.fileno()
        self._saved = os.dup(self._fd)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, self._fd)
        return self
    def __exit__(self, exc_type, exc, tb):
        import os
        os.dup2(self._saved, self._fd)
        os.close(self._devnull)
        os.close(self._saved)

# Suprimir TODOS os warnings antes de importar numpy
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configurar NumPy ANTES de importar para evitar warnings de RNG
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
np.random.seed(42)
import numpy.random
numpy.random.seed(42)

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Silencia avisos nativos (libgcrypt/RDRAND etc.) durante os imports do Qt
with _FdStderrSilencer():
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
        QCheckBox, QTextEdit, QFileDialog, QGroupBox, QTabWidget,
        QMessageBox, QProgressBar
    )
    from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt
    from PyQt6.QtGui import QPixmap, QFont

try:
    import PyPDF2
    # Suprimir warnings do PyPDF2
    PyPDF2.PdfReader._get_warnings = lambda *args: []
    PDF_DISPONIVEL = True
except ImportError:
    PDF_DISPONIVEL = False


# ============================================================================
# MODELOS DE DADOS
# ============================================================================

@dataclass
class RegistroArtigo:
    """Representa metadados de um único PDF."""
    id: str
    titulo: str
    autores: list[str] = field(default_factory=list)
    resumo: str = ""
    ano: Optional[int] = None
    caminho_arquivo: str = ""
    
    def __str__(self) -> str:
        autores_str = ", ".join(self.autores[:3])
        if len(self.autores) > 3:
            autores_str += " et al."
        return f"{self.titulo} ({self.ano or 'N/A'}) - {autores_str}"


@dataclass
class ResultadoEmbedding:
    """Container para embeddings de artigos."""
    embeddings: np.ndarray
    ids_artigos: list[str]
    nome_modelo: str = "TF-IDF"
    
    def __len__(self) -> int:
        return len(self.ids_artigos)


@dataclass
class ResultadoAgrupamento:
    """Resultados da análise de agrupamento."""
    grupos: dict[int, list[str]]
    modularidade: Optional[float] = None
    balanceamento: Optional[float] = None
    n_grupos: int = 0
    
    def __post_init__(self):
        if self.n_grupos == 0:
            self.n_grupos = len(self.grupos)
    
    def obter_resumo(self) -> dict:
        """Retorna estatísticas resumidas."""
        return {
            "n_grupos": self.n_grupos,
            "modularidade": self.modularidade,
            "balanceamento": self.balanceamento,
            "melhor_pontuacao": None
        }


# ============================================================================
# FUNÇÕES PRINCIPAIS
# ============================================================================

def listar_pdfs(diretorio_entrada: str | Path) -> list[Path]:
    """Escaneia diretório procurando arquivos PDF."""
    caminho = Path(diretorio_entrada)
    
    if not caminho.exists():
        raise ValueError(f"Diretório não existe: {diretorio_entrada}")
    
    if not caminho.is_dir():
        raise ValueError(f"Caminho não é um diretório: {diretorio_entrada}")
    
    arquivos_pdf = list(caminho.rglob("*.pdf"))
    logging.info(f"Encontrados {len(arquivos_pdf)} arquivos PDF")
    
    return arquivos_pdf


def extrair_metadados_pdf(caminho_pdf: Path) -> RegistroArtigo:
    """Extrai metadados básicos de um arquivo PDF."""
    if not PDF_DISPONIVEL:
        return RegistroArtigo(
            id=caminho_pdf.stem,
            titulo=caminho_pdf.stem,
            caminho_arquivo=str(caminho_pdf)
        )
    
    try:
        with open(caminho_pdf, 'rb') as f:
            leitor = PyPDF2.PdfReader(f)
            
            # Extrai texto da primeira página
            primeira_pagina = ""
            if leitor.pages:
                try:
                    primeira_pagina = leitor.pages[0].extract_text() or ""
                except:
                    pass
            
            # Obtém metadados do PDF
            metadados = {}
            try:
                metadados = leitor.metadata or {}
            except:
                pass
            
            # Extrai título
            titulo = caminho_pdf.stem
            try:
                if metadados.get('/Title'):
                    titulo = metadados.get('/Title')
                elif primeira_pagina:
                    titulo = primeira_pagina.split('\n')[0][:100]
            except:
                pass
            
            # Extrai autor
            autores = []
            try:
                autor = metadados.get('/Author', '')
                if autor:
                    autores = [autor]
            except:
                pass
            
            # Usa primeiras linhas como resumo
            resumo = ""
            try:
                if primeira_pagina:
                    resumo = '\n'.join(primeira_pagina.split('\n')[:5])[:500]
            except:
                pass
            
            return RegistroArtigo(
                id=caminho_pdf.stem,
                titulo=titulo.strip(),
                autores=autores,
                resumo=resumo.strip(),
                caminho_arquivo=str(caminho_pdf)
            )
            
    except:
        return RegistroArtigo(
            id=caminho_pdf.stem,
            titulo=caminho_pdf.stem,
            caminho_arquivo=str(caminho_pdf)
        )


def extrair_lote(caminhos_pdf: list[Path]) -> dict[str, RegistroArtigo]:
    """Extrai metadados de múltiplos PDFs."""
    logging.info(f"Extraindo metadados de {len(caminhos_pdf)} PDFs...")
    
    artigos = {}
    for caminho_pdf in caminhos_pdf:
        artigo = extrair_metadados_pdf(caminho_pdf)
        artigos[artigo.id] = artigo
    
    logging.info(f"✓ Metadados extraídos com sucesso")
    return artigos


def gerar_embeddings(artigos: list[RegistroArtigo]) -> ResultadoEmbedding:
    """Gera embeddings usando TF-IDF (rápido e leve)."""
    if not artigos:
        raise ValueError("Nenhum artigo fornecido para embedding")
    
    logging.info(f"Gerando embeddings para {len(artigos)} artigos...")
    
    # Combina título e resumo
    textos = [f"{artigo.titulo}. {artigo.resumo}" for artigo in artigos]
    
    # Gera embeddings com TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    embeddings = vectorizer.fit_transform(textos).toarray()
    ids_artigos = [artigo.id for artigo in artigos]
    
    logging.info(f"✓ Embeddings gerados: {embeddings.shape}")
    
    return ResultadoEmbedding(
        embeddings=embeddings,
        ids_artigos=ids_artigos,
        nome_modelo="TF-IDF"
    )


def construir_grafo_knn(resultado_embedding: ResultadoEmbedding, k: int = 5) -> nx.Graph:
    """Constrói grafo k-vizinhos mais próximos."""
    logging.info(f"Construindo grafo k-NN com k={k}...")
    
    embeddings = resultado_embedding.embeddings
    ids_artigos = resultado_embedding.ids_artigos
    
    # Calcula similaridade cosseno
    matriz_similaridade = cosine_similarity(embeddings)
    
    # Cria grafo
    G = nx.Graph()
    
    # Adiciona nós
    for id_artigo in ids_artigos:
        G.add_node(id_artigo)
    
    # Adiciona arestas (k vizinhos mais próximos)
    n = len(ids_artigos)
    for i in range(n):
        indices_similares = np.argsort(matriz_similaridade[i])[::-1][:k+1]
        
        for j in indices_similares:
            if i != j:
                peso = float(matriz_similaridade[i, j])
                if peso > 0.01:
                    G.add_edge(ids_artigos[i], ids_artigos[j], weight=peso)
    
    logging.info(f"✓ Grafo criado: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    
    return G


def detectar_comunidades_simples(G: nx.Graph) -> dict:
    """Detecta comunidades usando algoritmo greedy."""
    logging.info("Detectando comunidades...")
    
    # Usa algoritmo greedy de modularidade (nativo NetworkX)
    comunidades = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    
    # Converte para formato de partição
    particao = {}
    for idx, comunidade in enumerate(comunidades):
        for no in comunidade:
            particao[no] = idx
    
    n_comunidades = len(comunidades)
    logging.info(f"✓ Encontradas {n_comunidades} comunidades")
    
    return particao


def finalizar_agrupamento(G: nx.Graph, particao: dict, tamanho_min_grupo: int = 3) -> ResultadoAgrupamento:
    """Finaliza agrupamento com pós-processamento."""
    logging.info("Finalizando agrupamento...")
    
    # Agrupa nós por cluster
    grupos_brutos = defaultdict(list)
    for no, id_grupo in particao.items():
        grupos_brutos[id_grupo].append(no)
    
    # Filtra grupos pequenos
    grupos = {}
    nao_agrupados = []
    
    contador_grupo = 0
    for id_grupo, nos in grupos_brutos.items():
        if len(nos) >= tamanho_min_grupo:
            grupos[contador_grupo] = nos
            contador_grupo += 1
        else:
            nao_agrupados.extend(nos)
    
    # Adiciona não agrupados
    if nao_agrupados:
        grupos[-1] = nao_agrupados
    
    # Calcula modularidade
    modularidade = None
    try:
        modularidade = nx.algorithms.community.modularity(G, grupos.values(), weight='weight')
        modularidade = round(modularidade, 3)
    except:
        pass
    
    # Calcula balanceamento
    tamanhos_grupos = [len(nos) for nos in grupos.values() if nos]
    balanceamento = None
    if tamanhos_grupos:
        balanceamento = round(1.0 / (1.0 + np.std(tamanhos_grupos)), 3)
    
    resultado = ResultadoAgrupamento(
        grupos=grupos,
        modularidade=modularidade,
        balanceamento=balanceamento,
        n_grupos=len(grupos)
    )
    
    logging.info(f"✓ Agrupamento final: {resultado.n_grupos} grupos")
    
    return resultado


def renderizar_grafo_png(G: nx.Graph, agrupamento: ResultadoAgrupamento, caminho_saida: Path) -> str:
    """Renderiza visualização do grafo em PNG."""
    logging.info("Renderizando grafo...")
    
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    
    # Cria mapa de cores
    cores_nos = {}
    cores = plt.cm.tab10(np.linspace(0, 1, max(10, agrupamento.n_grupos)))
    
    for id_grupo, nos in agrupamento.grupos.items():
        idx_cor = id_grupo % len(cores) if id_grupo >= 0 else -1
        for no in nos:
            cores_nos[no] = cores[idx_cor] if id_grupo >= 0 else [0.7, 0.7, 0.7, 1.0]
    
    lista_cores = [cores_nos.get(no, [0.5, 0.5, 0.5, 1.0]) for no in G.nodes()]
    
    # Desenha grafo
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=20)
    
    nx.draw_networkx_nodes(G, pos, node_color=lista_cores, node_size=60, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.3)
    
    plt.title(f"Grafo de Similaridade - {agrupamento.n_grupos} Grupos", fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(caminho_saida, dpi=90, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✓ Grafo salvo")
    return str(caminho_saida)

# ============================================================================
# EXTRAÇÃO DE PALAVRAS-CHAVE
# ============================================================================

def extrair_palavras_chave(textos: list[str], n_palavras: int = 5) -> list[list[str]]:
    """
    Extrai palavras-chave mais relevantes de cada texto usando TF-IDF.
    
    Args:
        textos: Lista de textos para análise
        n_palavras: Número de palavras-chave a extrair
        
    Returns:
        Lista de listas contendo as palavras-chave de cada texto
    """
    if not textos:
        logging.warning("⚠️ Nenhum texto fornecido para extração de palavras-chave")
        return []
    
    # Filtra textos vazios
    textos_validos = [t.strip() for t in textos if t and t.strip()]
    if not textos_validos:
        logging.warning("⚠️ Todos os textos estão vazios")
        return [[] for _ in textos]
    
    logging.info(f"🔍 Extraindo palavras-chave de {len(textos_validos)} textos...")
    
    try:
        # Configura TF-IDF com parâmetros mais flexíveis
        vectorizer = TfidfVectorizer(
            max_features=150,
            stop_words='english',
            ngram_range=(1, 2),  # Uni e bigramas
            min_df=1,
            max_df=0.9,
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Apenas palavras com 3+ letras
        )
        
        # Gera matriz TF-IDF
        tfidf_matrix = vectorizer.fit_transform(textos_validos)
        feature_names = vectorizer.get_feature_names_out()
        
        logging.info(f"✓ Vocabulário extraído: {len(feature_names)} termos únicos")
        
        # Extrai top palavras-chave para cada texto
        palavras_chave_por_texto = []
        for i in range(len(textos_validos)):
            # Obtém scores TF-IDF para este texto
            scores = tfidf_matrix[i].toarray().flatten()
            
            # Pega índices das top palavras
            top_indices = scores.argsort()[-n_palavras*2:][::-1]  # Pega o dobro para filtrar depois
            
            # Extrai palavras correspondentes com score > 0
            palavras = []
            for idx in top_indices:
                if scores[idx] > 0.05:  # Threshold mínimo
                    palavra = feature_names[idx]
                    # Filtra palavras muito curtas ou genéricas
                    if len(palavra) >= 4 and palavra.isalpha():
                        palavras.append(palavra)
                    if len(palavras) >= n_palavras:
                        break
            
            palavras_chave_por_texto.append(palavras[:n_palavras])
            logging.debug(f"  Texto {i+1}: {palavras[:n_palavras]}")
        
        return palavras_chave_por_texto
    
    except Exception as e:
        logging.error(f"❌ Erro ao extrair palavras-chave: {e}")
        # Fallback: retorna listas vazias
        return [[] for _ in textos]

def gerar_nome_grupo_inteligente(
    artigos: list[RegistroArtigo],
    id_grupo: int,
    n_palavras: int = 3
) -> str:
    """
    Gera nome descritivo para um grupo baseado em palavras-chave comuns.
    
    Args:
        artigos: Lista de artigos do grupo
        id_grupo: ID numérico do grupo
        n_palavras: Número de palavras no nome
        
    Returns:
        Nome descritivo do grupo
    """
    if not artigos:
        logging.warning(f"⚠️ Grupo {id_grupo} sem artigos")
        return f"Grupo_{id_grupo:02d}"
    
    logging.info(f"🏷️ Gerando nome para Grupo {id_grupo} com {len(artigos)} artigos...")
    
    # Combina títulos e resumos de TODOS os artigos do grupo
    textos_por_artigo = []
    for a in artigos:
        texto = f"{a.titulo} {a.resumo}".strip()
        if texto and len(texto) > 10:  # Ignora textos muito curtos
            textos_por_artigo.append(texto)
    
    if not textos_por_artigo:
        logging.warning(f"⚠️ Grupo {id_grupo} sem textos válidos")
        return f"Grupo_{id_grupo:02d}"
    
    # Combina todos os textos do grupo
    texto_completo_grupo = " ".join(textos_por_artigo)
    
    # Extrai palavras-chave do grupo inteiro
    palavras_chave_grupo = extrair_palavras_chave([texto_completo_grupo], n_palavras=n_palavras * 4)
    
    if not palavras_chave_grupo or not palavras_chave_grupo[0]:
        logging.warning(f"⚠️ Falha ao extrair palavras-chave do Grupo {id_grupo}")
        return f"Grupo_{id_grupo:02d}"
    
    # Pega as top N palavras
    top_palavras = palavras_chave_grupo[0][:n_palavras]
    
    logging.info(f"  Palavras-chave extraídas: {top_palavras}")
    
    # Formata nome (capitaliza e limpa)
    palavras_formatadas = []
    for palavra in top_palavras:
        # Remove caracteres especiais e capitaliza
        palavra_limpa = "".join(c for c in palavra if c.isalnum())
        palavra_limpa = palavra_limpa.strip().capitalize()
        if palavra_limpa and len(palavra_limpa) >= 4:
            palavras_formatadas.append(palavra_limpa)
    
    if palavras_formatadas:
        nome_descritivo = "_".join(palavras_formatadas)
        nome_final = f"Grupo_{id_grupo:02d}_{nome_descritivo}"
        logging.info(f"✓ Nome gerado: {nome_final}")
        return nome_final
    else:
        logging.warning(f"⚠️ Nenhuma palavra válida encontrada para Grupo {id_grupo}")
        return f"Grupo_{id_grupo:02d}"

def gerar_nome_arquivo_inteligente(
    artigo: RegistroArtigo,
    usar_palavras_chave: bool = True,
    n_palavras: int = 5
) -> str:
    """
    Gera nome de arquivo inteligente para PDF.
    
    Args:
        artigo: Registro do artigo
        usar_palavras_chave: Se True, usa palavras-chave quando título não é legível
        n_palavras: Número de palavras-chave a usar
        
    Returns:
        Nome de arquivo seguro (sem extensão)
    """
    titulo = artigo.titulo.strip()
    
    # Verifica se o título é legível (tem palavras razoáveis)
    titulo_limpo = "".join(c if c.isalnum() or c == ' ' else ' ' for c in titulo)
    palavras_titulo = [p for p in titulo_limpo.split() if len(p) > 2]
    
    # Se título for bom, usa ele
    if len(palavras_titulo) >= 3 and len(titulo) >= 20:
        titulo_seguro = "_".join(palavras_titulo[:8])  # Primeiras 8 palavras
        return titulo_seguro[:80]  # Limita tamanho
    
    # Senão, tenta palavras-chave
    if usar_palavras_chave and artigo.resumo and len(artigo.resumo) > 30:
        logging.debug(f"  Usando palavras-chave para: {artigo.id}")
        # Extrai palavras-chave do resumo/título
        texto_completo = f"{titulo} {artigo.resumo}"
        palavras_chave = extrair_palavras_chave([texto_completo], n_palavras=n_palavras)
        
        if palavras_chave and palavras_chave[0]:
            # Formata palavras-chave
            palavras_formatadas = []
            for palavra in palavras_chave[0]:
                palavra_limpa = "".join(c for c in palavra if c.isalnum())
                palavra_limpa = palavra_limpa.strip().capitalize()
                if palavra_limpa and len(palavra_limpa) >= 4:
                    palavras_formatadas.append(palavra_limpa)
            
            if palavras_formatadas:
                nome_base = "_".join(palavras_formatadas[:n_palavras])
                return nome_base[:80]  # Limita tamanho
    
    # Fallback: título simples
    titulo_seguro = "".join(
        c for c in titulo[:70]
        if c.isalnum() or c in (' ', '-', '_')
    ).strip().replace(' ', '_')
    
    # Se ainda estiver vazio, usa ID
    if not titulo_seguro or len(titulo_seguro) < 5:
        return artigo.id
    
    return titulo_seguro

def escrever_arquivos_agrupados(
    raiz_saida: Path,
    agrupamento: ResultadoAgrupamento,
    artigos_por_id: dict[str, RegistroArtigo],
    renomear_com_titulo: bool = False
) -> None:
    """Escreve PDFs agrupados no diretório de saída com nomes inteligentes."""
    logging.info("Copiando arquivos para grupos com nomes inteligentes...")
    
    raiz_saida.mkdir(parents=True, exist_ok=True)
    
    for id_grupo, ids_artigos in agrupamento.grupos.items():
        # Obtém artigos do grupo
        artigos_grupo = [artigos_por_id[id_art] for id_art in ids_artigos if id_art in artigos_por_id]
        
        # ✨ Gera nome inteligente para o grupo
        if id_grupo == -1:
            nome_grupo = "Nao_Agrupados"
        else:
            nome_grupo = gerar_nome_grupo_inteligente(artigos_grupo, id_grupo, n_palavras=3)
        
        dir_grupo = raiz_saida / nome_grupo
        dir_grupo.mkdir(exist_ok=True)
        
        logging.info(f"📁 Criando grupo: {nome_grupo} ({len(ids_artigos)} artigos)")
        
        # Copia arquivos
        for id_artigo in ids_artigos:
            artigo = artigos_por_id.get(id_artigo)
            if not artigo:
                continue
            
            origem = Path(artigo.caminho_arquivo)
            if not origem.exists():
                continue
            
            # ✨ Define nome do arquivo de destino (inteligente)
            if renomear_com_titulo:
                nome_base = gerar_nome_arquivo_inteligente(
                    artigo,
                    usar_palavras_chave=True,
                    n_palavras=4
                )
                nome_destino = f"{nome_base}.pdf"
            else:
                nome_destino = origem.name
            
            destino = dir_grupo / nome_destino
            
            # Evita duplicatas
            contador = 1
            while destino.exists():
                stem = destino.stem
                destino = dir_grupo / f"{stem}_{contador}.pdf"
                contador += 1
            
            shutil.copy2(origem, destino)
    
    logging.info(f"✓ Arquivos copiados com sucesso para grupos nomeados!")

# ============================================================================
# CONTROLADOR
# ============================================================================

class ControladorApp:
    """Controlador principal da aplicação."""
    
    def __init__(self):
        self._cancelamento_solicitado = False
    
    def solicitar_cancelamento(self):
        """Solicita cancelamento da operação atual."""
        self._cancelamento_solicitado = True
    
    def _verificar_cancelamento(self):
        """Verifica se cancelamento foi solicitado."""
        if self._cancelamento_solicitado:
            raise InterruptedError("Operação cancelada pelo usuário")
    
    def _limpar_cancelamento(self):
        """Limpa flag de cancelamento."""
        self._cancelamento_solicitado = False
    
    def executar_manual(
        self,
        dir_entrada: str,
        dir_saida: Optional[str],
        k: int,
        resolucao: float,
        grupo_min: int,
        renomear: bool
    ) -> dict:
        """Executa pipeline de agrupamento manual."""
        self._limpar_cancelamento()
        
        try:
            logging.info("=" * 60)
            logging.info("🚀 INICIANDO AGRUPAMENTO")
            logging.info("=" * 60)
            
            # Passo 1: Escanear PDFs
            self._verificar_cancelamento()
            caminhos_pdf = listar_pdfs(dir_entrada)
            
            if not caminhos_pdf:
                return {
                    "error": "Nenhum arquivo PDF encontrado",
                    "traceback": ""
                }
            
            # Passo 2: Extrair metadados
            self._verificar_cancelamento()
            artigos_por_id = extrair_lote(caminhos_pdf)
            lista_artigos = list(artigos_por_id.values())
            
            # Passo 3: Gerar embeddings
            self._verificar_cancelamento()
            resultado_emb = gerar_embeddings(lista_artigos)
            
            # Passo 4: Construir grafo
            self._verificar_cancelamento()
            G = construir_grafo_knn(resultado_emb, k=k)
            
            # Passo 5: Detectar comunidades
            self._verificar_cancelamento()
            particao = detectar_comunidades_simples(G)
            
            # Passo 6: Finalizar agrupamento
            self._verificar_cancelamento()
            agrupamento = finalizar_agrupamento(G, particao, tamanho_min_grupo=grupo_min)
            
            # Passo 7: Preparar saída
            self._verificar_cancelamento()
            if dir_saida:
                raiz_saida = Path(dir_saida)
            else:
                raiz_saida = Path(dir_entrada) / "artigos_agrupados"
            
            raiz_saida.mkdir(parents=True, exist_ok=True)
            
            # Passo 8: Renderizar grafo
            self._verificar_cancelamento()
            caminho_png = raiz_saida / "visualizacao_grafo.png"
            renderizar_grafo_png(G, agrupamento, caminho_png)
            
            # Passo 9: Escrever arquivos
            self._verificar_cancelamento()
            escrever_arquivos_agrupados(raiz_saida, agrupamento, artigos_por_id, renomear_com_titulo=renomear)
            
            # Retorna resultado
            resumo = agrupamento.obter_resumo()
            
            logging.info("=" * 60)
            logging.info("✅ AGRUPAMENTO CONCLUÍDO COM SUCESSO!")
            logging.info("=" * 60)
            
            return {
                "status": "ok",
                "caminho_png": str(caminho_png),
                "raiz_saida": str(raiz_saida),
                "resumo": resumo
            }
            
        except InterruptedError as e:
            logging.warning(f"⚠️ Operação cancelada")
            return {
                "error": "Cancelado",
                "traceback": str(e)
            }
        except Exception as e:
            msg_erro = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            logging.error(f"❌ Erro: {msg_erro}")
            return {
                "error": msg_erro,
                "traceback": tb
            }


# ============================================================================
# WORKER THREAD
# ============================================================================

class Trabalhador(QObject):
    """Worker para executar agrupamento em thread separada."""
    
    finalizado = pyqtSignal(dict)
    progresso = pyqtSignal(str)
    erro = pyqtSignal(str)
    
    def __init__(self, controlador: ControladorApp, parametros: dict):
        super().__init__()
        self.controlador = controlador
        self.parametros = parametros
    
    def executar(self):
        """Executa o trabalho."""
        try:
            resultado = self.controlador.executar_manual(**self.parametros)
            self.finalizado.emit(resultado)
        except Exception as e:
            msg_erro = f"{type(e).__name__}: {str(e)}"
            self.erro.emit(msg_erro)


# ============================================================================
# JANELA PRINCIPAL
# ============================================================================

class JanelaPrincipal(QMainWindow):
    """Janela principal da aplicação."""
    
    def __init__(self):
        super().__init__()
        
        self.controlador = ControladorApp()
        self.trabalhador = None
        self.thread = None
        
        self.iniciar_ui()
        self.configurar_logging()
        
        # Maximizar janela
        self.showMaximized()
    
    def iniciar_ui(self):
        """Inicializa interface do usuário."""
        self.setWindowTitle("📚 Agrupador de Artigos - Ferramenta de Agrupamento de PDFs")
        
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        
        layout_principal = QHBoxLayout()
        central.setLayout(layout_principal)
        
        # Painel esquerdo - Controles
        painel_esquerdo = self.criar_painel_esquerdo()
        layout_principal.addWidget(painel_esquerdo, stretch=1)
        
        # Painel direito - Preview e logs
        painel_direito = self.criar_painel_direito()
        layout_principal.addWidget(painel_direito, stretch=3)
    
    def criar_painel_esquerdo(self) -> QWidget:
        """Cria painel de controle esquerdo."""
        painel = QWidget()
        layout = QVBoxLayout()
        painel.setLayout(layout)
        
        # Título
        titulo = QLabel("📚 Agrupador de Artigos")
        fonte_titulo = QFont()
        fonte_titulo.setPointSize(18)
        fonte_titulo.setBold(True)
        titulo.setFont(fonte_titulo)
        titulo.setStyleSheet("color: #2196F3; padding: 10px;")
        layout.addWidget(titulo)
        
        subtitulo = QLabel("Agrupe PDFs automaticamente por similaridade")
        subtitulo.setStyleSheet("color: #666; padding-bottom: 15px;")
        layout.addWidget(subtitulo)
        
        # Diretório de entrada
        grupo_entrada = QGroupBox("📁 Diretório de Entrada")
        grupo_entrada.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_entrada = QVBoxLayout()
        
        linha_entrada = QHBoxLayout()
        self.edit_entrada = QLineEdit()
        self.edit_entrada.setPlaceholderText("Selecione a pasta com os PDFs...")
        self.edit_entrada.setMinimumHeight(35)
        linha_entrada.addWidget(self.edit_entrada)
        
        btn_procurar = QPushButton("🔍 Procurar")
        btn_procurar.setMinimumHeight(35)
        btn_procurar.clicked.connect(self.procurar_entrada)
        linha_entrada.addWidget(btn_procurar)
        
        layout_entrada.addLayout(linha_entrada)
        grupo_entrada.setLayout(layout_entrada)
        layout.addWidget(grupo_entrada)
        
        # Diretório de saída
        grupo_saida = QGroupBox("💾 Diretório de Saída (opcional)")
        grupo_saida.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_saida = QVBoxLayout()
        
        linha_saida = QHBoxLayout()
        self.edit_saida = QLineEdit()
        self.edit_saida.setPlaceholderText("Deixe vazio para criar automaticamente...")
        self.edit_saida.setMinimumHeight(35)
        linha_saida.addWidget(self.edit_saida)
        
        btn_procurar_saida = QPushButton("🔍 Procurar")
        btn_procurar_saida.setMinimumHeight(35)
        btn_procurar_saida.clicked.connect(self.procurar_saida)
        linha_saida.addWidget(btn_procurar_saida)
        
        layout_saida.addLayout(linha_saida)
        grupo_saida.setLayout(layout_saida)
        layout.addWidget(grupo_saida)
        
        # Parâmetros
        grupo_params = QGroupBox("⚙️ Parâmetros de Agrupamento")
        grupo_params.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_params = QVBoxLayout()
        
        # ============= NOVO: Seleção de Modo =============
        linha_modo = QHBoxLayout()
        label_modo = QLabel("Modo de Operação:")
        label_modo.setStyleSheet("font-weight: bold; font-size: 13px;")
        linha_modo.addWidget(label_modo)
        
        self.radio_manual = QPushButton("⚙️ Manual")
        self.radio_manual.setCheckable(True)
        self.radio_manual.setChecked(True)
        self.radio_manual.setMinimumHeight(35)
        self.radio_manual.clicked.connect(self.alternar_modo_manual)
        
        self.radio_automatico = QPushButton("🤖 Automático")
        self.radio_automatico.setCheckable(True)
        self.radio_automatico.setMinimumHeight(35)
        self.radio_automatico.clicked.connect(self.alternar_modo_automatico)
        
        # Estilo dos botões de modo
        estilo_radio = """
            QPushButton {
                background-color: #e0e0e0;
                border: 2px solid #bbb;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
                border: 2px solid #1976D2;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:checked:hover {
                background-color: #1E88E5;
            }
        """
        self.radio_manual.setStyleSheet(estilo_radio)
        self.radio_automatico.setStyleSheet(estilo_radio)
        
        linha_modo.addWidget(self.radio_manual)
        linha_modo.addWidget(self.radio_automatico)
        layout_params.addLayout(linha_modo)
        
        # Linha separadora
        separador = QLabel("─" * 50)
        separador.setStyleSheet("color: #ccc;")
        layout_params.addWidget(separador)
        
        # ============= Container de Parâmetros Manuais =============
        self.widget_params_manuais = QWidget()
        layout_params_manuais = QVBoxLayout()
        self.widget_params_manuais.setLayout(layout_params_manuais)
        
        # Parâmetro k
        linha_k = QHBoxLayout()
        label_k = QLabel("Vizinhos próximos (k):")
        label_k.setMinimumWidth(180)
        label_k.setToolTip("Número de vizinhos mais próximos para conectar no grafo")
        linha_k.addWidget(label_k)
        self.spin_k = QSpinBox()
        self.spin_k.setRange(3, 30)
        self.spin_k.setValue(8)
        self.spin_k.setMinimumHeight(30)
        linha_k.addWidget(self.spin_k)
        linha_k.addStretch()
        layout_params_manuais.addLayout(linha_k)
        
        # Tamanho mínimo do grupo
        linha_min = QHBoxLayout()
        label_min = QLabel("Tamanho mínimo do grupo:")
        label_min.setMinimumWidth(180)
        label_min.setToolTip("Grupos menores serão movidos para 'Não Agrupados'")
        linha_min.addWidget(label_min)
        self.spin_min = QSpinBox()
        self.spin_min.setRange(2, 15)
        self.spin_min.setValue(3)
        self.spin_min.setMinimumHeight(30)
        linha_min.addWidget(self.spin_min)
        linha_min.addStretch()
        layout_params_manuais.addLayout(linha_min)
        
        layout_params.addWidget(self.widget_params_manuais)
        
        # ============= Mensagem Modo Automático =============
        self.label_modo_auto = QLabel(
            "🤖 <b>Modo Automático Ativado</b><br><br>"
            "Os parâmetros serão otimizados automaticamente<br>"
            "com base no seu conjunto de PDFs.<br><br>"
            "<i>Nenhuma configuração manual necessária!</i>"
        )
        self.label_modo_auto.setStyleSheet("""
            QLabel {
                background-color: #E3F2FD;
                border: 2px solid #2196F3;
                border-radius: 8px;
                padding: 20px;
                color: #1565C0;
            }
        """)
        self.label_modo_auto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_modo_auto.setWordWrap(True)
        self.label_modo_auto.setVisible(False)
        layout_params.addWidget(self.label_modo_auto)
        
        # Separador
        separador2 = QLabel("─" * 50)
        separador2.setStyleSheet("color: #ccc;")
        layout_params.addWidget(separador2)
        
        # Checkbox renomear
        self.check_renomear = QCheckBox("📝 Renomear com título/palavras-chave inteligentes")
        self.check_renomear.setChecked(True)  # ✨ Ativado por padrão
        self.check_renomear.setStyleSheet("padding: 10px;")
        self.check_renomear.setToolTip("Renomeia PDFs com títulos legíveis ou palavras-chave extraídas")
        layout_params.addWidget(self.check_renomear)
        
        grupo_params.setLayout(layout_params)
        layout.addWidget(grupo_params)
        
        # Parâmetro k
        linha_k = QHBoxLayout()
        label_k = QLabel("Vizinhos próximos (k):")
        label_k.setMinimumWidth(180)
        linha_k.addWidget(label_k)
        self.spin_k = QSpinBox()
        self.spin_k.setRange(3, 30)
        self.spin_k.setValue(8)
        self.spin_k.setMinimumHeight(30)
        linha_k.addWidget(self.spin_k)
        linha_k.addStretch()
        layout_params.addLayout(linha_k)
        
        # Tamanho mínimo do grupo
        linha_min = QHBoxLayout()
        label_min = QLabel("Tamanho mínimo do grupo:")
        label_min.setMinimumWidth(180)
        linha_min.addWidget(label_min)
        self.spin_min = QSpinBox()
        self.spin_min.setRange(2, 15)
        self.spin_min.setValue(3)
        self.spin_min.setMinimumHeight(30)
        linha_min.addWidget(self.spin_min)
        linha_min.addStretch()
        layout_params.addLayout(linha_min)
        
        # Checkbox renomear
        self.check_renomear = QCheckBox("📝 Renomear arquivos com título/palavras-chave inteligentes")
        self.check_renomear.setChecked(False)
        self.check_renomear.setStyleSheet("padding: 10px;")
        layout_params.addWidget(self.check_renomear)
        
        grupo_params.setLayout(layout_params)
        layout.addWidget(grupo_params)
        
        # Botão executar
        self.btn_executar = QPushButton("▶️ EXECUTAR AGRUPAMENTO")
        self.btn_executar.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_executar.clicked.connect(self.executar_agrupamento)
        layout.addWidget(self.btn_executar)
        
        # Botão cancelar
        self.btn_cancelar = QPushButton("⏹️ CANCELAR")
        self.btn_cancelar.setEnabled(False)
        self.btn_cancelar.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_cancelar.clicked.connect(self.cancelar_agrupamento)
        layout.addWidget(self.btn_cancelar)
        
        # Barra de progresso
        self.barra_progresso = QProgressBar()
        self.barra_progresso.setRange(0, 0)
        self.barra_progresso.setVisible(False)
        self.barra_progresso.setMinimumHeight(25)
        self.barra_progresso.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        layout.addWidget(self.barra_progresso)
        
        layout.addStretch()
        
        # Rodapé
        rodape = QLabel("v1.0 - Desenvolvido com ❤️")
        rodape.setStyleSheet("color: #999; font-size: 10px; padding: 10px;")
        rodape.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(rodape)
        
        return painel
    
    def criar_painel_direito(self) -> QWidget:
        """Cria painel de preview/log direito."""
        painel = QWidget()
        layout = QVBoxLayout()
        painel.setLayout(layout)
        
        # Abas
        abas = QTabWidget()
        abas.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            QTabBar::tab {
                padding: 10px 20px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Aba de preview
        aba_preview = QWidget()
        layout_preview = QVBoxLayout()
        
        self.label_preview = QLabel("🖼️ O gráfico de visualização aparecerá aqui após o agrupamento")
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 3px dashed #ccc;
                border-radius: 10px;
                padding: 50px;
                font-size: 14px;
                color: #666;
            }
        """)
        self.label_preview.setMinimumHeight(500)
        
        layout_preview.addWidget(self.label_preview)
        aba_preview.setLayout(layout_preview)
        
        # Aba de log
        aba_log = QWidget()
        layout_log = QVBoxLayout()
        
        self.texto_log = QTextEdit()
        self.texto_log.setReadOnly(True)
        self.texto_log.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 2px solid #333;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        layout_log.addWidget(self.texto_log)
        aba_log.setLayout(layout_log)
        
        # Aba de detalhes
        aba_detalhes = QWidget()
        layout_detalhes = QVBoxLayout()
        
        self.texto_detalhes = QTextEdit()
        self.texto_detalhes.setReadOnly(True)
        self.texto_detalhes.setStyleSheet("""
            QTextEdit {
                font-size: 12px;
                background-color: #fafafa;
                color: #111; 
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 15px;
            }
        """)
        self.texto_detalhes.setPlainText("📊 As estatísticas do agrupamento aparecerão aqui...")
        
        layout_detalhes.addWidget(self.texto_detalhes)
        aba_detalhes.setLayout(layout_detalhes)
        
        abas.addTab(aba_preview, "🖼️ Visualização")
        abas.addTab(aba_log, "📋 Logs")
        abas.addTab(aba_detalhes, "📊 Estatísticas")
        
        layout.addWidget(abas)
        
        return painel
    
    def configurar_logging(self):
        """Configura logging para GUI."""
        class ManipuladorQt(logging.Handler):
            def __init__(self, widget_texto):
                super().__init__()
                self.widget_texto = widget_texto
            
            def emit(self, registro):
                msg = self.format(registro)
                self.widget_texto.append(msg)
        
        manipulador = ManipuladorQt(self.texto_log)
        manipulador.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(manipulador)
        logging.getLogger().setLevel(logging.INFO)
    
    def alternar_modo_manual(self):
        """Alterna para modo manual."""
        self.radio_manual.setChecked(True)
        self.radio_automatico.setChecked(False)
        self.widget_params_manuais.setVisible(True)
        self.label_modo_auto.setVisible(False)
        logging.info("🔧 Modo MANUAL ativado")
    
    def alternar_modo_automatico(self):
        """Alterna para modo automático."""
        self.radio_automatico.setChecked(True)
        self.radio_manual.setChecked(False)
        self.widget_params_manuais.setVisible(False)
        self.label_modo_auto.setVisible(True)
        logging.info("🤖 Modo AUTOMÁTICO ativado")
    
    def procurar_entrada(self):
        """Procura diretório de entrada."""
        pasta = QFileDialog.getExistingDirectory(self, "Selecione o Diretório com os PDFs")
        if pasta:
            self.edit_entrada.setText(pasta)
    
    def procurar_saida(self):
        """Procura diretório de saída."""
        pasta = QFileDialog.getExistingDirectory(self, "Selecione o Diretório de Saída")
        if pasta:
            self.edit_saida.setText(pasta)
    
    def executar_agrupamento(self):
        """Inicia trabalho de agrupamento."""
        # Valida entrada
        dir_entrada = self.edit_entrada.text().strip()
        if not dir_entrada:
            QMessageBox.warning(self, "⚠️ Entrada Necessária", "Por favor, selecione um diretório de entrada com os PDFs")
            return
        
        if not Path(dir_entrada).exists():
            QMessageBox.warning(self, "⚠️ Entrada Inválida", "O diretório de entrada não existe")
            return
        
        # Verifica PDFs
        contagem_pdf = len(list(Path(dir_entrada).rglob("*.pdf")))
        if contagem_pdf == 0:
            QMessageBox.warning(self, "⚠️ Sem PDFs", "Nenhum arquivo PDF foi encontrado no diretório selecionado")
            return
        
        # Desabilita controles
        self.btn_executar.setEnabled(False)
        self.btn_cancelar.setEnabled(True)
        self.barra_progresso.setVisible(True)
        
        # Limpa preview e detalhes
        self.label_preview.clear()
        self.label_preview.setText("🔄 Processando... Aguarde...")
        self.texto_detalhes.clear()
        
        # Prepara parâmetros
                # Prepara parâmetros baseado no modo
        if self.radio_automatico.isChecked():
            # Modo automático: calcula parâmetros otimizados
            k_auto = max(5, min(15, contagem_pdf // 10))  # Entre 5 e 15
            grupo_min_auto = max(2, contagem_pdf // 30)  # Dinâmico baseado no total
            
            logging.info(f"🤖 Modo AUTOMÁTICO - Parâmetros otimizados:")
            logging.info(f"   k = {k_auto} (vizinhos)")
            logging.info(f"   grupo_min = {grupo_min_auto}")
            
            parametros = {
                "dir_entrada": dir_entrada,
                "dir_saida": self.edit_saida.text().strip() or None,
                "k": k_auto,
                "resolucao": 1.0,
                "grupo_min": grupo_min_auto,
                "renomear": self.check_renomear.isChecked()
            }
        else:
            # Modo manual: usa valores da interface
            logging.info(f"🔧 Modo MANUAL - Parâmetros do usuário:")
            logging.info(f"   k = {self.spin_k.value()}")
            logging.info(f"   grupo_min = {self.spin_min.value()}")
            
            parametros = {
                "dir_entrada": dir_entrada,
                "dir_saida": self.edit_saida.text().strip() or None,
                "k": self.spin_k.value(),
                "resolucao": 1.0,
                "grupo_min": self.spin_min.value(),
                "renomear": self.check_renomear.isChecked()
            }
        
        # Cria thread worker
        self.thread = QThread()
        self.trabalhador = Trabalhador(self.controlador, parametros)
        self.trabalhador.moveToThread(self.thread)
        
        # Conecta sinais
        self.thread.started.connect(self.trabalhador.executar)
        self.trabalhador.finalizado.connect(self.ao_trabalho_finalizado)
        self.trabalhador.erro.connect(self.ao_erro_trabalho)
        self.trabalhador.finalizado.connect(self.thread.quit)
        self.trabalhador.erro.connect(self.thread.quit)
        self.thread.finished.connect(self.ao_thread_finalizada)
        
        # Inicia thread
        self.thread.start()
        
        logging.info(f"🚀 Iniciando agrupamento com {contagem_pdf} PDFs...")
    
    def cancelar_agrupamento(self):
        """Cancela trabalho atual."""
        self.controlador.solicitar_cancelamento()
        logging.warning("⚠️ Cancelamento solicitado pelo usuário...")
    
    def ao_trabalho_finalizado(self, resultado: dict):
        """Trata conclusão do trabalho."""
        if "error" in resultado:
            self.ao_erro_trabalho(resultado["error"])
            return
        
        # Atualiza preview
        if "caminho_png" in resultado and Path(resultado["caminho_png"]).exists():
            pixmap = QPixmap(resultado["caminho_png"])
            escalado = pixmap.scaled(
                self.label_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.label_preview.setPixmap(escalado)
        
        # Atualiza detalhes
        resumo = resultado.get("resumo", {})
        detalhes = f"""
╔═══════════════════════════════════════════════════╗
║          ✅ AGRUPAMENTO CONCLUÍDO COM SUCESSO!          ║
╚═══════════════════════════════════════════════════╝

📁 Diretório de Saída:
   {resultado.get('raiz_saida', 'N/A')}

🖼️ Visualização do Grafo:
   {resultado.get('caminho_png', 'N/A')}

📊 ESTATÍSTICAS DO AGRUPAMENTO:

   🔢 Número de Grupos: {resumo.get('n_grupos', 'N/A')}
   
   📈 Modularidade: {resumo.get('modularidade', 'N/A')}
      (Quanto maior, melhor a divisão em comunidades)
   
   ⚖️ Balanceamento: {resumo.get('balanceamento', 'N/A')}
      (Quanto mais próximo de 1, mais equilibrados os grupos)

═════════════════════════════════════════════════════

💡 Dica: Abra o diretório de saída para ver seus PDFs
         organizados em grupos!
        """
        self.texto_detalhes.setPlainText(detalhes.strip())
        
        QMessageBox.information(
            self,
            "✅ Sucesso!",
            f"Agrupamento concluído com sucesso!\n\n"
            f"📁 {resumo.get('n_grupos', 0)} grupos foram criados\n\n"
            f"📂 Saída: {resultado.get('raiz_saida')}"
        )
    
    def ao_erro_trabalho(self, msg_erro: str):
        """Trata erro no trabalho."""
        logging.error(f"❌ Falha: {msg_erro}")
        self.label_preview.setText("❌ Erro ao processar")
        QMessageBox.critical(
            self,
            "❌ Erro",
            f"O agrupamento falhou:\n\n{msg_erro}\n\nVerifique os logs para mais detalhes."
        )
    
    def ao_thread_finalizada(self):
        """Trata limpeza da thread."""
        # Reabilita controles
        self.btn_executar.setEnabled(True)
        self.btn_cancelar.setEnabled(False)
        self.barra_progresso.setVisible(False)
        
        # Limpa thread
        if self.thread:
            self.thread.deleteLater()
            self.thread = None
        if self.trabalhador:
            self.trabalhador.deleteLater()
            self.trabalhador = None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ponto de entrada principal."""
    # Suprimir completamente warnings do sistema
    import warnings
    warnings.filterwarnings("ignore")
    
    with _FdStderrSilencer():
        app = QApplication(sys.argv)

    app.setStyle('Fusion')
    
    # Verifica dependências
    if not PDF_DISPONIVEL:
        resposta = QMessageBox.question(
            None,
            "⚠️ Dependência Opcional",
            "PyPDF2 não está instalado. A extração de metadados será limitada.\n\n"
            "Deseja continuar mesmo assim?\n\n"
            "Para instalar: pip install PyPDF2",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if resposta == QMessageBox.StandardButton.No:
            return
    
    janela = JanelaPrincipal()
    janela.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()