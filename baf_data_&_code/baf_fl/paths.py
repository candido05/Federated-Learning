"""
Caminhos e diretorios utilizados pelo pacote baf_fl.

Centraliza todos os caminhos de entrada/saida para facilitar
a manutencao e adaptacao a diferentes ambientes.
"""

import os

# Diretorio raiz do projeto (baf_data_&_code/)
_PACKAGE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.normpath(os.path.join(_PACKAGE_DIR, '..'))

# --- Entrada ---
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'Base.csv')

# --- Saida ---
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
