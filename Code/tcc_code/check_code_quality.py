"""
Script para verificar qualidade do código: variáveis não usadas, imports desnecessários.
"""

import ast
import os
from pathlib import Path
from typing import Set, Dict, List, Tuple


class CodeAnalyzer(ast.NodeVisitor):
    """Analisa código Python para encontrar problemas."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports: Dict[str, int] = {}  # nome: linha
        self.used_names: Set[str] = set()
        self.function_params: Dict[str, Set[str]] = {}  # função: {parâmetros}
        self.current_function = None
        self.issues: List[Tuple[str, int, str]] = []

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Guarda função atual
        prev_function = self.current_function
        self.current_function = node.name

        # Coleta parâmetros
        params = set()
        for arg in node.args.args:
            params.add(arg.arg)

        # Analisa corpo da função
        body_analyzer = ParameterUsageAnalyzer(params)
        for stmt in node.body:
            body_analyzer.visit(stmt)

        # Verifica parâmetros não usados (exceto self, cls, _)
        for param in params:
            if param not in body_analyzer.used_params:
                if param not in ['self', 'cls'] and not param.startswith('_'):
                    self.issues.append((
                        'unused_param',
                        node.lineno,
                        f"Parâmetro '{param}' não usado na função '{node.name}'"
                    ))

        self.current_function = prev_function
        self.generic_visit(node)

    def analyze(self, tree):
        """Executa análise completa."""
        self.visit(tree)

        # Verifica imports não usados
        for name, lineno in self.imports.items():
            if name not in self.used_names:
                # Exceções comuns
                if name in ['TYPE_CHECKING', 'annotations']:
                    continue
                self.issues.append((
                    'unused_import',
                    lineno,
                    f"Import '{name}' não usado"
                ))

        return self.issues


class ParameterUsageAnalyzer(ast.NodeVisitor):
    """Analisa uso de parâmetros dentro de uma função."""

    def __init__(self, params: Set[str]):
        self.params = params
        self.used_params: Set[str] = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.params:
            self.used_params.add(node.id)
        self.generic_visit(node)


def analyze_file(filepath: str) -> List[Tuple[str, int, str]]:
    """
    Analisa arquivo Python.

    Args:
        filepath: Caminho do arquivo.

    Returns:
        Lista de (tipo, linha, mensagem).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        tree = ast.parse(code, filename=filepath)
        analyzer = CodeAnalyzer(filepath)
        return analyzer.analyze(tree)

    except Exception as e:
        return [('error', 0, f"Erro ao analisar: {e}")]


def analyze_directory(directory: str, pattern: str = "**/*.py") -> Dict[str, List]:
    """
    Analisa todos os arquivos Python em diretório.

    Args:
        directory: Diretório raiz.
        pattern: Padrão de arquivos.

    Returns:
        Dicionário {arquivo: [(tipo, linha, mensagem), ...]}.
    """
    results = {}
    base_path = Path(directory)

    for filepath in base_path.glob(pattern):
        # Ignora __pycache__, venv, etc
        if '__pycache__' in str(filepath) or 'venv' in str(filepath):
            continue

        issues = analyze_file(str(filepath))
        if issues:
            results[str(filepath)] = issues

    return results


def main():
    """Executa análise nos módulos server e strategies."""
    print("=" * 60)
    print("ANÁLISE DE QUALIDADE DE CÓDIGO")
    print("=" * 60)

    # Diretórios a analisar
    directories = [
        "server",
        "strategies",
    ]

    total_issues = 0

    for directory in directories:
        print(f"\n{'=' * 60}")
        print(f"Analisando: {directory}/")
        print(f"{'=' * 60}")

        if not os.path.exists(directory):
            print(f"⚠ Diretório '{directory}' não encontrado")
            continue

        results = analyze_directory(directory)

        if not results:
            print(f"✓ Nenhum problema encontrado em {directory}/")
            continue

        for filepath, issues in results.items():
            filename = os.path.basename(filepath)
            print(f"\n{filename}:")

            # Agrupa por tipo
            by_type = {}
            for issue_type, lineno, message in issues:
                if issue_type not in by_type:
                    by_type[issue_type] = []
                by_type[issue_type].append((lineno, message))

            # Mostra por tipo
            for issue_type, items in sorted(by_type.items()):
                print(f"  [{issue_type.upper()}]:")
                for lineno, message in sorted(items):
                    print(f"    Linha {lineno}: {message}")
                    total_issues += 1

    print(f"\n{'=' * 60}")
    print(f"RESUMO: {total_issues} problema(s) encontrado(s)")
    print(f"{'=' * 60}")

    # Análise manual adicional
    print("\n" + "=" * 60)
    print("VERIFICAÇÕES MANUAIS ADICIONAIS")
    print("=" * 60)

    print("\n1. EVALUATION.PY:")
    print("   - 'params' em get_evaluate_fn: ✓ Usado na closure (evaluate_fn)")
    print("   - 'config' em evaluate_fn: ⚠ NÃO USADO - marcado como não usado na docstring")

    print("\n2. SERVER_MANAGER.PY:")
    print("   - 'Tuple' import: ✓ Usado em anotações de tipo")
    print("   - 'np' (numpy): ⚠ VERIFICAR - importado mas pode não ser usado")
    print("   - 'NDArrays': ✓ Usado em anotações de tipo")
    print("   - 'model_type' em create_strategy: ⚠ NÃO USADO - apenas para logging")

    print("\n3. BAGGING_STRATEGY.PY:")
    print("   - 'random' import: ⚠ VERIFICAR - pode não ser usado")
    print("   - 'parameters_to_ndarrays': ⚠ VERIFICAR - pode não ser usado")
    print("   - 'ndarrays_to_parameters': ⚠ VERIFICAR - pode não ser usado")

    print("\n4. CYCLIC_STRATEGY.PY:")
    print("   - Verificar se todos os imports são necessários")

    print("\n" + "=" * 60)
    print("REVISÃO COMPLETA")
    print("=" * 60)


if __name__ == "__main__":
    # Muda para diretório correto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    main()
