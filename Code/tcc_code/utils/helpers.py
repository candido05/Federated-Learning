"""Funções auxiliares para experimentos de Aprendizado Federado.

Este módulo contém funções utilitárias gerais para manipulação de dados,
execução segura de simulações e obtenção de informações do sistema.
"""

import logging
import platform
import re
from typing import Any, Callable, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


def replace_keys(
    data: Dict[str, Any],
    match: str = "-",
    target: str = "_",
) -> Dict[str, Any]:
    """Substitui caracteres nas chaves de um dicionário.

    Útil para normalizar nomes de chaves (ex: converter hífens em underscores).

    Args:
        data: Dicionário com chaves a serem modificadas.
        match: String ou caractere a ser substituído (padrão: "-").
        target: String ou caractere de substituição (padrão: "_").

    Returns:
        Novo dicionário com chaves modificadas.

    Example:
        >>> replace_keys({"num-clients": 5, "log-dir": "logs"})
        {'num_clients': 5, 'log_dir': 'logs'}
    """
    new_dict = {}

    for key, value in data.items():
        # Substitui o caractere na chave
        new_key = key.replace(match, target)

        # Processa recursivamente se o valor for um dicionário
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value

    return new_dict


def safe_run_simulation(
    simulation_func: Callable[..., Any],
    *args: Any,
    fallback_value: Any = None,
    error_message: str = "Erro ao executar simulação",
    **kwargs: Any,
) -> Any:
    """Executa uma função de simulação com tratamento de erros e fallback.

    Args:
        simulation_func: Função de simulação a ser executada (ex: fl.simulation.start_simulation).
        *args: Argumentos posicionais para a função.
        fallback_value: Valor a retornar em caso de erro (padrão: None).
        error_message: Mensagem de erro a ser logada (padrão: "Erro ao executar simulação").
        **kwargs: Argumentos nomeados para a função.

    Returns:
        Resultado da função ou fallback_value em caso de erro.

    Example:
        >>> from flwr.simulation import start_simulation
        >>> result = safe_run_simulation(
        ...     start_simulation,
        ...     client_fn=my_client_fn,
        ...     num_clients=10,
        ...     config=fl.server.ServerConfig(num_rounds=5),
        ...     fallback_value={},
        ... )
    """
    try:
        logger.info("Iniciando simulação...")
        result = simulation_func(*args, **kwargs)
        logger.info("Simulação concluída com sucesso")
        return result

    except KeyboardInterrupt:
        logger.warning("Simulação interrompida pelo usuário")
        raise

    except Exception as e:
        logger.error(f"{error_message}: {e}", exc_info=True)
        logger.warning(f"Retornando valor de fallback: {fallback_value}")
        return fallback_value


def get_device_info() -> Dict[str, Any]:
    """Obtém informações sobre dispositivos de CPU/GPU disponíveis.

    Returns:
        Dicionário contendo informações sobre:
            - cpu: Informações da CPU (processador, núcleos, uso)
            - memory: Informações de memória RAM
            - gpu: Informações de GPU (se disponível)
            - platform: Informações do sistema operacional

    Example:
        >>> info = get_device_info()
        >>> print(f"CPU: {info['cpu']['processor']}")
        >>> print(f"Núcleos: {info['cpu']['physical_cores']}")
    """
    device_info: Dict[str, Any] = {}

    # Informações da CPU
    try:
        cpu_info = {
            "processor": platform.processor() or "Desconhecido",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq_current_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "cpu_freq_max_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
        }
        device_info["cpu"] = cpu_info
    except Exception as e:
        logger.warning(f"Erro ao obter informações da CPU: {e}")
        device_info["cpu"] = {"error": str(e)}

    # Informações de memória
    try:
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent,
        }
        device_info["memory"] = memory_info
    except Exception as e:
        logger.warning(f"Erro ao obter informações de memória: {e}")
        device_info["memory"] = {"error": str(e)}

    # Informações de GPU (tenta detectar CUDA/PyTorch)
    gpu_info = {"available": False, "device_count": 0, "devices": []}

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": ".".join(
                        map(str, torch.cuda.get_device_capability(i))
                    ),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        logger.debug("PyTorch não disponível - não é possível detectar GPUs CUDA")
    except Exception as e:
        logger.warning(f"Erro ao obter informações de GPU: {e}")

    device_info["gpu"] = gpu_info

    # Informações da plataforma
    try:
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        }
        device_info["platform"] = platform_info
    except Exception as e:
        logger.warning(f"Erro ao obter informações da plataforma: {e}")
        device_info["platform"] = {"error": str(e)}

    return device_info


def format_device_info(device_info: Dict[str, Any]) -> str:
    """Formata informações de dispositivo para exibição legível.

    Args:
        device_info: Dicionário retornado por get_device_info().

    Returns:
        String formatada com informações do dispositivo.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("INFORMAÇÕES DO SISTEMA")
    lines.append("=" * 70)

    # Plataforma
    if "platform" in device_info:
        platform_data = device_info["platform"]
        lines.append("\nPlataforma:")
        lines.append(f"  Sistema Operacional: {platform_data.get('system', 'N/A')}")
        lines.append(f"  Versão: {platform_data.get('release', 'N/A')}")
        lines.append(f"  Arquitetura: {platform_data.get('machine', 'N/A')}")
        lines.append(f"  Python: {platform_data.get('python_version', 'N/A')}")

    # CPU
    if "cpu" in device_info:
        cpu_data = device_info["cpu"]
        lines.append("\nCPU:")
        lines.append(f"  Processador: {cpu_data.get('processor', 'N/A')}")
        lines.append(f"  Núcleos Físicos: {cpu_data.get('physical_cores', 'N/A')}")
        lines.append(f"  Núcleos Lógicos: {cpu_data.get('logical_cores', 'N/A')}")
        if cpu_data.get("cpu_freq_current_mhz"):
            lines.append(f"  Frequência Atual: {cpu_data['cpu_freq_current_mhz']:.0f} MHz")
        lines.append(f"  Uso de CPU: {cpu_data.get('cpu_usage_percent', 'N/A')}%")

    # Memória
    if "memory" in device_info:
        mem_data = device_info["memory"]
        lines.append("\nMemória RAM:")
        lines.append(f"  Total: {mem_data.get('total_gb', 'N/A')} GB")
        lines.append(f"  Disponível: {mem_data.get('available_gb', 'N/A')} GB")
        lines.append(f"  Usado: {mem_data.get('used_gb', 'N/A')} GB ({mem_data.get('percent_used', 'N/A')}%)")

    # GPU
    if "gpu" in device_info:
        gpu_data = device_info["gpu"]
        lines.append("\nGPU:")
        if gpu_data.get("available"):
            lines.append(f"  GPUs Disponíveis: {gpu_data.get('device_count', 0)}")
            for device in gpu_data.get("devices", []):
                lines.append(f"    [{device['id']}] {device['name']}")
                lines.append(f"        Memória: {device['total_memory_gb']} GB")
                lines.append(f"        Compute Capability: {device['compute_capability']}")
        else:
            lines.append("  Nenhuma GPU disponível")

    lines.append("=" * 70)

    return "\n".join(lines)


def validate_config(config: Dict[str, Any], required_keys: list[str]) -> bool:
    """Valida se um dicionário de configuração contém todas as chaves necessárias.

    Args:
        config: Dicionário de configuração a validar.
        required_keys: Lista de chaves obrigatórias.

    Returns:
        True se todas as chaves estão presentes, False caso contrário.

    Example:
        >>> config = {"num_clients": 10, "num_rounds": 5}
        >>> validate_config(config, ["num_clients", "num_rounds"])
        True
        >>> validate_config(config, ["num_clients", "dataset"])
        False
    """
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        logger.error(f"Chaves obrigatórias ausentes na configuração: {missing_keys}")
        return False

    return True


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Remove ou substitui caracteres inválidos de um nome de arquivo.

    Args:
        filename: Nome do arquivo a ser sanitizado.
        replacement: Caractere de substituição para caracteres inválidos (padrão: "_").

    Returns:
        Nome de arquivo sanitizado.

    Example:
        >>> sanitize_filename("model:xgboost/strategy:fedavg")
        'model_xgboost_strategy_fedavg'
    """
    # Caracteres inválidos em nomes de arquivo (Windows/Linux)
    invalid_chars = r'[<>:"/\\|?*]'

    # Substitui caracteres inválidos
    sanitized = re.sub(invalid_chars, replacement, filename)

    # Remove espaços extras e caracteres de controle
    sanitized = re.sub(r'\s+', replacement, sanitized)
    sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)

    # Remove underscores duplicados
    sanitized = re.sub(f'{re.escape(replacement)}+', replacement, sanitized)

    # Remove underscores no início/fim
    sanitized = sanitized.strip(replacement)

    return sanitized or "unnamed"
