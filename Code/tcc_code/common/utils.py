"""
Funções utilitárias para Federated Learning
"""


def replace_keys(input_dict, match="-", target="_"):
    """Substitui caracteres em chaves de dicionário recursivamente"""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
