# Federated XGBoost — Rede Real (GNS3 / SDN / WSL)

Versão adaptada do exemplo `xgboost_comprehensive` do Flower para rodar em
**rede real via gRPC**, compatível com simuladores como GNS3 e ambientes SDN.

---

## Arquivos

```
fl_network/
├── server.py           # Servidor FL (roda em 1 nó)
├── client.py           # Cliente FL (roda em cada nó cliente)
├── test_local_wsl.sh   # Script para testar tudo localmente no WSL
└── README.md           # Este arquivo
```

---

## Requisitos

```bash
pip install flwr==1.8.0 flwr-datasets xgboost datasets numpy
```

> **Nota:** Todas as máquinas (servidor e clientes) devem ter o mesmo ambiente instalado.

---

## 1. Teste Local no WSL (antes do GNS3)

O script `test_local_wsl.sh` sobe o servidor e 3 clientes **em processos separados**
na mesma máquina, simulando a comunicação de rede via `127.0.0.1`.

```bash
chmod +x test_local_wsl.sh
./test_local_wsl.sh
```

Logs individuais ficam em `./logs/`:
```
logs/server.log
logs/client_0.log
logs/client_1.log
logs/client_2.log
```

Para monitorar em tempo real:
```bash
tail -f logs/server.log
tail -f logs/client_0.log
```

---

## 2. Deploy no GNS3 / Rede Real

### Topologia sugerida

```
[ Cliente 0 ]──┐
               ├──[Switch/SDN]──[ Servidor FL ]
[ Cliente 1 ]──┤
               │
[ Cliente 2 ]──┘
```

### No nó SERVIDOR (ex: IP 192.168.1.10)

```bash
python server.py \
    --host 0.0.0.0 \
    --port 8080 \
    --rounds 20 \
    --min-clients 3
```

Parâmetros disponíveis:
| Argumento | Padrão | Descrição |
|---|---|---|
| `--host` | `0.0.0.0` | IP para escutar (0.0.0.0 = todas as interfaces) |
| `--port` | `8080` | Porta gRPC |
| `--rounds` | `20` | Número de rounds federados |
| `--min-clients` | `3` | Mínimo de clientes para iniciar |
| `--centralised-eval` | off | Avaliação centralizada no servidor |

### Em cada nó CLIENTE

```bash
# Cliente 0
python client.py \
    --server-ip 192.168.1.10 \
    --port 8080 \
    --partition-id 0 \
    --num-partitions 3 \
    --local-epochs 20

# Cliente 1
python client.py --server-ip 192.168.1.10 --port 8080 \
    --partition-id 1 --num-partitions 3 --local-epochs 20

# Cliente 2
python client.py --server-ip 192.168.1.10 --port 8080 \
    --partition-id 2 --num-partitions 3 --local-epochs 20
```

Parâmetros disponíveis:
| Argumento | Padrão | Descrição |
|---|---|---|
| `--server-ip` | — | **Obrigatório.** IP do servidor |
| `--port` | `8080` | Porta gRPC do servidor |
| `--partition-id` | — | **Obrigatório.** ID único do cliente (0, 1, 2, ...) |
| `--num-partitions` | `3` | Total de clientes |
| `--local-epochs` | `20` | Épocas locais (árvores XGBoost por round) |
| `--test-fraction` | `0.2` | Fração dos dados para validação local |
| `--seed` | `42` | Semente aleatória |

---

## 3. Como funciona

### Fluxo por round:
```
Servidor
  │
  ├─► envia modelo global → Cliente 0
  ├─► envia modelo global → Cliente 1
  └─► envia modelo global → Cliente 2
         │
         │  (treino local: 20 épocas cada)
         │
  ◄──────┴─── recebe modelos atualizados
  │
  ├─► agrega árvores (bagging)
  ├─► avalia (distribuído ou centralizado)
  └─► próximo round...
```

### Agregação (Bagging):
- Cada cliente treina **20 árvores novas** localmente
- O servidor **acumula** as árvores de todos os clientes no modelo global
- Resultado: a cada round, o modelo global cresce em `N_clientes × local_epochs` árvores

---

## 4. Configurar porta no firewall (se necessário)

```bash
# Ubuntu/WSL — liberar porta 8080
sudo ufw allow 8080/tcp

# Verificar se a porta está aberta
ss -tlnp | grep 8080
```

---

## 5. Verificar conectividade (antes de rodar)

Do cliente, teste a conexão com o servidor:
```bash
# Ping
ping 192.168.1.10

# Teste de porta TCP (requer netcat)
nc -zv 192.168.1.10 8080

# Alternativa com Python
python3 -c "
import socket
s = socket.socket()
s.settimeout(3)
result = s.connect_ex(('192.168.1.10', 8080))
print('Porta ABERTA' if result == 0 else f'Porta FECHADA (código: {result})')
s.close()
"
```

---

## 6. Saída esperada

### Servidor (a cada round):
```
[SERVER] Round 1: agregando 3 cliente(s)...
[SERVER] Round 1 | AUC distribuído (média ponderada): 0.7823
[SERVER] Round 2: agregando 3 cliente(s)...
...
[SERVER] Modelo salvo em: final_model.json
```

### Cliente (a cada round):
```
[CLIENT 0] [Round 1] Iniciando treino local — 20 épocas locais
[CLIENT 0] [Round 1] Treino local concluído. Árvores: 20
[CLIENT 0] [Avaliação] AUC local: 0.7891 (X exemplos)
```

---

## Diferenças da versão original (Flower SDK interno)

| Aspecto | Original (flwr SDK) | Esta versão |
|---|---|---|
| Comunicação | Simulação interna | gRPC via socket real |
| Deploy | `flwr run` | `python server.py` + `python client.py` |
| Configuração | `pyproject.toml` | Argumentos CLI |
| Topologia | Simulada | Compatível com GNS3/SDN/VMs |
| Escalonamento | Automático | Manual por `--partition-id` |