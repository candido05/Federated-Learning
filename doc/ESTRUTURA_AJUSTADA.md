# Estrutura do TCC - Ajustada para Template Oficial CIn/UFPB

## Mudanças Realizadas

### 1. Arquivos Principais Criados/Modificados

- **tcc.cls**: Classe LaTeX oficial do CIn copiada do template
- **dados_tcc.tex**: Metadados do trabalho (autor, orientador, banca, etc.)
- **main.tex**: Documento principal seguindo padrão do template oficial

### 2. Dados Configurados

- **Autor**: Cândido Leandro de Queiroga Bisneto
- **Orientador**: Fernando Menezes Matos
- **Curso**: Ciência da Computação
- **Instituição**: Centro de Informática - UFPB
- **Título**: Otimização de Modelos de Aprendizado Federado com Redes Definidas por Software

### 3. Elementos Pré-Textuais (pre_textual/)

- a_dedicatoria.tex
- b_agradecimentos.tex  
- c_resumo.tex
- d_abstract.tex
- e_abreviaturas.tex

### 4. Capítulos (chapters/)

Todos os capítulos foram reescritos seguindo o estilo do template:
- Removidos avisos em laranja (\hl{})
- Mantida apenas a estrutura/esqueleto
- Tabelas essenciais preservadas
- Seções e subseções seguem nomenclatura do template

**Capítulos criados:**
1. 01_introducao.tex - INTRODUÇÃO
2. 02_revisao_literatura.tex - REVISÃO DE LITERATURA
3. 03_fundamentos_teoricos.tex - FUNDAMENTOS TEÓRICOS
4. 04_metodologia.tex - METODOLOGIA
5. 05_dataset.tex - DATASET E PREPARAÇÃO DOS DADOS
6. 06_config_experimental.tex - CONFIGURAÇÃO EXPERIMENTAL
7. 07_resultados.tex - RESULTADOS
8. 08_discussao.tex - DISCUSSÃO
9. 09_conclusao.tex - CONCLUSÕES E TRABALHOS FUTUROS

### 5. Referências e Anexos

- **references.tex**: Arquivo para referências em formato ABNT
- **anexos.tex**: Arquivo para anexos (código, documentação)

### 6. Formatação Seguindo Template Oficial

- Margens ABNT: 3cm (esq/sup), 2cm (dir/inf)
- Espaçamento: 1.5 linhas
- Parágrafo com recuo de 0.49in
- Seções iniciam em nova página (\section)
- Elementos pré-textuais sem numeração (\section*)
- Listas de figuras e tabelas automáticas
- Sumário automático

### 7. Próximos Passos

**Para preencher posteriormente:**

1. **dados_tcc.tex**:
   - \profb: [Nome do Professor B]
   - \profc: [Nome do Professor C]
   - \coordenador: [Nome do Coordenador]

2. **Elementos pré-textuais**:
   - Dedicatória (opcional)
   - Agradecimentos
   - Resumo em português (máx. 200 palavras)
   - Abstract em inglês (máx. 200 palavras)

3. **Capítulos**: Preencher conteúdo de cada seção

4. **Logo UFPB**: 
   - Obter logo oficial da UFPB/CI
   - Salvar como: doc/figures/logo_ufpb.png

5. **Referências**: 
   - Adicionar referências bibliográficas em references.tex
   - Formato ABNT

### 8. Compilação

```bash
cd doc
latexmk -pdf main.tex    # Compila com resolução de referências
make                     # Alternativa com Makefile
```

O documento já está compilando com sucesso (main.pdf gerado - 860KB).

### 9. Diferenças do Template Anterior

- **Removido**: Pacote abntex2, soul (\hl), todonotes
- **Adicionado**: Classe tcc.cls oficial, estrutura simplificada
- **Estilo**: Mais próximo do padrão CIn/UFPB
- **Avisos laranja**: Todos removidos
- **Conteúdo**: Apenas estrutura/esqueleto mantido

## Estrutura de Arquivos Atual

```
doc/
├── main.tex                    # Documento principal
├── tcc.cls                     # Classe LaTeX oficial
├── dados_tcc.tex               # Metadados do trabalho
├── references.tex              # Referências ABNT
├── anexos.tex                  # Anexos
├── pre_textual/
│   ├── a_dedicatoria.tex
│   ├── b_agradecimentos.tex
│   ├── c_resumo.tex
│   ├── d_abstract.tex
│   └── e_abreviaturas.tex
├── chapters/
│   ├── 01_introducao.tex
│   ├── 02_revisao_literatura.tex
│   ├── 03_fundamentos_teoricos.tex
│   ├── 04_metodologia.tex
│   ├── 05_dataset.tex
│   ├── 06_config_experimental.tex
│   ├── 07_resultados.tex
│   ├── 08_discussao.tex
│   └── 09_conclusao.tex
├── figures/
│   └── README_LOGO.txt         # Instruções para logo
├── tables/
├── Makefile
└── main.pdf                    # PDF compilado
```
