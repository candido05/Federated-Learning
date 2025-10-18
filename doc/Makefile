# ==============================================================================
# Makefile para Compilação do TCC
# ==============================================================================
# Uso:
#   make        - Compila o documento
#   make clean  - Remove arquivos auxiliares
#   make cleanall - Remove todos arquivos gerados (incluindo PDF)
#   make full   - Compilação completa (pdflatex + biber + pdflatex x2)
#   make view   - Abre o PDF
#   make watch  - Compilação contínua (recompila ao detectar mudanças)
# ==============================================================================

# Variáveis
MAIN = main
PDFLATEX = pdflatex -interaction=nonstopmode -synctex=1
BIBER = biber
VIEWER = start # Windows (ajustar para 'evince' no Linux ou 'open' no macOS)

# Arquivo principal
MAINTEX = $(MAIN).tex
MAINPDF = $(MAIN).pdf

# Extensões de arquivos auxiliares
AUX_EXT = aux log out toc lof lot bbl blg bcf run.xml synctex.gz \
          fdb_latexmk fls nav snm vrb

# ==============================================================================
# Targets
# ==============================================================================

# Target padrão: compilar com latexmk
.PHONY: all
all:
	@echo "Compilando $(MAINTEX) com latexmk..."
	latexmk -pdf $(MAINTEX)
	@echo "Compilação concluída! PDF: $(MAINPDF)"

# Compilação completa manual (sem latexmk)
.PHONY: full
full:
	@echo "Compilação completa (pdflatex + biber + pdflatex x2)..."
	$(PDFLATEX) $(MAINTEX)
	$(BIBER) $(MAIN)
	$(PDFLATEX) $(MAINTEX)
	$(PDFLATEX) $(MAINTEX)
	@echo "Compilação completa concluída!"

# Limpar arquivos auxiliares
.PHONY: clean
clean:
	@echo "Removendo arquivos auxiliares..."
	@for ext in $(AUX_EXT); do \
		rm -f *.$$ext; \
		rm -f chapters/*.$$ext; \
	done
	@rm -f $(MAIN).pdf
	@echo "Limpeza concluída!"

# Limpar tudo (incluindo PDF)
.PHONY: cleanall
cleanall: clean
	@echo "Removendo PDF..."
	@rm -f $(MAINPDF)
	@echo "Limpeza completa concluída!"

# Visualizar PDF
.PHONY: view
view:
	@echo "Abrindo $(MAINPDF)..."
	@$(VIEWER) $(MAINPDF) || open $(MAINPDF) || xdg-open $(MAINPDF)

# Compilação contínua (watch mode)
.PHONY: watch
watch:
	@echo "Modo watch ativado. Pressione Ctrl+C para parar."
	latexmk -pdf -pvc $(MAINTEX)

# Help
.PHONY: help
help:
	@echo "Makefile para compilação do TCC"
	@echo ""
	@echo "Targets disponíveis:"
	@echo "  make          - Compila o documento com latexmk"
	@echo "  make full     - Compilação completa manual"
	@echo "  make clean    - Remove arquivos auxiliares"
	@echo "  make cleanall - Remove todos arquivos gerados"
	@echo "  make view     - Abre o PDF gerado"
	@echo "  make watch    - Compilação contínua (recompila automaticamente)"
	@echo "  make help     - Mostra esta mensagem"

# FIM DO MAKEFILE
