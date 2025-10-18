# ==============================================================================
# Arquivo de Configuração do latexmk
# ==============================================================================
# Este arquivo configura o latexmk para compilação automática do documento LaTeX
# Uso: latexmk -pdf main.tex
# ==============================================================================

# Compilador principal: pdflatex
$pdf_mode = 1;

# Comando pdflatex com opções
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Comando biber para bibliografia
$biber = 'biber %O %S';

# Extensões de arquivos auxiliares a serem limpos
$clean_ext = 'bbl nav snm vrb synctex.gz run.xml';

# Número máximo de repetições para resolver referências
$max_repeat = 5;

# Sempre gerar PDF
$postscript_mode = 0;
$dvi_mode = 0;

# Visualizador de PDF padrão (opcional - ajustar conforme seu sistema)
# Windows:
# $pdf_previewer = 'start';
# Linux:
# $pdf_previewer = 'evince';
# macOS:
# $pdf_previewer = 'open -a Preview';

# Modo de preview contínuo
$preview_continuous_mode = 1;

# FIM DO ARQUIVO
