import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io

# Tenta importar pymatgen para leitura nativa de arquivos .cif
HAS_PYMATGEN = True
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
except ImportError:
    HAS_PYMATGEN = False

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DiffraPy - XRD Analysis & Plotter",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 DiffraPy")
st.caption("Processador e Plotador de Difração de Raios X")
st.markdown("---")

# Paletas pré-definidas para artigos
PALETAS_PREDEFINIDAS = {
    "Nature / Scientific": ["#1B3B6F", "#2A9D8F", "#E76F51", "#264653", "#F4A261", "#A8201A", "#14213D", "#6B705C"],
    "Classic Academic": ["#000000", "#E74C3C", "#1F77B4", "#2CA02C", "#9467BD", "#D62728", "#17BECF", "#8C564B"],
    "Monochrome (Teses/P&B)": ["#000000", "#333333", "#555555", "#777777", "#999999", "#BBBBBB", "#DDDDDD", "#1A1A1A"],
    "Ocean Scale": ["#03045E", "#023E8A", "#0077B6", "#0096C7", "#00B4D8", "#48CAE4", "#90E0EF", "#ADE8F4"],
    "Flame Scale": ["#370617", "#6A040F", "#9D0208", "#D00000", "#DC2F02", "#E85D04", "#F48C06", "#FAA307"],
    "Viridis (Artigos)": ["#440154", "#46327E", "#365C8D", "#277F8E", "#1FA187", "#4AC16D", "#A0DA39", "#FDE725"],
    "Plasma": ["#0D0887", "#46039F", "#7201A8", "#9C179E", "#BD3786", "#D8576B", "#ED7953", "#FB9F3A"],
    "High Contrast": ["#000000", "#FF0000", "#0000FF", "#008000", "#FF00FF", "#00FFFF", "#800000", "#000080"]
}

# -----------------------------------------------------------------------------
# DIALOG MODAIS
# -----------------------------------------------------------------------------
@st.dialog("🎨 Paletas de Cores Pré-Definidas", width="large")
def modal_paletas(qtd_amostras):
    st.markdown("Escolha uma paleta de cores padrão de publicação para aplicar às amostras:")
    
    for nome_p, cores in PALETAS_PREDEFINIDAS.items():
        col_nome, col_amostra, col_btn = st.columns([2, 4, 2])
        with col_nome:
            st.markdown(f"**{nome_p}**")
        with col_amostra:
            html_cores = "".join([f'<span style="background-color:{c};padding:5px 12px;margin:1px;border-radius:3px;"></span>' for c in cores])
            st.markdown(html_cores, unsafe_allow_html=True)
        with col_btn:
            if st.button("Aplicar", key=f"paleta_{nome_p}", use_container_width=True):
                for idx in range(qtd_amostras):
                    st.session_state[f"cor_a_{idx}"] = cores[idx % len(cores)]
                st.rerun()

    st.markdown("---")
    if st.button("Fechar", use_container_width=True):
        st.rerun()


@st.dialog("🔤 Inserir Símbolos & Formatação", width="large")
def modal_simbolos(key_alvo):
    st.markdown("Clique ou copie a sintaxe de formatação LaTeX:")
    tab_format, tab_gregas, tab_formulas = st.tabs(["Sub/Sobescrito", "Letras Gregas", "Fórmulas DRX"])
    
    with tab_format:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.code(r"$_2$", language=None)
            st.code(r"$_3$", language=None)
        with col2:
            st.code(r"$_x$", language=None)
            st.code(r"$_{1-x}$", language=None)
        with col3:
            st.code(r"$^{2+}$", language=None)
            st.code(r"$^{3+}$", language=None)
        with col4:
            st.code(r"$^{\circ}$", language=None)
            st.code(r"$\AA$", language=None)

    with tab_gregas:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.code(r"$\theta$", language=None)
            st.code(r"2$\theta$", language=None)
        with col2:
            st.code(r"$\alpha$", language=None)
            st.code(r"$\beta$", language=None)
        with col3:
            st.code(r"$\gamma$", language=None)
            st.code(r"$\delta$", language=None)
        with col4:
            st.code(r"$\lambda$", language=None)

    with tab_formulas:
        col1, col2 = st.columns(2)
        with col1:
            st.code(r"TiO$_2$", language=None)
            st.code(r"IrO$_2$", language=None)
            st.code(r"Al$_2$O$_3$", language=None)
        with col2:
            st.code(r"Cu-K$\alpha$", language=None)
            st.code(r"Co-K$\alpha$", language=None)
            st.code(r"2$\theta$ (°)", language=None)

    st.markdown("---")
    if st.button("Fechar", use_container_width=True):
        st.rerun()

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE LEITURA
# -----------------------------------------------------------------------------
def ler_arquivo_drx(file):
    conteudo = file.getvalue().decode("utf-8", errors="ignore")
    linhas = conteudo.splitlines()
    
    linha_inicio = 0
    for idx, linha in enumerate(linhas):
        partes = linha.strip().replace(',', '.').split()
        if len(partes) >= 2:
            try:
                float(partes[0])
                float(partes[1])
                linha_inicio = idx
                break
            except ValueError:
                continue

    df = pd.read_csv(
        io.StringIO(conteudo),
        skiprows=linha_inicio,
        sep=r'\s+|,',
        engine='python',
        header=None,
        usecols=[0, 1],
        names=['2theta', 'Intensidade']
    )
    df['2theta'] = pd.to_numeric(df['2theta'].astype(str).str.replace(',', '.'), errors='coerce')
    df['Intensidade'] = pd.to_numeric(df['Intensidade'].astype(str).str.replace(',', '.'), errors='coerce')
    return df.dropna()


def ler_cif(file, anodo="CuKa"):
    if not HAS_PYMATGEN:
        st.error("Biblioteca 'pymatgen' não encontrada. Instale via terminal: pip install pymatgen")
        return None

    conteudo = file.getvalue().decode("utf-8", errors="ignore")
    estrutura = Structure.from_str(conteudo, fmt="cif")
    
    try:
        calculadora = XRDCalculator(wavelength=anodo)
    except TypeError:
        try:
            calculadora = XRDCalculator(radiation=anodo)
        except TypeError:
            calculadora = XRDCalculator()
            
    padrao = calculadora.get_pattern(estrutura)
    
    return pd.DataFrame({
        '2theta': padrao.x,
        'Intensidade': padrao.y
    })

# -----------------------------------------------------------------------------
# BARRA LATERAL (PAINEL DE CONTROLE)
# -----------------------------------------------------------------------------
st.sidebar.header("📁 1. Arquivos de Entrada")

arquivos_amostras = st.sidebar.file_uploader(
    "Difratogramas das Amostras (.txt / .csv / .dat / .xy)", 
    type=["txt", "csv", "dat", "xy"], 
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ 2. Radiação do Difratômetro")
anodo_selecionado = st.sidebar.selectbox(
    "Anodo (Comprimento de Onda)",
    options=["CuKa", "CoKa", "FeKa", "MoKa", "CrKa"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("📐 3. Eixos e Limites (2θ)")
usar_limites = st.sidebar.checkbox("Customizar intervalo de 2θ", value=True)
x_min, x_max = 20.0, 80.0
if usar_limites:
    col_x1, col_x2 = st.sidebar.columns(2)
    x_min = col_x1.number_input("2θ Mínimo", value=20.0, step=1.0)
    x_max = col_x2.number_input("2θ Máximo", value=80.0, step=1.0)

col_label_x, col_btn_x = st.sidebar.columns([4, 1])
with col_label_x:
    label_eixo_x = st.text_input("Rótulo Eixo X", value=r"2$\theta$ (°)", key="label_x_input")
with col_btn_x:
    st.write("")
    if st.button("🔤", key="btn_modal_x", help="Inserir símbolos"):
        modal_simbolos("label_x_input")

label_eixo_y = st.sidebar.text_input("Rótulo Eixo Y", value="Intensidade (a.u.)")

st.sidebar.markdown("---")
st.sidebar.header("🎨 4. Estilo, Fontes & Legenda")
with st.sidebar.expander("Configurar Tamanhos (cm), Legenda e Tipografia"):
    fig_width_cm = st.slider("Largura da Imagem (cm)", 8.0, 30.0, 16.0, 0.5)
    fig_height_base_cm = st.slider("Altura do Gráfico Principal (cm)", 5.0, 25.0, 10.0, 0.5)
    ficha_height_ratio = st.slider("Proporção da Altura das Fichas", 0.1, 0.5, 0.2, 0.05)
    
    st.markdown("**Controles da Legenda**")
    posicao_legenda = st.selectbox(
        "Posição da Legenda",
        options=[
            "Lado Direito (Fora)",
            "Superior Direito",
            "Superior Esquerdo",
            "Inferior Direito",
            "Inferior Esquerdo",
            "Superior Centro"
        ],
        index=0
    )
    exibir_moldura_legenda = st.checkbox("Exibir moldura (frame) na legenda", value=False)
    
    font_labels = st.slider("Tamanho Fonte dos Eixos", 8, 20, 12)
    font_ticks = st.slider("Tamanho Fonte dos Números (Ticks)", 6, 18, 10)
    font_legend = st.slider("Tamanho Fonte da Legenda", 6, 16, 9)
    
    offset = st.slider("Deslocamento Y (Offset)", 0.0, 2.0, 0.4, 0.05)
    largura_barra_ficha = st.slider("Espessura das Barras da Ficha", 0.05, 1.0, 0.1, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("💾 5. Exportação")
dpi = st.sidebar.select_slider("Resolução da Imagem (DPI)", options=[150, 300, 600], value=300)
formato_exportacao = st.sidebar.selectbox("Formato", ["tiff", "png", "pdf"])

# -----------------------------------------------------------------------------
# PROCESSAMENTO E PLOTAGEM
# -----------------------------------------------------------------------------
if arquivos_amostras:
    col_grafico, col_painel = st.columns([3, 1])

    with col_painel:
        st.subheader("Configurações das Amostras")
        
        if st.button("🎨 Escolher Paleta de Cores Pronta", use_container_width=True):
            modal_paletas(len(arquivos_amostras))

        paleta_nature = PALETAS_PREDEFINIDAS["Nature / Scientific"]
        
        config_amostras = []
        for idx, file_a in enumerate(arquivos_amostras):
            nome_padrao = file_a.name.rsplit('.', 1)[0]
            
            key_cor = f"cor_a_{idx}"
            if key_cor not in st.session_state:
                st.session_state[key_cor] = paleta_nature[idx % len(paleta_nature)]

            with st.expander(f"📌 {nome_padrao}", expanded=False):
                col_n_in, col_n_btn = st.columns([4, 1])
                key_nome_amostra = f"nome_a_{idx}"
                with col_n_in:
                    nome_custom = st.text_input("Legenda", value=nome_padrao, key=key_nome_amostra)
                with col_n_btn:
                    st.write("")
                    if st.button("🔤", key=f"btn_modal_a_{idx}", help="Inserir símbolos"):
                        modal_simbolos(key_nome_amostra)

                cor_custom = st.color_picker("Cor da linha", key=key_cor)
                largura_linha = st.slider("Espessura da linha", 0.5, 4.0, 1.2, 0.1, key=f"w_a_{idx}")
                ordem_empilhamento = st.number_input("Ordem (0 = topo)", 0, len(arquivos_amostras)-1, idx, 1, key=f"ordem_a_{idx}")
                
                config_amostras.append({
                    "file": file_a,
                    "nome": nome_custom,
                    "cor": cor_custom,
                    "largura": largura_linha,
                    "ordem": ordem_empilhamento
                })

        config_amostras = sorted(config_amostras, key=lambda x: x["ordem"])

        st.markdown("---")
        st.subheader("Fichas Catalográficas")
        num_fichas = st.number_input("Quantidade de Fichas", 0, 5, 1, 1)
        
        dados_fichas = []
        for n in range(int(num_fichas)):
            st.markdown(f"**Fase {n+1}**")
            file_f = st.file_uploader(f"Upload Ficha {n+1} (.cif / .txt)", type=["cif", "txt", "csv", "dat", "xy"], key=f"f_{n}")
            
            col_f_in, col_f_btn = st.columns([4, 1])
            key_nome_ficha = f"nome_f_{n}"
            with col_f_in:
                nome_fase = st.text_input(f"Nome da Fase {n+1}", value=f"Fase {n+1}", key=key_nome_ficha)
            with col_f_btn:
                st.write("")
                if st.button("🔤", key=f"btn_modal_f_{n}", help="Inserir símbolos"):
                    modal_simbolos(key_nome_ficha)

            cor_f = st.color_picker(f"Cor da Fase {n+1}", value="#E74C3C" if n==0 else "#3498DB", key=f"c_f_{n}")
            
            if file_f is not None:
                dados_fichas.append({
                    "file": file_f,
                    "nome": nome_fase,
                    "cor": cor_f
                })

    # Área Central: Gerador do Gráfico
    with col_grafico:
        num_fichas_validas = len(dados_fichas)
        total_subplots = 1 + num_fichas_validas

        fig_width_inches = fig_width_cm / 2.54
        fig_height_base_inches = fig_height_base_cm / 2.54

        altura_total_inches = fig_height_base_inches + (num_fichas_validas * fig_height_base_inches * ficha_height_ratio)
        fig = plt.figure(figsize=(fig_width_inches, altura_total_inches))
        
        height_ratios = [1.0] + [ficha_height_ratio] * num_fichas_validas
        gs = gridspec.GridSpec(total_subplots, 1, height_ratios=height_ratios, hspace=0.0)

        axes = []
        for s in range(total_subplots):
            if s == 0:
                axes.append(fig.add_subplot(gs[s]))
            else:
                axes.append(fig.add_subplot(gs[s], sharex=axes[0]))

        ax_main = axes[0]

        # 1. Plot das Amostras
        for i, item_a in enumerate(reversed(config_amostras)):
            try:
                df_a = ler_arquivo_drx(item_a["file"])
                y_norm = (df_a['Intensidade'] - df_a['Intensidade'].min()) / (df_a['Intensidade'].max() - df_a['Intensidade'].min() + 1e-9)
                y_plot = y_norm + (i * offset)
                
                ax_main.plot(
                    df_a['2theta'], 
                    y_plot, 
                    label=item_a["nome"], 
                    color=item_a["cor"], 
                    linewidth=item_a["largura"]
                )
            except Exception as e:
                st.error(f"Erro ao processar amostra {item_a['nome']}: {e}")

        ax_main.set_ylabel(label_eixo_y, fontsize=font_labels)
        ax_main.set_yticks([])
        ax_main.tick_params(axis='x', which='both', labelbottom=False, direction='in', labelsize=font_ticks)
        ax_main.tick_params(axis='y', direction='in')
        
        # Mapeamento de Posições da Legenda
        if posicao_legenda == "Lado Direito (Fora)":
            ax_main.legend(
                bbox_to_anchor=(1.02, 1), 
                loc='upper left', 
                frameon=exibir_moldura_legenda, 
                fontsize=font_legend
            )
        else:
            mapa_posicoes = {
                "Superior Direito": "upper right",
                "Superior Esquerdo": "upper left",
                "Inferior Direito": "lower right",
                "Inferior Esquerdo": "lower left",
                "Superior Centro": "upper center"
            }
            ax_main.legend(
                loc=mapa_posicoes[posicao_legenda], 
                frameon=exibir_moldura_legenda, 
                fontsize=font_legend
            )

        # 2. Plot das Fichas
        for idx_f, item_f in enumerate(dados_fichas):
            ax_ficha = axes[idx_f + 1]
            try:
                extensao = item_f["file"].name.rsplit('.', 1)[-1].lower()
                if extensao == "cif":
                    df_f = ler_cif(item_f["file"], anodo=anodo_selecionado)
                else:
                    df_f = ler_arquivo_drx(item_f["file"])

                if df_f is not None and not df_f.empty:
                    ax_ficha.bar(
                        df_f['2theta'], 
                        df_f['Intensidade'], 
                        width=largura_barra_ficha, 
                        color=item_f["cor"], 
                        edgecolor=item_f["cor"]
                    )
                
                ax_ficha.set_ylim(0, df_f['Intensidade'].max() * 1.15)
                ax_ficha.set_yticks([])
                
                ax_ficha.text(
                    0.98, 0.82, 
                    item_f["nome"], 
                    transform=ax_ficha.transAxes, 
                    fontsize=font_legend, 
                    verticalalignment='top', 
                    horizontalalignment='right',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
                )
            except Exception as e:
                st.error(f"Erro ao ler ficha {item_f['nome']}: {e}")

            if idx_f < num_fichas_validas - 1:
                ax_ficha.tick_params(axis='x', which='both', labelbottom=False, direction='in')
            else:
                ax_ficha.set_xlabel(label_eixo_x, fontsize=font_labels)
                ax_ficha.tick_params(axis='x', direction='in', labelsize=font_ticks)

        if usar_limites and x_min < x_max:
            ax_main.set_xlim(x_min, x_max)

        plt.subplots_adjust(hspace=0.0)

        st.pyplot(fig)

        # Exportação em alta resolução
        buf = io.BytesIO()
        fig.savefig(buf, format=formato_exportacao, dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label=f"⬇️ Baixar Difratograma ({formato_exportacao.upper()} - {dpi} DPI)",
            data=buf,
            file_name=f"diffrapy_plot.{formato_exportacao}",
            mime=f"image/{formato_exportacao}"
        )

else:
    st.info("👈 Para começar, faça o upload de uma ou mais amostras no painel lateral.")