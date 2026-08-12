import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import json
from scipy.signal import find_peaks

# Tenta importar pymatgen para leitura nativa de arquivos .cif
HAS_PYMATGEN = True
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
except ImportError:
    HAS_PYMATGEN = False

# Comprimentos de onda dos anodos em Angstroms (A)
WAVELENGTHS = {
    "CuKa": 1.5406,
    "CoKa": 1.7890,
    "FeKa": 1.9360,
    "MoKa": 0.7107,
    "CrKa": 2.2897
}

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DiffraPy - XRD Analysis & Plotter",
    layout="wide"
)

st.title("DiffraPy")
st.caption("Processador, Plotador e Analisador Estrutural de Difração de Raios X - Padrão de Publicação")
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
@st.dialog("Paletas de Cores Pré-Definidas", width="large")
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


@st.dialog("Inserir Símbolos & Formatação", width="large")
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
# FUNÇÕES AUXILIARES DE LEITURA E CÁLCULO
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
    return df.dropna().sort_values('2theta')


def ler_cif(file, anodo="CuKa"):
    if not HAS_PYMATGEN:
        st.error("Biblioteca 'pymatgen' não encontrada.")
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
    
    hkl_labels = []
    for hkls in padrao.hkls:
        if hkls and len(hkls) > 0:
            hkl_tuple = hkls[0]['hkl']
            hkl_str = f"({hkl_tuple[0]}{hkl_tuple[1]}{hkl_tuple[2]})"
        else:
            hkl_str = ""
        hkl_labels.append(hkl_str)
    
    return pd.DataFrame({
        '2theta': padrao.x,
        'Intensidade': padrao.y,
        'hkl': hkl_labels
    })


def auto_detectar_picos_df(df, min_prominence_ratio=0.05, distance_pts=15):
    if df.empty:
        return []
    
    intensidades = df['Intensidade'].values
    theta_vals = df['2theta'].values
    
    range_i = intensidades.max() - intensidades.min()
    prominence_val = range_i * min_prominence_ratio
    
    peaks_indices, _ = find_peaks(intensidades, prominence=prominence_val, distance=distance_pts)
    
    picos_encontrados = np.round(theta_vals[peaks_indices], 2)
    return picos_encontrados.tolist()


def calcular_fwhm_e_scherrer_por_faixa(df, x_min_range, x_max_range, k_factor, wavelength):
    df_sub = df[(df['2theta'] >= x_min_range) & (df['2theta'] <= x_max_range)].copy()
    if df_sub.empty or len(df_sub) < 3:
        return None

    idx_peak = df_sub['Intensidade'].idxmax()
    x_peak = df_sub.loc[idx_peak, '2theta']
    y_peak = df_sub.loc[idx_peak, 'Intensidade']

    y_min = df_sub['Intensidade'].min()
    y_half = y_min + (y_peak - y_min) / 2.0

    left_sub = df_sub[df_sub['2theta'] <= x_peak]
    right_sub = df_sub[df_sub['2theta'] >= x_peak]

    if left_sub.empty or right_sub.empty:
        return None

    x_left = np.interp(y_half, left_sub['Intensidade'], left_sub['2theta'])
    x_right = np.interp(y_half, right_sub['Intensidade'][::-1], right_sub['2theta'][::-1])

    fwhm_deg = abs(x_right - x_left)
    fwhm_rad = np.radians(fwhm_deg)

    theta_rad = np.radians(x_peak / 2.0)
    
    d_angstrom = (k_factor * wavelength) / (fwhm_rad * np.cos(theta_rad))
    d_nm = d_angstrom / 10.0

    d_spacing = wavelength / (2.0 * np.sin(theta_rad))

    y_wh = fwhm_rad * np.cos(theta_rad)
    x_wh = 4.0 * np.sin(theta_rad)

    return {
        "df_sub": df_sub,
        "2theta_peak": x_peak,
        "intensity_peak": y_peak,
        "fwhm_deg": fwhm_deg,
        "fwhm_rad": fwhm_rad,
        "x_left": x_left,
        "x_right": x_right,
        "y_half": y_half,
        "y_min": y_min,
        "cristalito_nm": d_nm,
        "d_spacing": d_spacing,
        "x_wh": x_wh,
        "y_wh": y_wh
    }


def buscar_pico_ref_cif(df_cif, target_2theta):
    if df_cif is None or df_cif.empty:
        return None
    
    idx_min = (df_cif['2theta'] - target_2theta).abs().idxmin()
    row_ref = df_cif.loc[idx_min]
    
    return {
        "2theta_cif": row_ref['2theta'],
        "intensity_cif": row_ref['Intensidade'],
        "hkl": row_ref['hkl'] if 'hkl' in row_ref else ""
    }

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
st.sidebar.header("💾 Gerenciador de Projetos")
with st.sidebar.expander("Salvar / Carregar Projeto (.json)"):
    # Botão para salvar projeto
    if st.button("💾 Exportar Estado do Projeto", use_container_width=True):
        estado_exportar = {}
        for key, val in st.session_state.items():
            # Filtra objetos não serializáveis em JSON
            if not callable(val) and not hasattr(val, "read"):
                estado_exportar[key] = val
        
        json_data = json.dumps(estado_exportar, indent=2)
        st.download_button(
            label="⬇️ Baixar Configurações (.json)",
            data=json_data,
            file_name="diffrapy_project.json",
            mime="application/json",
            use_container_width=True
        )

    # File uploader para carregar projeto
    proj_uploaded = st.file_uploader("Carregar Projeto Salvo (.json)", type=["json"], key="uploader_projeto_json")
    if proj_uploaded is not None:
        try:
            dados_proj = json.load(proj_uploaded)
            for k_p, v_p in dados_proj.items():
                st.session_state[k_p] = v_p
            st.success("Projeto carregado com sucesso!")
        except Exception as e_p:
            st.error(f"Erro ao carregar projeto: {e_p}")

st.sidebar.markdown("---")
st.sidebar.header("⚡ 2. Radiação do Difratômetro")
anodo_selecionado = st.sidebar.selectbox(
    "Anodo (Comprimento de Onda)",
    options=["CuKa", "CoKa", "FeKa", "MoKa", "CrKa"],
    index=0
)
lambda_onda = WAVELENGTHS[anodo_selecionado]

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
    ficha_height_ratio = st.slider("Proporção da Altura das Fichas", 0.1, 0.6, 0.25, 0.05)
    
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
st.sidebar.header("💾 5. Exportação da Imagem")
dpi = st.sidebar.select_slider("Resolução da Imagem (DPI)", options=[150, 300, 600], value=300)
formato_exportacao = st.sidebar.selectbox("Formato", ["tiff", "png", "pdf"])

# -----------------------------------------------------------------------------
# PROCESSAMENTO E ABAS DA APLICAÇÃO
# -----------------------------------------------------------------------------
if arquivos_amostras:
    tab_plotter, tab_pico_unico, tab_multi_picos = st.tabs([
        "Plotter de Difratogramas", 
        "Análise de Pico Individual", 
        "Múltiplos Picos & Williamson-Hall"
    ])

    # -------------------------------------------------------------------------
    # ABA 1: PLOTTER
    # -------------------------------------------------------------------------
    with tab_plotter:
        col_grafico, col_painel = st.columns([3, 1])

        with col_painel:
            st.subheader("Configurações das Amostras")
            
            if st.button("Escolher Paleta de Cores Pronta", use_container_width=True):
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
                        "ordem": ordem_empilhamento,
                        "idx_orig": idx
                    })

            config_amostras = sorted(config_amostras, key=lambda x: x["ordem"])

            st.markdown("---")
            st.subheader("Fichas Catalográficas")
            num_fichas = st.number_input("Quantidade de Fichas", 0, 5, 0, 1)
            
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
                
                exibir_hkl = st.checkbox("Exibir Índices (hkl) nos Picos", value=False, key=f"chk_hkl_{n}")
                corte_hkl = 10.0
                tamanho_fonte_hkl = 8
                margem_topo_hkl = 1.6
                
                if exibir_hkl:
                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        corte_hkl = st.number_input("I Mín (%)", value=10.0, step=5.0, key=f"corte_hkl_{n}")
                    with col_c2:
                        tamanho_fonte_hkl = st.number_input("Tam hkl", value=8, step=1, key=f"font_hkl_{n}")
                    with col_c3:
                        margem_topo_hkl = st.number_input("Folga Y", value=1.6, step=0.1, min_value=1.1, max_value=3.0, key=f"margem_hkl_{n}")

                if file_f is not None:
                    dados_fichas.append({
                        "file": file_f,
                        "nome": nome_fase,
                        "cor": cor_f,
                        "exibir_hkl": exibir_hkl,
                        "corte_hkl": corte_hkl,
                        "font_hkl": tamanho_fonte_hkl,
                        "margem_topo": margem_topo_hkl
                    })

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

            if num_fichas_validas > 0:
                ax_main.tick_params(axis='x', which='both', labelbottom=False, direction='in', labelsize=font_ticks)
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

                            if item_f["exibir_hkl"] and "hkl" in df_f.columns:
                                i_max = df_f['Intensidade'].max()
                                for _, row in df_f.iterrows():
                                    if (row['Intensidade'] / i_max * 100) >= item_f["corte_hkl"] and row['hkl']:
                                        if not usar_limites or (x_min <= row['2theta'] <= x_max):
                                            ax_ficha.text(
                                                row['2theta'], 
                                                row['Intensidade'] + (i_max * 0.04), 
                                                row['hkl'], 
                                                ha='center', 
                                                va='bottom', 
                                                fontsize=item_f["font_hkl"], 
                                                rotation=90, 
                                                color=item_f["cor"]
                                            )

                        fator_topo = item_f["margem_topo"] if item_f["exibir_hkl"] else 1.15
                        ax_ficha.set_ylim(0, df_f['Intensidade'].max() * fator_topo)
                        ax_ficha.set_yticks([])
                        
                        ax_ficha.text(
                            0.98, 0.88, 
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
            else:
                ax_main.set_xlabel(label_eixo_x, fontsize=font_labels)
                ax_main.tick_params(axis='x', which='both', labelbottom=True, direction='in', labelsize=font_ticks)

            if usar_limites and x_min < x_max:
                ax_main.set_xlim(x_min, x_max)

            plt.subplots_adjust(hspace=0.0)

            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format=formato_exportacao, dpi=dpi, bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label=f"Baixar Difratograma ({formato_exportacao.upper()} - {dpi} DPI)",
                data=buf,
                file_name=f"diffrapy_plot.{formato_exportacao}",
                mime=f"image/{formato_exportacao}"
            )

    # -------------------------------------------------------------------------
    # ABA 2: ANÁLISE DE PICO INDIVIDUAL
    # -------------------------------------------------------------------------
    with tab_pico_unico:
        st.header("Análise de Pico Individual")
        st.markdown("Selecione a amostra e a referência abaixo. Posicione o slider para mover a linha guia do gráfico em tempo real.")

        fichas_cif_disponiveis = [f for f in dados_fichas if f["file"].name.rsplit('.', 1)[-1].lower() == "cif"]

        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            nomes_amostras = [f.name.rsplit('.', 1)[0] for f in arquivos_amostras]
            amostra_alvo_idx = st.selectbox("Amostra Experimental", options=range(len(nomes_amostras)), format_func=lambda x: nomes_amostras[x], key="pico_amostra_slider")
            df_alvo_p = ler_arquivo_drx(arquivos_amostras[amostra_alvo_idx])

        with r1_c2:
            df_cif_ref_p = None
            if fichas_cif_disponiveis:
                ficha_p_sel_idx = st.selectbox(
                    "Ficha CIF de Referência", 
                    options=range(len(fichas_cif_disponiveis)), 
                    format_func=lambda x: fichas_cif_disponiveis[x]["nome"],
                    key="pico_cif_sel_slider"
                )
                file_cif_p_obj = fichas_cif_disponiveis[ficha_p_sel_idx]["file"]
                df_cif_ref_p = ler_cif(file_cif_p_obj, anodo=anodo_selecionado)
            else:
                st.info("💡 Para calcular o deslocamento exato vs CIF, faça o upload de ao menos uma ficha `.cif` na Aba 1.")

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            k_scherrer = st.number_input("Fator de Forma (K)", value=0.94, step=0.01, key="pico_k_slider")
        with r2_c2:
            janela_largura = st.number_input("Janela de Busca (±°)", value=1.5, step=0.5, min_value=0.2, key="janela_busca_simples")

        st.markdown("---")

        theta_min_data = float(df_alvo_p['2theta'].min())
        theta_max_data = float(df_alvo_p['2theta'].max())

        pico_centro_pos = st.slider(
            "🎯 Posicione o Seletor no Pico Desejado (2θ):",
            min_value=theta_min_data,
            max_value=theta_max_data,
            value=float(np.round((theta_min_data + theta_max_data) / 2.0, 1)),
            step=0.1,
            key="slider_simples_pico"
        )

        sel_min = pico_centro_pos - janela_largura
        sel_max = pico_centro_pos + janela_largura

        fig_overview, ax_ov = plt.subplots(figsize=(14, 2.5))
        fig_overview.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.25)

        ax_ov.plot(df_alvo_p['2theta'], df_alvo_p['Intensidade'], color='#1B3B6F', linewidth=1.2)
        ax_ov.axvline(x=pico_centro_pos, color='red', linestyle='--', linewidth=1.8, label=f"Posição Guia: {pico_centro_pos:.1f}°")
        ax_ov.axvspan(sel_min, sel_max, color='red', alpha=0.15, label="Janela de Busca")

        ax_ov.set_yticks([])
        ax_ov.set_ylabel("")

        ax_ov.set_xlabel("2θ (°)", fontsize=10)
        ax_ov.set_xlim(theta_min_data, theta_max_data)
        ax_ov.legend(loc="upper right", fontsize=8)
        ax_ov.grid(True, linestyle="--", alpha=0.3)
        
        st.pyplot(fig_overview, use_container_width=True)

        st.markdown("---")

        res_pico = calcular_fwhm_e_scherrer_por_faixa(df_alvo_p, sel_min, sel_max, k_scherrer, lambda_onda)

        col_zoom, col_metrica = st.columns([1, 1])

        with col_zoom:
            st.subheader("🔎 Zoom do Pico Isolado")
            if res_pico:
                res_cif = buscar_pico_ref_cif(df_cif_ref_p, res_pico['2theta_peak']) if df_cif_ref_p is not None else None
                ref_2theta_val = res_cif["2theta_cif"] if res_cif else res_pico['2theta_peak']
                hkl_label = res_cif["hkl"] if res_cif else ""

                fig_zoom, ax_zoom = plt.subplots(figsize=(6, 4))
                df_sub = res_pico["df_sub"]
                
                ax_zoom.plot(df_sub['2theta'], df_sub['Intensidade'], color='#1F77B4', linewidth=2.0, label="Perfil Medido")
                ax_zoom.axvline(x=res_pico['2theta_peak'], color='red', linestyle='--', linewidth=1.2, label=f"Topo ({res_pico['2theta_peak']:.3f}°)")
                ax_zoom.hlines(y=res_pico['y_half'], xmin=res_pico['x_left'], xmax=res_pico['x_right'], color='darkred', linewidth=2.0, label=f"FWHM ({res_pico['fwhm_deg']:.3f}°)")
                
                if res_cif:
                    ax_zoom.axvline(x=ref_2theta_val, color='green', linestyle=':', linewidth=1.5, label=f"CIF {hkl_label} ({ref_2theta_val:.3f}°)")

                ax_zoom.set_xlabel("2θ (°)", fontsize=10)
                ax_zoom.set_ylabel("Intensidade (a.u.)", fontsize=10)
                ax_zoom.legend(loc="upper right", fontsize=8)
                ax_zoom.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig_zoom, use_container_width=True)
            else:
                st.warning("Nenhum pico claro foi encontrado na janela selecionada pelo slider.")

        with col_metrica:
            st.subheader("📊 Métricas Calculadas")
            if res_pico:
                delta_2theta = res_pico["2theta_peak"] - ref_2theta_val
                d_cif = lambda_onda / (2.0 * np.sin(np.radians(ref_2theta_val / 2.0))) if ref_2theta_val > 0 else 0
                delta_d_perc = ((res_pico["d_spacing"] - d_cif) / d_cif) * 100.0 if d_cif > 0 else 0.0

                m1, m2 = st.columns(2)
                m1.metric("Pico Medido (2θ)", f"{res_pico['2theta_peak']:.3f}°")
                m2.metric("Pico Teórico (CIF)", f"{ref_2theta_val:.3f}°", f"Plano {hkl_label}" if hkl_label else "")

                m3, m4 = st.columns(2)
                m3.metric("FWHM (β)", f"{res_pico['fwhm_deg']:.3f}°", f"{res_pico['fwhm_rad']:.5f} rad")
                m4.metric("Cristalito (D)", f"{res_pico['cristalito_nm']:.2f} nm")

                st.markdown("---")
                col_d1, col_d2 = st.columns(2)
                col_d1.metric("Deslocamento (Δ2θ)", f"{delta_2theta:+.3f}°")
                col_d2.metric("Variação Interplanar (Δd/d₀)", f"{delta_d_perc:+.2f}%", f"d = {res_pico['d_spacing']:.4f} Å vs d₀ = {d_cif:.4f} Å")

    # -------------------------------------------------------------------------
    # ABA 3: MÚLTIPLOS PICOS, MÉDIA & WILLIAMSON-HALL (COM EXPORTAÇÃO DA TABELA)
    # -------------------------------------------------------------------------
    with tab_multi_picos:
        st.header("Análise Avançada de Picos: Média de Scherrer & Williamson-Hall")
        st.markdown("Adicione múltiplos picos da sua amostra para calcular o **tamanho médio de cristalito** e obter o gráfico de **Williamson-Hall** para separação de microdeformação de rede ($\epsilon$).")

        fichas_cif_disponiveis = [f for f in dados_fichas if f["file"].name.rsplit('.', 1)[-1].lower() == "cif"]

        col_cfg_wh, col_res_wh = st.columns([1, 2])

        with col_cfg_wh:
            st.subheader("Configurações da Análise")
            
            nomes_amostras = [f.name.rsplit('.', 1)[0] for f in arquivos_amostras]
            amostra_wh_idx = st.selectbox("Amostra Experimental", options=range(len(nomes_amostras)), format_func=lambda x: nomes_amostras[x], key="wh_amostra")
            
            k_wh = st.number_input("Fator de Forma (K)", value=0.94, step=0.01, key="wh_k")
            pico_window_wh = st.number_input("Janela de Busca (±°)", value=1.5, step=0.5, min_value=0.2, key="wh_window")
            
            st.markdown("---")
            st.markdown("**🔍 Detecção Automática de Picos**")
            sensibilidade = st.slider("Sensibilidade (Proeminência Mínima %)", 1.0, 30.0, 5.0, 1.0, help="Diminua o valor para encontrar picos menores ou aumente para capturar apenas os picos mais fortes.")
            
            df_alvo_wh = ler_arquivo_drx(arquivos_amostras[amostra_wh_idx])
            
            if "picos_list_state" not in st.session_state:
                st.session_state["picos_list_state"] = [25.3, 37.8, 48.0, 53.9, 55.0]

            if st.button("🔍 Detectar Picos Automaticamente", use_container_width=True):
                picos_auto = auto_detectar_picos_df(df_alvo_wh, min_prominence_ratio=sensibilidade/100.0)
                st.session_state["picos_list_state"] = picos_auto
                st.rerun()

            st.markdown("---")
            st.markdown("**Adicionar Pico Manualmente**")
            col_add_p, col_add_btn = st.columns([3, 1])
            with col_add_p:
                novo_pico_val = st.number_input("Novo Pico (2θ)", value=30.0, step=0.1, key="novo_pico_input")
            with col_add_btn:
                st.write("")
                if st.button("➕ Adicionar", use_container_width=True):
                    if novo_pico_val not in st.session_state["picos_list_state"]:
                        st.session_state["picos_list_state"].append(novo_pico_val)
                        st.session_state["picos_list_state"].sort()
                        st.rerun()

            df_cif_ref_wh = None
            if fichas_cif_disponiveis:
                st.markdown("---")
                st.markdown("**Referência CIF (Opcional)**")
                ficha_wh_sel_idx = st.selectbox(
                    "Ficha CIF de Referência", 
                    options=range(len(fichas_cif_disponiveis)), 
                    format_func=lambda x: fichas_cif_disponiveis[x]["nome"],
                    key="wh_cif_sel"
                )
                file_cif_wh_obj = fichas_cif_disponiveis[ficha_wh_sel_idx]["file"]
                df_cif_ref_wh = ler_cif(file_cif_wh_obj, anodo=anodo_selecionado)

        with col_res_wh:
            picos_list = st.session_state["picos_list_state"]

            resultados_picos = []
            for idx_p, p_center in enumerate(picos_list):
                res_p = calcular_fwhm_e_scherrer_por_faixa(df_alvo_wh, p_center - pico_window_wh, p_center + pico_window_wh, k_wh, lambda_onda)
                if res_p:
                    res_cif_p = buscar_pico_ref_cif(df_cif_ref_wh, res_p['2theta_peak']) if df_cif_ref_wh is not None else None
                    hkl_p = res_cif_p["hkl"] if res_cif_p else f"Pico {idx_p+1}"
                    cif_2theta = res_cif_p["2theta_cif"] if res_cif_p else p_center
                    
                    resultados_picos.append({
                        "Usar": True,
                        "Rótulo": hkl_p,
                        "2θ Medido (°)": round(res_p['2theta_peak'], 3),
                        "2θ CIF (°)": round(cif_2theta, 3),
                        "Δ2θ (°)": round(res_p['2theta_peak'] - cif_2theta, 3),
                        "FWHM β (°)": round(res_p['fwhm_deg'], 3),
                        "D Scherrer (nm)": round(res_p['cristalito_nm'], 2),
                        "d (Å)": round(res_p['d_spacing'], 4),
                        "x_wh": res_p['x_wh'],
                        "y_wh": res_p['y_wh']
                    })

            if resultados_picos:
                st.subheader("📊 Seleção de Picos & Resultados do Cálculo")
                st.caption("💡 Para desconsiderar um pico, basta desmarcar a caixinha **'Usar'** na tabela abaixo.")
                
                df_resultados = pd.DataFrame(resultados_picos)
                
                df_editado = st.data_editor(
                    df_resultados,
                    column_config={
                        "Usar": st.column_config.CheckboxColumn("Usar?", default=True)
                    },
                    disabled=["Rótulo", "2θ Medido (°)", "2θ CIF (°)", "Δ2θ (°)", "FWHM β (°)", "D Scherrer (nm)", "d (Å)", "x_wh", "y_wh"],
                    hide_index=True,
                    key="editor_picos_wh"
                )

                df_filtrado = df_editado[df_editado["Usar"] == True]

                # Exportação da Tabela da 3ª Aba
                col_exp_t1, col_exp_t2 = st.columns(2)
                with col_exp_t1:
                    csv_tab3 = df_editado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Exportar Tabela (CSV)",
                        data=csv_tab3,
                        file_name="diffrapy_analise_picos.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_exp_t2:
                    buffer_excel = io.BytesIO()
                    with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                        df_editado.to_excel(writer, sheet_name='Análise de Picos', index=False)
                    st.download_button(
                        label="📊 Exportar Tabela (Excel .xlsx)",
                        data=buffer_excel.getvalue(),
                        file_name="diffrapy_analise_picos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                if not df_filtrado.empty:
                    d_medio = df_filtrado["D Scherrer (nm)"].mean()
                    d_std = df_filtrado["D Scherrer (nm)"].std()
                    
                    st.markdown("---")
                    st.subheader("Média Ponderada dos Picos Selecionados")
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Tamanho Médio de Cristalito (D_médio)", f"{d_medio:.2f} nm")
                    col_m2.metric("Desvio Padrão (σ)", f"± {d_std:.2f} nm" if not np.isnan(d_std) else "N/A (1 pico)")

                    # Williamson-Hall
                    st.markdown("---")
                    st.subheader("📈 Método de Williamson-Hall (W-H Plot)")
                    
                    if len(df_filtrado) >= 2:
                        x_data = df_filtrado["x_wh"].values
                        y_data = df_filtrado["y_wh"].values

                        slope, intercept = np.polyfit(x_data, y_data, 1)
                        r_squared = np.corrcoef(x_data, y_data)[0, 1] ** 2

                        d_wh_angstrom = (k_wh * lambda_onda) / intercept if intercept > 0 else np.nan
                        d_wh_nm = d_wh_angstrom / 10.0 if not np.isnan(d_wh_angstrom) else np.nan
                        microstrain = slope

                        if microstrain < -1e-6:
                            tag_deformacao = "🔴 Compressão (ε < 0)"
                        elif microstrain > 1e-6:
                            tag_deformacao = "🔵 Tração / Expansão (ε > 0)"
                        else:
                            tag_deformacao = "⚪ Sem deformação (ε ≈ 0)"

                        mW1, mW2, mW3 = st.columns(3)
                        mW1.metric("D (Williamson-Hall)", f"{d_wh_nm:.2f} nm" if not np.isnan(d_wh_nm) else "N/A")
                        mW2.metric("Microdeformação (ε)", f"{microstrain:.5f}", delta=tag_deformacao, delta_color="normal")
                        mW3.metric("Ajuste Linear (R²)", f"{r_squared:.4f}")

                        fig_wh, ax_wh = plt.subplots(figsize=(8, 4.5))
                        ax_wh.scatter(x_data, y_data, color='#E74C3C', s=60, zorder=3, label="Picos Selecionados")
                        
                        x_line = np.linspace(min(x_data)*0.9, max(x_data)*1.1, 100)
                        y_line = slope * x_line + intercept
                        ax_wh.plot(x_line, y_line, color='#2C3E50', linestyle='--', linewidth=1.5, label=f"Ajuste Linear (R² = {r_squared:.3f})")

                        for _, row_p in df_filtrado.iterrows():
                            ax_wh.annotate(
                                f"{row_p['Rótulo']} ({row_p['2θ Medido (°)']}°)",
                                (row_p['x_wh'], row_p['y_wh']),
                                textcoords="offset points",
                                xytext=(0, 8),
                                ha='center',
                                fontsize=8
                            )

                        ax_wh.set_xlabel(r"4 $\cdot$ sin($\theta$)", fontsize=11)
                        ax_wh.set_ylabel(r"$\beta \cdot$ cos($\theta$) [rad]", fontsize=11)
                        ax_wh.legend(loc="upper left", fontsize=9)
                        ax_wh.grid(True, linestyle="--", alpha=0.5)

                        st.pyplot(fig_wh)
                    else:
                        st.info("💡 Selecione ao menos **2 picos** na tabela acima para gerar a regressão linear de Williamson-Hall.")
                else:
                    st.warning("Nenhum pico marcado na tabela. Marque ao menos uma caixa para calcular a média.")
            else:
                st.warning("Nenhum pico válido foi detectado nas posições informadas.")

else:
    st.info("Para começar, faça o upload de uma ou mais amostras no painel lateral.")