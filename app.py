import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick 
from datetime import datetime
from io import BytesIO

# ==============================================================================
# --- 1. PENGATURAN HALAMAN & KONFIGURASI AWAL ---
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Optimasi Portofolio ABC",
    layout="wide"
)

# Daftar 24 emiten yang akan digunakan
USER_TICKERS = [
    "ANTM.JK", "INCO.JK", "ASII.JK", "INDF.JK", "AUTO.JK", "INTP.JK",
    "AVIA.JK", "JSMR.JK", "BBCA.JK", "KLBF.JK", "BBNI.JK", "MTEL.JK",
    "BBRI.JK", "SIDO.JK", "BBTN.JK", "SMGR.JK", "BMRI.JK", "SMSM.JK",
    "DSNG.JK", "SSMS.JK", "EMTK.JK", "UNTR.JK", "ICBP.JK", "UNVR.JK", "PGEO.JK"
]

# Peta Sektor Manual (Untuk Keperluan Boxplot)
SECTOR_MAP = {
    "Basic Materials": ["ANTM.JK", "INCO.JK", "INTP.JK", "SMGR.JK", "AVIA.JK"],
    "Financials": ["BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK"],
    "Consumer Non-Cyclicals": ["INDF.JK", "ICBP.JK", "SIDO.JK", "SSMS.JK", "DSNG.JK", "UNVR.JK"],
    "Consumer Cyclicals": ["ASII.JK", "AUTO.JK", "SMSM.JK"],
    "Infrastructure": ["JSMR.JK", "MTEL.JK", "PGEO.JK"],
    "Technology": ["EMTK.JK"],
    "Healthcare": ["KLBF.JK"],
    "Industrials": ["UNTR.JK"]
}

# ==============================================================================
# --- 2. FUNGSI-FUNGSI INTI (ALGORITMA & KALKULASI) ---
# ==============================================================================

def calculate_objective(weights, z_vector, mean_returns, cov_matrix, lambda_val):
    active_weights = weights * z_vector
    portfolio_return = np.sum(mean_returns * active_weights)
    portfolio_variance = np.dot(active_weights.T, np.dot(cov_matrix, active_weights))
    objective = lambda_val * portfolio_variance - (1 - lambda_val) * portfolio_return
    return objective

def arrange_function(weights, z_vector, K, lower_bound, upper_bound, num_assets):
    weights = np.copy(weights)
    z_vector = np.copy(z_vector)
    weights[z_vector == 0] = 0
    selected_indices = np.where(z_vector == 1)[0]
    K_star = len(selected_indices)

    while K_star > K:
        if K_star > 0:
            min_weight_idx_global = selected_indices[np.argmin(weights[selected_indices])]
            weights[min_weight_idx_global], z_vector[min_weight_idx_global] = 0, 0
            selected_indices = np.where(z_vector == 1)[0]
            K_star = len(selected_indices)
        else: break

    while K_star < K:
        unselected_indices = np.where(z_vector == 0)[0]
        if len(unselected_indices) > 0:
            add_idx = np.random.choice(unselected_indices)
            z_vector[add_idx], weights[add_idx] = 1, lower_bound
            selected_indices = np.where(z_vector == 1)[0]
            K_star = len(selected_indices)
        else: break

    selected_indices = np.where(z_vector == 1)[0]
    if len(selected_indices) > 0:
        total_weight = np.sum(weights[selected_indices])
        if total_weight > 0:
            weights[selected_indices] /= total_weight
        else:
            weights[selected_indices] = 1.0 / K

        for _ in range(20):
            eta = np.sum(np.maximum(0, weights - upper_bound * z_vector))
            theta = np.sum(np.maximum(0, lower_bound * z_vector - weights))
            if eta == 0 and theta == 0:
                break
            
            t_j = np.zeros(num_assets)
            e_j = np.zeros(num_assets)
            
            selected_indices_set = set(selected_indices)
            for idx in selected_indices_set:
                t_j[idx] = max(0, upper_bound - weights[idx])
                e_j[idx] = max(0, weights[idx] - lower_bound)
            
            delta_star = np.sum(t_j)
            epsilon_star = np.sum(e_j)

            for j in selected_indices_set:
                if t_j[j] > 0 and delta_star > 0:
                    weights[j] += (t_j[j] / delta_star) * eta
                else:
                    weights[j] = upper_bound if t_j[j] <= 0 else weights[j]

                if e_j[j] > 0 and epsilon_star > 0:
                    weights[j] -= (e_j[j] / epsilon_star) * theta
                else:
                    weights[j] = lower_bound if e_j[j] <= 0 else weights[j]
    
    weights[z_vector == 0] = 0
    final_sum = np.sum(weights)
    if final_sum > 0:
        weights /= final_sum

    return weights, z_vector


def abc_for_ccmv(mean_returns, cov_matrix, K, lambda_val, num_assets,
                 food_sources, max_iter, limit,
                 lower_bound=0.01, upper_bound=1.0):
    search_space_min = 0.0
    search_space_max = upper_bound
    solutions = []
    for _ in range(food_sources):
        rand_vector = np.random.rand(num_assets)
        raw_weights = search_space_min + rand_vector * (search_space_max - search_space_min)
        raw_z = np.zeros(num_assets)
        if K > 0 and K <= num_assets:
            k_indices = np.random.choice(range(num_assets), K, replace=False)
            raw_z[k_indices] = 1
        weights, z_vec = arrange_function(raw_weights, raw_z, K, lower_bound, upper_bound, num_assets)
        solutions.append({'weights': weights, 'z': z_vec})
    
    fitness = np.array([calculate_objective(s['weights'], s['z'], mean_returns, cov_matrix, lambda_val) for s in solutions])
    trial_counters = np.zeros(food_sources)
    best_fitness_idx = np.argmin(fitness)
    best_solution = solutions[best_fitness_idx]
    best_fitness = fitness[best_fitness_idx]
    
    for it in range(max_iter):
        for i in range(food_sources):
            k = np.random.randint(0, food_sources)
            while k == i: k = np.random.randint(0, food_sources)
            
            phi = np.random.uniform(-1, 1)
            j = np.random.randint(0, num_assets)
            
            new_weights = np.copy(solutions[i]['weights'])
            new_z = np.copy(solutions[i]['z'])
            
            new_weights[j] += phi * (solutions[i]['weights'][j] - solutions[k]['weights'][j])
            new_weights[j] = np.clip(new_weights[j], 0, upper_bound)
            
            if new_weights[j] > 0 and new_z[j] == 0:
                new_z[j] = 1

            final_weights, final_z = arrange_function(new_weights, new_z, K, lower_bound, upper_bound, num_assets)
            new_fitness = calculate_objective(final_weights, final_z, mean_returns, cov_matrix, lambda_val)

            if new_fitness < fitness[i]:
                solutions[i] = {'weights': final_weights, 'z': final_z}
                fitness[i] = new_fitness
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
        
        fit_min = np.min(fitness)
        fit_positive = fitness - fit_min + 1e-9
        probabilities = (1 / fit_positive) / np.sum(1 / fit_positive)

        for _ in range(food_sources):
            if np.sum(probabilities) > 0:
                i = np.random.choice(range(food_sources), p=probabilities)
            else:
                i = np.random.randint(0, food_sources)
            
            k = np.random.randint(0, food_sources)
            while k == i: k = np.random.randint(0, food_sources)
            phi = np.random.uniform(-1, 1)
            j = np.random.randint(0, num_assets)
            
            new_weights = np.copy(solutions[i]['weights'])
            new_z = np.copy(solutions[i]['z'])
            new_weights[j] += phi * (solutions[i]['weights'][j] - solutions[k]['weights'][j])
            new_weights[j] = np.clip(new_weights[j], 0, upper_bound)
            if new_weights[j] > 0 and new_z[j] == 0:
                new_z[j] = 1
            
            final_weights, final_z = arrange_function(new_weights, new_z, K, lower_bound, upper_bound, num_assets)
            new_fitness = calculate_objective(final_weights, final_z, mean_returns, cov_matrix, lambda_val)
            
            if new_fitness < fitness[i]:
                solutions[i] = {'weights': final_weights, 'z': final_z}
                fitness[i] = new_fitness
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
                
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = solutions[current_best_idx]
        
        for i in range(food_sources):
            if trial_counters[i] > limit:
                rand_vector = np.random.rand(num_assets)
                raw_weights = search_space_min + rand_vector * (search_space_max - search_space_min)
                raw_z = np.zeros(num_assets)
                if K > 0 and K <= num_assets:
                    k_indices = np.random.choice(range(num_assets), K, replace=False)
                    raw_z[k_indices] = 1
                weights, z_vec = arrange_function(raw_weights, raw_z, K, lower_bound, upper_bound, num_assets)
                solutions[i] = {'weights': weights, 'z': z_vec}
                fitness[i] = calculate_objective(weights, z_vec, mean_returns, cov_matrix, lambda_val)
                trial_counters[i] = 0
                
    return best_solution

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, rf_daily):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - rf_daily) / portfolio_std if portfolio_std != 0 else 0
    return portfolio_return, portfolio_std, sharpe_ratio

# ==============================================================================
# --- 3. FUNGSI UNTUK TAMPILAN (PAGES) ---
# ==============================================================================

def page_selamat_datang():
    st.title("Selamat Datang di Aplikasi Optimasi Portofolio")
    st.markdown("---")
    st.subheader("Maksimalkan Potensi Investasi Anda dengan *Artificial Bee Colony* (ABC)")
    st.write(
        """
        Aplikasi ini dirancang untuk membantu Anda membangun portofolio saham yang optimal 
        berdasarkan model *Cardinality Constrained Mean-Variance* (CCMV). Dengan memanfaatkan 
        kecerdasan algoritma ABC, kami akan menganalisis dan menemukan kombinasi saham terbaik 
        untuk mencapai keseimbangan antara **return** yang tinggi dan **risiko** yang terkendali.
        """
    )
    st.info("👈 **Mulai dengan memilih menu di sidebar kiri.** Anda bisa membaca panduan atau langsung menuju ke halaman optimasi.", icon="ℹ️")

def page_panduan():
    st.title("Panduan Penggunaan Dashboard")
    st.markdown("---")
    st.write("Ikuti langkah-langkah mudah berikut untuk mendapatkan hasil portofolio optimal Anda:")

    st.markdown("""
    **1. Buka Menu Optimasi Portofolio**
    - Pada sidebar di sebelah kiri, pilih menu **'Optimasi Portofolio'**.

    **2. Atur Parameter Investasi Anda**
    - **Pilih Rentang Tanggal:** Tentukan periode data historis harga saham yang akan dianalisis. Semakin panjang rentangnya, semakin banyak data yang digunakan.
    - **Pilih Emiten Saham:** Pilih emiten saham dari daftar yang tersedia. Saham-saham ini adalah kandidat untuk portofolio Anda.
    - **Input Nominal Investasi:** Masukkan total dana yang ingin Anda investasikan. Nilai ini digunakan untuk simulasi alokasi dana.
    - **Input Risk-Free Rate:** Masukkan tingkat return bebas risiko tahunan yang dapat dilihat pada link terlampir, nilai ini digunakan sebagai acuan dalam proses eliminasi saham.

    **3. Mulai Proses Optimasi**
    - Setelah semua parameter diatur, klik tombol **'🚀 Mulai Optimasi'**.
    - Harap bersabar, proses ini mungkin membutuhkan beberapa menit karena melibatkan komputasi yang kompleks.

    **4. Analisis Hasil**
    - Aplikasi akan menampilkan serangkaian hasil secara bertahap:
        - **Eliminasi Saham:** Tabel dan grafik saham yang lolos seleksi awal.
        - **Analisis Volatilitas:** Boxplot distribusi return per sektor untuk saham yang lolos.
        - **Pencarian K Optimum:** Tabel dan grafik interaktif tren Sharpe Ratio.
        - **Efficient Frontier:** Grafik interaktif yang memetakan semua kemungkinan portofolio.
        - **Hasil Akhir:** Ringkasan kinerja, alokasi bobot, dan simulasi investasi.

    **5. Unduh Laporan**
    - Di bagian paling bawah, Anda akan menemukan tombol **'📥 Unduh Hasil ke Excel'** untuk menyimpan semua tabel data ke dalam satu file Excel.
    """)
    st.success("Selamat mencoba dan semoga investasi Anda menguntungkan!")


def page_optimasi():
    st.title("Optimasi Portofolio dengan ABC")
    st.markdown("---")

    # --- SIDEBAR UNTUK INPUT ---
    st.sidebar.header("Atur Parameter Anda")
    start_date = st.sidebar.date_input("Tanggal Mulai", datetime(2025, 3, 1))
    end_date = st.sidebar.date_input("Tanggal Selesai", datetime(2025, 8, 30))
    
    selected_tickers = st.sidebar.multiselect(
        "Pilih Emiten Saham",
        options=USER_TICKERS,
        default=USER_TICKERS 
    )
    
    jumlah_investasi = st.sidebar.number_input(
        "Nominal Investasi (Rp)", 
        min_value=1000000, 
        value=10000000, 
        step=1000000,
        format="%d"
    )
    
    st.sidebar.markdown("Risk-Free Rate Tahunan (%) - [Sumber BI](https://www.bi.go.id/id/statistik/indikator/BI-Rate.aspx)")
    rf_rate_annual = st.sidebar.number_input(
        "Input RFR Tahunan (%)", 
        min_value=0.0, 
        value=5.0, 
        step=0.1,
        format="%.2f",
        label_visibility="collapsed" 
    ) / 100.0
    rf_daily = rf_rate_annual / 365
    
    # --- Tombol untuk memulai proses ---
    if st.sidebar.button("🚀 Mulai Optimasi"):
        if len(selected_tickers) < 2:
            st.error("⚠️ Harap pilih minimal 2 emiten saham untuk memulai optimasi.")
        elif start_date >= end_date:
            st.error("⚠️ Tanggal mulai harus sebelum tanggal selesai.")
        else:
            run_optimization_process(start_date, end_date, selected_tickers, jumlah_investasi, rf_daily)

# ==============================================================================
# --- 4. FUNGSI UTAMA UNTUK PROSES OPTIMASI & TAMPILAN HASIL ---
# ==============================================================================

def run_optimization_process(start_date, end_date, tickers, jumlah_investasi, rf_daily):
    
    excel_sheets = {}

    # --- 1. DOWNLOAD DATA ---
    with st.spinner("Mengunduh data harga saham... 📥"):
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
            if data.empty:
                st.error("Gagal mengunduh data. Periksa kembali ticker saham atau rentang tanggal.")
                return
            returns = data.pct_change().dropna()
            if returns.empty:
                st.error("⚠️ Tidak ada data trading yang ditemukan untuk rentang tanggal yang dipilih. Silakan pilih rentang tanggal yang berbeda.")
                return
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mengunduh data: {e}")
            return
            
    st.success("Data harga saham berhasil diunduh.")

    # --- 2. ELIMINASI SAHAM ---
    st.subheader("1. Proses Eliminasi Saham Berdasarkan Risk-Free Rate")
    with st.spinner("Menghitung return dan melakukan eliminasi... 🔬"):
        # HITUNG GEOMETRIC MEAN DI SINI (Digunakan lagi nanti di Boxplot)
        geom_mean_returns = ((1 + returns).prod())**(1/len(returns)) - 1
        
        risks = returns.std()
        elimination_status = ['Lolos' if geom_mean_returns.get(ticker, 0) > rf_daily else 'Tereliminasi' for ticker in tickers]
        
        results_df = pd.DataFrame({
            'Emiten': tickers,
            'Expected Return Harian': [geom_mean_returns.get(t, 0) for t in tickers],
            'Risiko Harian': [risks.get(t, 0) for t in tickers],
            'Keterangan': elimination_status
        })
        excel_sheets['Eliminasi Saham'] = results_df.copy()
        
        filtered_stocks_tickers = geom_mean_returns[geom_mean_returns > rf_daily].index.tolist()
        filtered_returns = returns[filtered_stocks_tickers]
        filtered_geom_mean = geom_mean_returns[filtered_stocks_tickers]
        
        st.write(f"Dari **{len(tickers)}** emiten, **{len(filtered_stocks_tickers)}** lolos seleksi dan **{len(tickers) - len(filtered_stocks_tickers)}** tereliminasi.")
        st.dataframe(results_df.style.apply(
            lambda row: ['background-color: #d4edda' if row.Keterangan == 'Lolos' else 'background-color: #f8d7da'] * len(row), axis=1
        ).format({'Expected Return Harian': '{:.3%}', 'Risiko Harian': '{:.3%}'}))

        results_df['Color'] = results_df['Keterangan'].apply(lambda x: '#4CAF50' if x == 'Lolos' else '#F44336')
        fig_elim = px.bar(results_df, x='Emiten', y='Expected Return Harian',
                            color='Keterangan',
                            color_discrete_map={'Lolos': '#4CAF50', 'Tereliminasi': '#F44336'},
                            title='Hasil Eliminasi Emiten vs Risk-Free Rate', 
                            labels={'Expected Return Harian': 'Return Harian'},
                            hover_data={'Emiten': True, 'Expected Return Harian': ':.3%', 'Risiko Harian': ':.3%', 'Keterangan': True})
        
        fig_elim.add_hline(y=rf_daily, line_dash="dash", line_color="dodgerblue",
                            annotation_text=f'RFR Harian ({rf_daily:.3%})', annotation_position="bottom right")

        fig_elim.update_layout(
            height=500, 
            xaxis_tickangle=-90, 
            yaxis_title='Expected Return Harian (%)', 
            yaxis_tickformat='.2%',
            title_x=0.5,
            xaxis_categoryorder='total descending'
        )

        st.plotly_chart(fig_elim, use_container_width=True)
        num_assets = len(filtered_stocks_tickers)
        if num_assets < 2:
            st.error(f"⚠️ Jumlah saham yang lolos seleksi ({num_assets}) kurang dari 2. Proses optimasi tidak dapat dilanjutkan.")
            return

    # --- 3. ANALISIS BOXPLOT GRID (REVISI: Menggunakan Geometric Mean) ---
    st.subheader("2. Analisis Distribusi Return per Sektor (Boxplot)")
    
    with st.spinner("Membuat Visualisasi Boxplot per Sektor... 📊"):
        
        sector_risks = {} 
        valid_sectors_count = 0 # Counter untuk mengatur grid
        
        # Buat layout 2 kolom untuk grid
        cols = st.columns(2) 

        # Iterasi setiap sektor di SECTOR_MAP
        for sector_name, sector_tickers in SECTOR_MAP.items():
            # Filter ticker yang lolos seleksi
            available_tickers = [t for t in sector_tickers if t in filtered_returns.columns]

            if available_tickers:
                sector_data = filtered_returns[available_tickers]
                
                # Hitung rata-rata risiko untuk kesimpulan nanti
                sector_avg_risk = sector_data.std().mean()
                sector_risks[sector_name] = sector_avg_risk

                # --- BAGIAN 1: PERSIAPAN DATA STATISTIK UNTUK EXCEL (REVISI) ---
                stats_list = []
                for ticker_code in available_tickers:
                    series = sector_data[ticker_code]
                    
                    # --- AMBIL NILAI GEOM MEAN YANG SUDAH DIHITUNG DI AWAL ---
                    geom_mean_val = geom_mean_returns[ticker_code]
                    
                    median_val = series.median()
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1

                    stats_list.append({
                        'Emiten': ticker_code,
                        'Geometric Mean (Rata-rata Harian)': geom_mean_val, # Nilai diambil dari atas
                        'Median (Tengah)': median_val,
                        'Q1 (25%)': q1,
                        'Q3 (75%)': q3,
                        'IQR': iqr
                        # Outlier stats dihapus agar sesuai dengan logika snippet
                    })

                # Simpan ke Excel Sheets Dictionary
                df_stats = pd.DataFrame(stats_list)
                safe_sheet_name = sector_name[:31]
                excel_sheets[f"Stat {safe_sheet_name}"] = df_stats

                # --- BAGIAN 2: VISUALISASI BOXPLOT GRID (REVISI) ---
                col_idx = valid_sectors_count % 2
                with cols[col_idx]:
                    
                    # Ukuran diperkecil (7, 5) untuk streamlit
                    fig, ax = plt.subplots(figsize=(7, 5)) 
                    
                    sns.boxplot(data=sector_data, palette="Set3", showmeans=True, ax=ax,
                                meanprops={"marker":"^", "markerfacecolor":"green", "markeredgecolor":"green", "markersize": 8})

                    # Loop Label Angka
                    for i, ticker_code in enumerate(available_tickers):
                        series = sector_data[ticker_code]
                        
                        # --- AMBIL NILAI GEOM MEAN LAGI UNTUK LABEL ---
                        geom_mean_val = geom_mean_returns[ticker_code]
                        median_val = series.median()

                        # Font size diperkecil agar muat
                        # Label Geom Mean (Hijau)
                        ax.text(i + 0.1, geom_mean_val, f'Geom Mean: {geom_mean_val:.3%}',
                                horizontalalignment='left', fontsize=8, color='green', weight='bold')

                        # Label Median (Hitam)
                        ax.text(i + 0.1, median_val, f'Median: {median_val:.3%}',
                                horizontalalignment='left', fontsize=8, color='black', weight='bold', verticalalignment='top')
                        
                        # Tidak ada label outlier (sesuai perintah revisi)

                    ax.set_title(f'Sektor {sector_name}', fontsize=12, weight='bold') 
                    ax.set_xlabel('', fontsize=9) 
                    ax.set_ylabel('Daily Return', fontsize=9)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                    
                    # Format Y-axis ke Persen menggunakan Matplotlib Ticker
                    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
                    
                    st.pyplot(fig) # Tampilkan di kolom yang aktif
                    plt.close(fig) 
                
                valid_sectors_count += 1 

        # Kesimpulan Risiko
        if sector_risks:
            riskiest_sector = max(sector_risks, key=sector_risks.get)
            st.info(f"💡 **Kesimpulan Risiko:** Berdasarkan rata-rata standar deviasi emiten yang lolos seleksi, Sektor **{riskiest_sector}** cenderung memiliki risiko volatilitas paling tinggi dibandingkan sektor lainnya.")

        # Heatmap Kecil
        st.write("---")
        st.write("**Matriks Korelasi Antar Saham yang Lolos Eliminasi**")
        corr_matrix = filtered_returns.corr()
        fig_corr = px.imshow(corr_matrix, 
                                text_auto=True, 
                                aspect="auto", 
                                color_continuous_scale='RdBu_r', 
                                title='Matriks Korelasi Return Saham')
        fig_corr.update_layout(height=450, width=600, title_x=0.5) 
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- 4. OPTIMASI (K Search) ---
    st.markdown("---")
    st.subheader("3. Tahap Optimasi: Pencarian K Optimum (λ = 0.90)")
    
    cov_matrix = filtered_returns.cov()
    PARAM_FOOD_SOURCES = 30; PARAM_MAX_ITER = 1000; PARAM_LIMIT = 200
    lambda_fixed = 0.90; k_search_results = []
    
    if num_assets < 5:
        K_range = range(2, num_assets + 1)
        st.warning(f"Jumlah saham lolos ({num_assets}) kurang dari 5. Pencarian K akan dijalankan dari 2 hingga {num_assets}.")
    else:
        K_range = range(5, num_assets + 1)
        
    progress_bar = st.progress(0, text="Memulai pencarian K Optimum...")
    status_text = st.empty()

    for i, k_val in enumerate(K_range):
        status_text.text(f"Menguji portofolio dengan {k_val} saham..."); np.random.seed(42)
        best_solution = abc_for_ccmv(mean_returns=filtered_geom_mean.values, cov_matrix=cov_matrix.values, K=k_val, lambda_val=lambda_fixed, num_assets=num_assets, food_sources=PARAM_FOOD_SOURCES, max_iter=PARAM_MAX_ITER, limit=PARAM_LIMIT)
        weights = best_solution['weights']
        ret, std, sharpe = calculate_portfolio_performance(weights, filtered_geom_mean.values, cov_matrix.values, rf_daily)
        k_search_results.append({'K': k_val, 'Return': ret, 'Risiko': std, 'Sharpe Ratio': sharpe})
        progress_bar.progress((i + 1) / len(K_range), text=f"Menguji K = {k_val}... Selesai.")

    status_text.text("Pencarian K Optimum selesai."); df_k_search = pd.DataFrame(k_search_results)
    excel_sheets['Pencarian K'] = df_k_search.copy()
    best_k_index = df_k_search['Sharpe Ratio'].idxmax(); best_k_result = df_k_search.loc[best_k_index]
    K_optimum = int(best_k_result['K'])
    
    st.dataframe(df_k_search.style.highlight_max(subset=['Sharpe Ratio'], color='lightgreen').format({'Return': '{:.3%}', 'Risiko': '{:.3%}', 'Sharpe Ratio': '{:.3%}'}))
    st.success(f"✅ **K Optimum ditemukan: {K_optimum}** dengan Sharpe Ratio tertinggi sebesar **{best_k_result['Sharpe Ratio']:.3%}**.")
    
    # --- VISUALISASI PERGERAKAN SHARPE RATIO (INTERAKTIF PLOTLY) ---
    st.write("---")
    st.write("**Visualisasi Pergerakan Sharpe Ratio per K (Interaktif)**")
    
    # Konversi data ke format persen untuk display
    df_chart = df_k_search.copy()
    df_chart['Sharpe Percent'] = df_chart['Sharpe Ratio'] * 100
    best_sharpe_percent = best_k_result['Sharpe Ratio'] * 100
    
    # Membuat Grafik Plotly
    fig_sharpe = go.Figure()

    # 1. Garis Utama
    fig_sharpe.add_trace(go.Scatter(
        x=df_chart['K'], 
        y=df_chart['Sharpe Percent'],
        mode='lines+markers',
        name='Sharpe Ratio',
        line=dict(color='#34495e', width=3),
        marker=dict(size=8, color='#3498db'),
        hovertemplate='<b>Jumlah Saham (K): %{x}</b><br>Sharpe Ratio: %{y:.3f}%<extra></extra>'
    ))

    # 2. Highlight Titik Optimum (Bintang Merah)
    fig_sharpe.add_trace(go.Scatter(
        x=[K_optimum], 
        y=[best_sharpe_percent],
        mode='markers',
        name='K Optimum',
        marker=dict(color='#e74c3c', size=18, symbol='star', line=dict(width=2, color='white')),
        hovertemplate='<b>K OPTIMUM: %{x}</b><br>Max Sharpe: %{y:.3f}%<extra></extra>'
    ))

    # 3. Layout Styling
    fig_sharpe.update_layout(
        title={
            'text': 'Pergerakan Sharpe Ratio Berdasarkan Jumlah Saham (K)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Jumlah Saham (K)',
        yaxis_title='Sharpe Ratio (%)',
        template='plotly_white',
        hovermode='x unified', # Mode hover interaktif yang nyaman
        xaxis=dict(
            tickmode='linear',
            dtick=1 # Memastikan sumbu X menampilkan angka bulat (1, 2, 3...)
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # --- 5. EFFICIENT FRONTIER ---
    st.markdown("---")
    st.subheader(f"4. Tahap Optimasi: Membuat Efficient Frontier (K={K_optimum})")
    
    lambda_range = np.arange(0, 1.02, 0.02)
    frontier_portfolios = []
    
    progress_bar_ef = st.progress(0, text="Memulai pembuatan Efficient Frontier...")
    status_text_ef = st.empty()

    for i, lam_val in enumerate(lambda_range):
        status_text_ef.text(f"Menjalankan ABC untuk Lambda = {lam_val:.2f}..."); np.random.seed(42)
        best_solution = abc_for_ccmv(mean_returns=filtered_geom_mean.values, cov_matrix=cov_matrix.values, K=K_optimum, lambda_val=lam_val, num_assets=num_assets, food_sources=PARAM_FOOD_SOURCES, max_iter=PARAM_MAX_ITER, limit=PARAM_LIMIT)
        weights = best_solution['weights']
        ret, std, sharpe = calculate_portfolio_performance(weights, filtered_geom_mean.values, cov_matrix.values, rf_daily)
        frontier_portfolios.append({'lambda': lam_val, 'return': ret, 'risk': std, 'sharpe': sharpe, 'weights': weights})
        progress_bar_ef.progress((i + 1) / len(lambda_range), text=f"Lambda = {lam_val:.2f}... Selesai.")

    status_text_ef.text("Pembuatan Efficient Frontier selesai."); df_frontier = pd.DataFrame(frontier_portfolios)
    df_frontier_excel = df_frontier.copy()
    df_frontier_excel['weights'] = df_frontier_excel['weights'].apply(lambda w: str(np.round(w, 4)))
    excel_sheets['Data Frontier'] = df_frontier_excel
    optimal_portfolio = df_frontier.loc[df_frontier['sharpe'].idxmax()]

    fig_frontier = px.scatter(df_frontier, x='risk', y='return', color='sharpe', color_continuous_scale='Viridis', labels={'risk': 'Risiko Harian (Std Dev)', 'return': 'Return Harian', 'sharpe': 'Sharpe Ratio'}, hover_data={'risk': ':.3%', 'return': ':.3%', 'sharpe': ':.3%', 'lambda': ':.2f'})
    fig_frontier.add_trace(go.Scatter(x=[optimal_portfolio['risk']], y=[optimal_portfolio['return']], mode='markers', marker=dict(color='red', size=18, symbol='star', line=dict(width=1, color='black')), name=f'Optimal (λ: {optimal_portfolio["lambda"]:.2f}, Sharpe: {optimal_portfolio["sharpe"]:.3%})'))
    fig_frontier.update_layout(height=600, template='plotly_white', title={'text': f'Efficient Frontier Interaktif (K = {K_optimum})', 'x': 0.5}, coloraxis_colorbar=dict(title='Sharpe Ratio'), yaxis_title='Return Harian (%)', yaxis_tickformat='.2%')
    st.plotly_chart(fig_frontier, use_container_width=True)

    # --- 6. HASIL AKHIR ---
    st.markdown("---")
    st.subheader("5. Hasil Akhir: Portofolio Optimal")
    st.info(f"Parameter Optimal: **K = {K_optimum}** saham, **Lambda (λ) = {optimal_portfolio['lambda']:.2f}**")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Summary Kinerja Portofolio**")
        summary_data = {'Metrik': ['Expected Daily Return', 'Daily Risk (Std Dev)', 'Sharpe Ratio'], 'Nilai': [optimal_portfolio['return'], optimal_portfolio['risk'], optimal_portfolio['sharpe']]}
        summary_df = pd.DataFrame(summary_data).set_index('Metrik')
        st.dataframe(summary_df.style.format({'Nilai': '{:.3%}'}))
        excel_sheets['Ringkasan Optimal'] = summary_df.copy()

    with col2:
        st.write("**📝 Interpretasi Kinerja**")
        daily_return_rp = optimal_portfolio['return'] * jumlah_investasi
        daily_risk_rp = optimal_portfolio['risk'] * jumlah_investasi
        
        st.markdown(f"""
        Dengan modal **Rp {jumlah_investasi:,.0f}**, portofolio ini memiliki:
        - **Potensi Keuntungan Harian:** Rata-rata **`{optimal_portfolio['return']:.3%}`** atau sekitar **`Rp {daily_return_rp:,.0f}`**.
        - **Potensi Risiko Harian:** Fluktuasi nilai harian sebesar **`{optimal_portfolio['risk']:.3%}`** atau sekitar **`±Rp {daily_risk_rp:,.0f}`**.
        
        *Ini adalah estimasi berdasarkan data historis dan bukan jaminan kinerja di masa depan.*
        """)

    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    with col3:
        st.write(f"**Alokasi Bobot & Investasi (Rp {jumlah_investasi:,.0f})**")
        optimal_weights_series = pd.Series(optimal_portfolio['weights'], index=filtered_stocks_tickers)
        allocated_stocks = optimal_weights_series[optimal_weights_series > 1e-5]
        alokasi_df = pd.DataFrame(allocated_stocks, columns=['Bobot']); alokasi_df['Jumlah Investasi (Rp)'] = alokasi_df['Bobot'] * jumlah_investasi
        alokasi_df.index.name = 'Emiten'
        st.dataframe(alokasi_df.style.format({'Bobot': '{:.2%}', 'Jumlah Investasi (Rp)': 'Rp {:,.0f}'}))
        excel_sheets['Alokasi Bobot'] = alokasi_df.copy()
        
    with col4:
        st.write("**📝 Interpretasi Alokasi**")
        st.markdown(f"""
        Tabel di samping menunjukkan bagaimana dana investasi Anda sebesar **Rp {jumlah_investasi:,.0f}** didistribusikan ke dalam **{K_optimum}** saham terpilih.
        
        Bobot terbesar dialokasikan pada saham dengan kombinasi return dan risiko terbaik menurut analisis algoritma.
        """)

    st.write("**Visualisasi Alokasi Bobot Interaktif**")
    alokasi_df_reset = alokasi_df.reset_index()
    fig_bar_alokasi = px.bar(alokasi_df_reset.sort_values('Bobot', ascending=False), 
                            x='Emiten', y='Bobot',
                            title=f'Alokasi Bobot Portofolio Optimal (K={K_optimum})',
                            labels={'Bobot': 'Bobot Alokasi', 'Emiten': 'Emiten Saham'},
                            text='Bobot')
    fig_bar_alokasi.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig_bar_alokasi.update_layout(yaxis_tickformat='.0%', title_x=0.5, yaxis_title='Bobot Alokasi')
    st.plotly_chart(fig_bar_alokasi, use_container_width=True)
    
    st.markdown("---")
    st.subheader("6. Unduh Laporan")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df_data in excel_sheets.items():
            df_data.to_excel(writer, sheet_name=sheet_name, index=True if sheet_name not in ['Pencarian K', 'Eliminasi Saham'] and not sheet_name.startswith('Stat') else False)
    st.download_button(label="📥 Unduh Hasil ke Excel", data=output.getvalue(), file_name=f"hasil_optimasi_abc_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet")

# ==============================================================================
# --- 5. NAVIGASI UTAMA APLIKASI ---
# ==============================================================================
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ("Halaman Utama", "Panduan Dashboard", "Optimasi Portofolio"))

if menu == "Halaman Utama":
    page_selamat_datang()
elif menu == "Panduan Dashboard":
    page_panduan()
elif menu == "Optimasi Portofolio":
    page_optimasi()
