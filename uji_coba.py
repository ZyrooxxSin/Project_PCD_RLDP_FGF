import os
import time
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 1. FAST GUIDED FILTER (Penyempurna Tepi Peta Transmisi - Kompleksitas O(N))
# ==============================================================================
def fast_guided_filter(I_guide, p_input, r=30, eps=1e-2, s=4):
    """
    Menyempurnakan peta transmisi kasar (p_input) mengikuti batas tepi objek pada
    citra panduan (I_guide) menggunakan Fast Guided Filter (subsampling rasio s).
    """
    h, w = p_input.shape[:2]
    h_sub, w_sub = max(2, h // s), max(2, w // s)
    r_sub = max(1, r // s)
    ksize = (2 * r_sub + 1, 2 * r_sub + 1)

    if I_guide.ndim == 3:
        I_gray = cv2.cvtColor((I_guide * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        I_gray = I_guide.astype(np.float64)

    # Langkah 1: Subsampling untuk mempercepat komputasi
    I_sub = cv2.resize(I_gray, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)
    p_sub = cv2.resize(p_input.astype(np.float64), (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)

    # Langkah 2: Statistik lokal pada jendela r_sub
    mean_I = cv2.boxFilter(I_sub, cv2.CV_64F, ksize)
    mean_p = cv2.boxFilter(p_sub, cv2.CV_64F, ksize)
    mean_Ip = cv2.boxFilter(I_sub * p_sub, cv2.CV_64F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I_sub * I_sub, cv2.CV_64F, ksize)
    var_I = mean_II - mean_I * mean_I

    # Langkah 3: Koefisien regresi linear lokal (a_k, b_k)
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)

    # Langkah 4: Upsampling kembali ke resolusi asli
    mean_a_up = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
    mean_b_up = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)

    # Langkah 5: Peta transmisi halus dengan batas tepi objek tajam
    q = mean_a_up * I_gray + mean_b_up
    return np.clip(q, 0.0, 1.0)


def hitung_statistik_patch(img_float, patch_size):
    """Menghitung rata-rata lokal (mu) dan standar deviasi lokal (sigma) pada patch."""
    ksize = (patch_size, patch_size)
    mu = cv2.boxFilter(img_float, cv2.CV_64F, ksize)
    mean_sq = cv2.boxFilter(img_float ** 2, cv2.CV_64F, ksize)
    sigma = np.sqrt(np.maximum(0.0, mean_sq - mu ** 2))
    return mu, sigma


def estimasi_cahaya_atmosfer(I, theta_rdcp):
    """Mengestimasi cahaya atmosfer global (A) dari 0.1% piksel paling terang pada RDCP."""
    jml_top = max(1, int(theta_rdcp.size * 0.001))
    idx_terang = np.argpartition(theta_rdcp.ravel(), -jml_top)[-jml_top:]
    A = float(np.max(np.mean(I.reshape(-1, 3)[idx_terang], axis=0)))
    return np.clip(A, 180.0, 245.0)


# ==============================================================================
# 2. METODE 1: RLDP MURNI (Ning et al., 2023 - Dengan Simulasi Loop CRME)
# ==============================================================================
def proses_rldp_murni(img_bgr, patch_optimal=45, omega=0.95, s_const=100.0,
                      t0=0.1, s0=0.25, simulasi_loop_crme=True):
    """
    Implementasi RLDP asli (RDCP + RLCP) tanpa Fast Guided Filter.
    Menyertakan simulasi perulangan evaluasi patch CRME (ukuran 20 s.d. 120)
    sesuai Persamaan (16) pada jurnal Ning et al. (2023).
    """
    t_start = time.perf_counter()
    I = img_bgr.astype(np.float64)

    # Simulasi beban iterasi pencarian ukuran patch CRME (k = 20, 30, ..., 120)
    if simulasi_loop_crme:
        prev_score = 0.0
        for k_patch in range(20, 121, 10):
            mu_tmp, sig_tmp = hitung_statistik_patch(I, k_patch)
            rdcp_tmp = np.min(np.maximum(0.0, mu_tmp - 3.0 * sig_tmp), axis=2)
            curr_score = float(np.cbrt(np.mean(rdcp_tmp ** 3) + 1e-6))
            if abs(curr_score - prev_score) < 1e-4 and k_patch >= patch_optimal:
                break
            prev_score = curr_score

    # --- TAHAP 1: Robust Dark Channel Prior (RDCP) untuk Haze Removal ---
    mu_I, sigma_I = hitung_statistik_patch(I, patch_optimal)
    theta_rdcp = np.min(np.maximum(0.0, mu_I - 3.0 * sigma_I), axis=2)
    A = estimasi_cahaya_atmosfer(I, theta_rdcp)

    # Peta transmisi kabut kasar t_tilde (tanpa penyempurnaan tepi FGF)
    t_map = np.clip(1.0 - omega * (theta_rdcp / A), t0, 1.0)
    J = np.clip((I - A) / t_map[:, :, None] + A, 0.0, 255.0)

    # --- TAHAP 2: Robust Light Channel Prior (RLCP) untuk Shadow Removal ---
    mu_J, sigma_J = hitung_statistik_patch(J, patch_optimal)
    theta_rlcp = np.max(np.minimum(255.0, mu_J + 3.0 * sigma_J), axis=2)

    minmaxv = max(float(np.percentile(np.max(J, axis=2), 5)), 50.0)
    s_tilde = (theta_rlcp + s_const) / (minmaxv + s_const)
    s_map = np.clip(s_tilde / (np.max(s_tilde) + 1e-6), s0, 1.0)

    S_akhir = np.clip(J / s_map[:, :, None], 0, 255).astype(np.uint8)
    waktu = time.perf_counter() - t_start
    return S_akhir, t_map, s_map, waktu


# ==============================================================================
# 3. METODE 2: RLDP + FAST GUIDED FILTER (Metode Modifikasi Terkalibrasi)
# ==============================================================================
def proses_rldp_fgf(img_bgr, patch_tetap=27, omega=0.90, s_const=100.0,
                    t0=0.15, s0=0.45, fgf_r=30, fgf_eps=1e-2, fgf_sub=4):
    """
    Metode Modifikasi:
    Mengganti perulangan pencarian patch CRME dengan ukuran patch tetap (27x27)
    dan menyempurnakan peta transmisi kabut (t) serta bayangan (s) memakai
    Fast Guided Filter (FGF) dalam satu kali jalan (single-pass O(N)).
    """
    t_start = time.perf_counter()
    I = img_bgr.astype(np.float64)
    I_norm = I / 255.0

    # --- TAHAP 1: RDCP + Fast Guided Filter pada Peta Transmisi Kabut (t) ---
    mu_I, sigma_I = hitung_statistik_patch(I, patch_tetap)
    theta_rdcp = np.min(np.maximum(0.0, mu_I - 3.0 * sigma_I), axis=2)
    A = estimasi_cahaya_atmosfer(I, theta_rdcp)

    t_kasar = np.clip(1.0 - omega * (theta_rdcp / A), t0, 1.0)
    t_fgf = np.clip(fast_guided_filter(I_norm, t_kasar, r=fgf_r, eps=fgf_eps, s=fgf_sub), t0, 1.0)

    # Pemulihan citra bebas kabut J(x) menggunakan t_fgf yang tajam di tepi
    J = np.clip((I - A) / t_fgf[:, :, None] + A, 0.0, 255.0)

    # --- TAHAP 2: RLCP + Fast Guided Filter pada Peta Transmisi Bayangan (s) ---
    mu_J, sigma_J = hitung_statistik_patch(J, patch_tetap)
    theta_rlcp = np.max(np.minimum(255.0, mu_J + 3.0 * sigma_J), axis=2)

    minmaxv = max(float(np.percentile(np.max(J, axis=2), 5)), 50.0)
    s_tilde = (theta_rlcp + s_const) / (minmaxv + s_const)
    s_kasar = np.clip(s_tilde / (np.max(s_tilde) + 1e-6), s0, 1.0)
    s_fgf = np.clip(fast_guided_filter(J / 255.0, s_kasar, r=fgf_r, eps=fgf_eps, s=fgf_sub), s0, 1.0)

    # Pemulihan citra akhir bebas bayangan S(x)
    S_akhir = np.clip(J / s_fgf[:, :, None], 0, 255).astype(np.uint8)
    waktu = time.perf_counter() - t_start
    return S_akhir, t_fgf, s_fgf, waktu


# ==============================================================================
# 4. EVALUASI METRIK (PSNR, SSIM, Ketajaman Tepi) & GENERATOR CONTOH
# ==============================================================================
def hitung_metrik(img_uji, img_gt=None):
    gray = cv2.cvtColor(img_uji, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tajam_tepi = float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2.0)))

    if img_gt is None:
        return None, None, tajam_tepi

    u, g = img_uji.astype(np.float64), img_gt.astype(np.float64)
    mse = np.mean((u - g) ** 2)
    psnr = 100.0 if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)

    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_u, mu_g = cv2.GaussianBlur(u, (11, 11), 1.5), cv2.GaussianBlur(g, (11, 11), 1.5)
    sig_u2 = cv2.GaussianBlur(u ** 2, (11, 11), 1.5) - mu_u ** 2
    sig_g2 = cv2.GaussianBlur(g ** 2, (11, 11), 1.5) - mu_g ** 2
    sig_ug = cv2.GaussianBlur(u * g, (11, 11), 1.5) - mu_u * mu_g
    ssim = float(np.mean(((2 * mu_u * mu_g + C1) * (2 * sig_ug + C2)) /
                         ((mu_u ** 2 + mu_g ** 2 + C1) * (sig_u2 + sig_g2 + C2))))
    return psnr, ssim, tajam_tepi


def buat_sampel_otomatis(dir_in, dir_gt):
    """Membuat 2 pasang citra satelit sintetis jika folder input masih kosong."""
    os.makedirs(dir_in, exist_ok=True)
    os.makedirs(dir_gt, exist_ok=True)
    rng = np.random.default_rng(42)

    for idx in [1, 2]:
        gt = np.zeros((512, 512, 3), dtype=np.uint8)
        gt[:, :] = [45, 95, 55] if idx == 1 else [55, 85, 105]
        cv2.rectangle(gt, (290, 50), (460, 450), (195, 205, 215), -1)
        cv2.circle(gt, (150, 260), 85, (180, 190, 200), -1)
        cv2.polylines(gt, [np.array([[20, 40], [180, 220], [260, 490]])], False, (125, 80, 40), 18)
        noise = rng.integers(-10, 10, size=gt.shape, dtype=np.int16)
        gt = np.clip(gt.astype(np.int16) + noise, 10, 245).astype(np.uint8)

        y_grid, x_grid = np.mgrid[0:512, 0:512]
        t_sim = 0.45 + 0.25 * np.sin(x_grid / (75.0 * idx)) * np.cos(y_grid / 80.0)
        s_sim = np.ones((512, 512), dtype=np.float64)
        cv2.circle(s_sim, (240, 260), 140, 0.65, -1)
        s_sim = cv2.GaussianBlur(s_sim, (51, 51), 20)

        hazy = gt.astype(np.float64) * s_sim[:, :, None] * t_sim[:, :, None] + 225.0 * (1.0 - t_sim[:, :, None])
        hazy = np.clip(hazy, 0, 255).astype(np.uint8)

        nama = f"sampel_satelit_0{idx}.png"
        cv2.imwrite(os.path.join(dir_gt, nama), gt)
        cv2.imwrite(os.path.join(dir_in, nama), hazy)

    print("[INFO] Folder 'input/' kosong -> 2 pasang citra uji contoh dibuat otomatis.\n")


# ==============================================================================
# 5. EKSEKUSI UTAMA (Maksimal 10 Citra & Output Terpisah)
# ==============================================================================
def main():
    dir_in = "input"
    dir_gt = os.path.join("input", "label")
    dir_out_rldp = os.path.join("output", "1_RLDP_Murni")
    dir_out_fgf = os.path.join("output", "2_RLDP_FGF")
    dir_out_cmp = os.path.join("output", "3_Perbandingan")

    for d in [dir_in, dir_out_rldp, dir_out_fgf, dir_out_cmp]:
        os.makedirs(d, exist_ok=True)

    ekstensi = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
    daftar_img = sorted([
        f for f in glob.glob(os.path.join(dir_in, "*"))
        if os.path.isfile(f) and f.lower().endswith(ekstensi)
    ])[:10]

    if not daftar_img:
        buat_sampel_otomatis(dir_in, dir_gt)
        daftar_img = sorted(glob.glob(os.path.join(dir_in, "*.png")))

    rekap = []
    print(f"Memproses {len(daftar_img)} citra...\n")

    for i, path_img in enumerate(daftar_img, 1):
        nama_file = os.path.basename(path_img)
        nama_base = os.path.splitext(nama_file)[0]
        img_hazy = cv2.imread(path_img)
        if img_hazy is None:
            continue

        # Eksekusi kedua metode
        hasil_rldp, t_rldp, s_rldp, w_rldp = proses_rldp_murni(
            img_hazy, patch_optimal=45, simulasi_loop_crme=True
        )
        hasil_fgf, t_fgf, s_fgf, w_fgf = proses_rldp_fgf(
            img_hazy, patch_tetap=45, omega=0.95, t0=0.1, s0=0.65,
            fgf_r=30, fgf_eps=1e-2, fgf_sub=4
        )

        # Simpan ke folder terpisah
        cv2.imwrite(os.path.join(dir_out_rldp, f"{nama_base}_hasil.png"), hasil_rldp)
        cv2.imwrite(os.path.join(dir_out_rldp, f"{nama_base}_peta_t.png"), (t_rldp * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(dir_out_rldp, f"{nama_base}_peta_s.png"), (s_rldp * 255).astype(np.uint8))

        cv2.imwrite(os.path.join(dir_out_fgf, f"{nama_base}_hasil.png"), hasil_fgf)
        cv2.imwrite(os.path.join(dir_out_fgf, f"{nama_base}_peta_t.png"), (t_fgf * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(dir_out_fgf, f"{nama_base}_peta_s.png"), (s_fgf * 255).astype(np.uint8))

        # Hitung metrik evaluasi (jika ada ground-truth di input/label/)
        path_gt = os.path.join(dir_gt, nama_file)
        img_gt = cv2.imread(path_gt) if os.path.exists(path_gt) else None

        psnr_r, ssim_r, tepi_r = hitung_metrik(hasil_rldp, img_gt)
        psnr_f, ssim_f, tepi_f = hitung_metrik(hasil_fgf, img_gt)

        rekap.append({
            "Citra": nama_file,
            "PSNR_RLDP(dB)": round(psnr_r, 2) if psnr_r is not None else "-",
            "PSNR_FGF(dB)": round(psnr_f, 2) if psnr_f is not None else "-",
            "SSIM_RLDP": round(ssim_r, 4) if ssim_r is not None else "-",
            "SSIM_FGF": round(ssim_f, 4) if ssim_f is not None else "-",
            "TajamTepi_RLDP": round(tepi_r, 2),
            "TajamTepi_FGF": round(tepi_f, 2),
            "Waktu_RLDP(s)": round(w_rldp, 4),
            "Waktu_FGF(s)": round(w_fgf, 4),
        })

        # Buat panel visual 6 bagian + heatmap letak kelemahan RLDP
        fig, ax = plt.subplots(2, 3, figsize=(15, 9))
        ax[0, 0].imshow(cv2.cvtColor(img_hazy, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title("1. Citra Input Berkabut")

        ax[0, 1].imshow(cv2.cvtColor(hasil_rldp, cv2.COLOR_BGR2RGB))
        ax[0, 1].set_title(f"2. RLDP Murni (PSNR: {psnr_r:.2f} dB)" if psnr_r else "2. RLDP Murni")

        ax[0, 2].imshow(cv2.cvtColor(hasil_fgf, cv2.COLOR_BGR2RGB))
        ax[0, 2].set_title(f"3. RLDP + FGF (PSNR: {psnr_f:.2f} dB)" if psnr_f else "3. RLDP + FGF")

        ax[1, 0].imshow(t_rldp, cmap="inferno", vmin=0.1, vmax=1.0)
        ax[1, 0].set_title("4. Peta Transmisi RLDP (Tepi Kabur)")

        ax[1, 1].imshow(t_fgf, cmap="inferno", vmin=0.1, vmax=1.0)
        ax[1, 1].set_title("5. Peta Transmisi RLDP + FGF (Tepi Tajam)")

        selisih = np.mean(np.abs(hasil_rldp.astype(np.float64) - hasil_fgf.astype(np.float64)), axis=2)
        im = ax[1, 2].imshow(selisih, cmap="jet")
        ax[1, 2].set_title("6. Letak Kelemahan RLDP (Merah = Efek Halo)")
        plt.colorbar(im, ax=ax[1, 2], fraction=0.046, pad=0.04)

        for a in ax.flat:
            a.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(dir_out_cmp, f"{nama_base}_perbandingan.png"), dpi=150)
        plt.close(fig)
        print(f"[{i}/{len(daftar_img)}] Selesai: {nama_file}")

    df = pd.DataFrame(rekap)
    df.to_csv(os.path.join(dir_out_cmp, "hasil_metrik.csv"), index=False)
    print("\n=== TABEL HASIL UJI COBA ===")
    print(df.to_string(index=False))

    if img_gt is not None:
        rata_psnr_r = pd.to_numeric(df["PSNR_RLDP(dB)"], errors="coerce").mean()
        rata_psnr_f = pd.to_numeric(df["PSNR_FGF(dB)"], errors="coerce").mean()
        rata_w_r = df["Waktu_RLDP(s)"].mean()
        rata_w_f = df["Waktu_FGF(s)"].mean()
        print(f"\nRata-rata PSNR  : RLDP Murni = {rata_psnr_r:.2f} dB  |  RLDP + FGF = {rata_psnr_f:.2f} dB")
        print(f"Rata-rata Waktu : RLDP Murni = {rata_w_r:.4f} s   |  RLDP + FGF = {rata_w_f:.4f} s")


if __name__ == "__main__":
    main()