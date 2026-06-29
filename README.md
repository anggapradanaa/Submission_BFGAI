# StudioAI: Image Generation with Stable Diffusion

Proyek ini terdiri dari dua notebook: satu untuk eksplorasi pipeline Stable Diffusion secara mendalam, dan satu untuk membangun antarmuka web interaktif menggunakan Streamlit.

---

## Deskripsi Proyek

StudioAI adalah aplikasi berbasis AI untuk membuat dan mengedit gambar menggunakan model **Stable Diffusion v1-5** dari RunwayML. Proyek ini mencakup fitur text-to-image, image-to-image refinement, inpainting, serta outpainting, semuanya dapat diakses melalui antarmuka web yang dibangun dengan Streamlit.

---

## Struktur File

```
.
├── Pipeline_submission_BFGAI_Angga_Yulian_Adi_Pradana.ipynb
└── Streamlit_submission_BFGAI_Angga_Yulian_Adi_Pradana.ipynb
```

### `Pipeline_submission_...ipynb`

Notebook ini berisi eksplorasi dan eksperimen pipeline Stable Diffusion secara menyeluruh, dibagi berdasarkan kriteria berikut:

**Kriteria 1: Text-to-Image Generation**

- Load model `runwayml/stable-diffusion-v1-5` dengan dukungan GPU (CUDA) maupun CPU.
- Penggunaan library `compel` untuk menangani prompt panjang di atas batas 77 token.
- Fungsi `generate_simple_image()` untuk generasi dasar dan `generate_advanced_image()` untuk generasi dengan hyperparameter kustom.
- **Guidance Scale Comparison**: Eksperimen dengan nilai 3, 5, 12, dan 15 untuk melihat pengaruh terhadap kesesuaian gaya gambar dengan prompt.
- **Inference Steps Comparison**: Eksperimen dengan 5, 15, 30, dan 50 steps, dengan kesimpulan bahwa 30 steps adalah titik optimal.
- **Batch Inference**: Menghasilkan 4 gambar sekaligus dari satu prompt.
- **Scheduler Comparison**: Perbandingan hasil antara `Euler A`, `DPM++`, dan `DDIM`.

**Kriteria 2: Image-to-Image (Refinement, Inpainting, Outpainting)**

- **Two-Stage Refinement**: Implementasi pola Base + Refiner menggunakan `StableDiffusionImg2ImgPipeline`. Menyertakan fallback otomatis karena SD v1-5 tidak mendukung parameter `denoising_end/start` secara native.
- **Inpainting (Manual Masking)**: Pembuatan mask manual menggunakan `ImageDraw` untuk mengedit area tertentu pada gambar.
- **Inpainting (Auto Segmentation)**: Pembuatan mask otomatis berbasis segmentasi warna/piksel untuk mengganti elemen gambar (contoh: mengganti pakaian astronaut menjadi tuksedo formal).
- **Outpainting (Directional)**: Memperluas kanvas gambar ke arah tertentu (kiri, kanan, atas, bawah) menggunakan background blur sebagai fill awal.
- **Outpainting (Zoom Out)**: Zoom out iteratif sebanyak 3 langkah, dengan teknik tile-based background fill untuk konsistensi warna tepi.

---

### `Streamlit_submission_...ipynb`

Notebook ini membangun dan menjalankan antarmuka web interaktif **StudioAI** menggunakan Streamlit, yang di-expose secara publik via `pyngrok`.

Aplikasi terdiri dari dua file utama yang ditulis langsung dari notebook:

**`logic.py`** (ditulis bertahap dalam tiga level):

| Level | Fitur yang ditambahkan |
|---|---|
| Basic | `load_models_cached()`, `generate_image()` versi dasar (single image) |
| Skilled | `flush_memory()`, `set_scheduler()`, `generate_image()` dengan dukungan batch dan pemilihan scheduler |
| Advanced | `run_inpainting()`, `prepare_outpainting()` dengan Gaussian blur background |

**`app.py`** (tidak perlu dimodifikasi, cukup dijalankan):

- Konfigurasi halaman Streamlit dengan layout wide.
- Sidebar berisi slider untuk `Quality Steps`, `CFG`, `Seed`, pilihan `Scheduler`, ukuran `Batch`, dan tombol flush memory.
- **Tab GENERATE**: Form input prompt dan negative prompt, kemudian menampilkan hasil gambar. Untuk batch lebih dari 1, gambar ditampilkan dalam grid 2 kolom dengan opsi memilih gambar untuk diedit.
- **Tab EDIT**: Dua mode edit yang dapat dipilih:
  - **Inpainting**: Canvas interaktif menggunakan `streamlit_drawable_canvas` untuk menggambar mask secara langsung di atas gambar.
  - **Outpainting**: Ekspansi kanvas ke segala arah (all-sides zoom out) menggunakan gambar yang sudah dipilih dari tab Generate.

---

## Dependensi

```
torch
diffusers
transformers
accelerate
xformers
Pillow
matplotlib
compel
streamlit
streamlit_drawable_canvas==0.8.0
pyngrok
huggingface_hub
```

---

## Model yang Digunakan

- `runwayml/stable-diffusion-v1-5` - model utama untuk text-to-image dan image-to-image
- `runwayml/stable-diffusion-inpainting` - model khusus untuk inpainting dan outpainting

Kedua model di-load dari Hugging Face dan membutuhkan koneksi internet pada saat pertama kali dijalankan. Disarankan menggunakan GPU (CUDA) untuk performa optimal.

---

## Cara Menjalankan

### Pipeline Notebook

Jalankan setiap sel secara berurutan dari atas ke bawah. Pastikan GPU tersedia (misalnya di Google Colab dengan runtime T4 atau lebih tinggi).

### Streamlit App

1. Jalankan sel instalasi dependensi.
2. Jalankan seluruh sel di blok `logic.py` (Basic, Skilled, Advanced) secara berurutan agar semua fungsi terdefinisi dengan benar.
3. Jalankan sel `app.py` untuk menulis file aplikasi.
4. Jalankan sel terakhir yang memulai Streamlit dan membuat tunnel ngrok. URL publik akan muncul di output sel tersebut.

---

## Catatan Teknis

- **Compel**: Digunakan untuk menangani prompt panjang yang melebihi batas 77 token pada Stable Diffusion. Prompt dikonversi menjadi `prompt_embeds` sebelum dikirim ke pipeline.
- **Scheduler**: Tiga pilihan scheduler tersedia: `Euler A` (hasil artistik/painterly), `DPM++` (flat cartoon, paling sesuai dengan prompt gaya vector art), dan `DDIM` (paling minimalis dan flat).
- **Memory Management**: Fungsi `flush_memory()` memanggil `gc.collect()` dan `torch.cuda.empty_cache()` untuk membebaskan VRAM antar proses generasi, penting terutama saat menjalankan batch atau beralih antara model txt2img dan inpainting.
- **SD v1-5 Limitation**: Model ini tidak mendukung `denoising_end`/`denoising_start` secara native (fitur SDXL). Implementasi two-stage refinement menggunakan fallback otomatis dengan parameter `strength=0.2` pada img2img pipeline untuk mensimulasikan efek yang setara.
