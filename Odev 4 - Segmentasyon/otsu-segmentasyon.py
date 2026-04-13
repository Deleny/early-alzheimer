# Alzheimer MRI - Otsu Tabanlı Segmentasyon

# Bu script, Alzheimer MRI görüntülerine Otsu eşikleme yöntemi uygulayarak
# beyin bölgesini arka plandan ayırır ve beyin içi dokuları 3 sınıfa böler.
#
# Adımlar:
#   1) Veriyi yükle
#   2) Ön-işleme: Gri seviye → CLAHE → Gaussian Blur
#   3) Otsu ile beyin maskesi çıkar
#   4) Morfolojik temizleme (en büyük bileşen, kapama/açma)
#   5) Multi-Otsu ile 3 doku sınıfı (CSF, gri madde, beyaz madde)
#   6) Metrikleri hesapla ve kaydet
#   7) Görsel çıktıları üret


from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from skimage.filters import threshold_multiotsu
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Sabitler

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # early-alzheimer klasörü
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ---------- Ayarlar ----------
SPLIT = "train"              # Kullanılacak veri bölümü: "train" veya "test"
SAMPLES_PER_CLASS = 20       # Her sınıftan alınacak örnek sayısı
SEED = 42                    # Tekrarlanabilirlik için rastgele tohum

# Veri kümesindeki 4 sınıf
LABEL_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]



# Veri yükleme fonksiyonları

def load_project_dataset(split: str):
    """
    Proje klasöründeki lokal parquet dosyasından veriyi yükler.
    """
    local_file = DATA_DIR / (
        "train-00000-of-00001-c08a401c53fe5312.parquet"
        if split == "train"
        else "test-00000-of-00001-44110b9df98c5585.parquet"
    )

    if not local_file.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {local_file}\n"
            f"Lütfen data/ klasörüne parquet dosyalarını koyun."
        )

    dataset = load_dataset(
        "parquet",
        data_files={split: str(local_file)},
        split=split,
    )
    return dataset, f"local parquet: {local_file.name}"


def select_balanced_subset(dataset, samples_per_class: int):
    """Her sınıftan eşit sayıda örnek seçer (dengeli alt küme)."""
    selected_indices: dict[int, list[int]] = defaultdict(list)

    for index, label_id in enumerate(dataset["label"]):
        if len(selected_indices[label_id]) < samples_per_class:
            selected_indices[label_id].append(index)
        # Tüm sınıflar dolunca dur
        if all(
            len(selected_indices[cid]) >= samples_per_class
            for cid in range(len(LABEL_NAMES))
        ):
            break

    # Eksik sınıf kontrolü
    missing = [
        LABEL_NAMES[cid]
        for cid in range(len(LABEL_NAMES))
        if len(selected_indices[cid]) < samples_per_class
    ]
    if missing:
        raise ValueError(f"Dengeli alt küme oluşturulamadı. Eksik: {', '.join(missing)}")

    ordered = []
    for cid in range(len(LABEL_NAMES)):
        ordered.extend(selected_indices[cid])
    return dataset.select(ordered)



# Ön-işleme

def preprocess_image(image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Görüntüyü gri seviyeye çevir
    2. CLAHE ile kontrastı artır
    3. Gaussian Blur ile gürültüyü azalt
    """
    # Parquet'ten gelen görüntü formatını PIL Image'a çevir
    if isinstance(image, dict):
        image = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    grayscale = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)

    return grayscale, clahe, blurred

# Segmentasyon fonksiyonları
def segment_brain_otsu(preprocessed: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Otsu eşikleme ile beyin maskesi oluşturur.
    - Otsu otomatik eşik değeri belirler
    - En büyük bağlı bileşeni korur (küçük parçaları atar)
    - Morfolojik kapama/açma ile maskeyi temizler
    """
    # Otsu thresholding: otomatik eşik değeri hesapla
    otsu_threshold, binary_mask = cv2.threshold(
        preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # En büyük bağlı bileşeni bul ve sadece onu koru
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    if num_components > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Morfolojik temizleme
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary_mask, float(otsu_threshold)


def segment_tissues(
    preprocessed: np.ndarray, brain_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Beyin maskesi içindeki pikselleri Multi-Otsu ile 3 doku sınıfına ayırır:
      Sınıf 0 → CSF-benzeri (koyu bölgeler)
      Sınıf 1 → Gri madde-benzeri (orta yoğunluk)
      Sınıf 2 → Beyaz madde-benzeri (parlak bölgeler)
    """
    brain_pixels = preprocessed[brain_mask > 0]
    thresholds = threshold_multiotsu(brain_pixels, classes=3)
    tissue_map = np.digitize(preprocessed, bins=thresholds)
    return tissue_map, thresholds.astype(float)



# Metrik hesaplama

def compute_metrics(
    preprocessed: np.ndarray,
    brain_mask: np.ndarray,
    tissue_map: np.ndarray,
    tissue_thresholds: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Segmentasyon kalitesini ölçen nicel metrikler:
    - brain_area_ratio:     Beyin alanı / toplam alan
    - silhouette_score:     Doku kümelerinin ne kadar iyi ayrıldığı (-1 ile 1 arası)
    - edge_alignment_score: Segment sınırlarının güçlü gradyanlarla hizalanması
    - connected_components: İdeal = 1 (tek parça maske)
    - CSF / gri / beyaz madde oranları
    """
    inside = brain_mask > 0
    if not inside.any():
        raise ValueError("Boş beyin maskesi üretildi.")

    # --- Silhouette skoru ---
    region_pixels = preprocessed[inside].reshape(-1, 1)
    region_labels = tissue_map[inside].reshape(-1)
    sample_size = min(1200, len(region_pixels))
    sample_idx = rng.choice(len(region_pixels), sample_size, replace=False)
    silhouette = silhouette_score(
        region_pixels[sample_idx], region_labels[sample_idx]
    )

    # --- Kenar uyum skoru ---
    gx = cv2.Sobel(preprocessed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(preprocessed, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    eroded = cv2.erode(brain_mask, np.ones((3, 3), np.uint8), iterations=1)
    boundary = (brain_mask > 0) & ~(eroded > 0)
    edge_alignment = float(
        grad_mag[boundary].mean() / (grad_mag.mean() + 1e-6)
    )

    # --- Bağlı bileşen sayısı ---
    n_components = cv2.connectedComponents(brain_mask, 8)[0] - 1

    # --- Doku oranları ---
    ratios = []
    for cls in range(3):
        ratios.append(float(((tissue_map == cls) & inside).sum() / inside.sum()))

    return {
        "brain_area_ratio": float(inside.mean()),
        "csf_like_ratio": ratios[0],
        "gray_matter_like_ratio": ratios[1],
        "white_matter_like_ratio": ratios[2],
        "silhouette_score": float(silhouette),
        "edge_alignment_score": edge_alignment,
        "connected_components": int(n_components),
        "tissue_threshold_1": float(tissue_thresholds[0]),
        "tissue_threshold_2": float(tissue_thresholds[1]),
        "mean_intensity": float(preprocessed.mean()),
    }



# Görselleştirme

def build_overlay(tissue_map: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Doku haritasını renkli overlay olarak çizer."""
    colors = np.array(
        [
            [59, 130, 246],   # Mavi  → CSF-benzeri
            [34, 197, 94],    # Yeşil → Gri madde-benzeri
            [249, 115, 22],   # Turuncu → Beyaz madde-benzeri
        ],
        dtype=np.uint8,
    )
    overlay = np.zeros((*tissue_map.shape, 3), dtype=np.uint8)
    inside = brain_mask > 0
    overlay[inside] = colors[tissue_map[inside]]
    return overlay


def save_example_panel(record: dict, output_path: Path) -> None:
    """Her sınıf için 5 sütunlu örnek panel kaydeder."""
    gray = record["grayscale"]
    clahe = record["clahe"]
    mask = record["brain_mask"]
    overlay = record["overlay"]
    otsu_t = record["otsu_threshold"]
    tissue_t = record["tissue_thresholds"]
    label = record["label_name"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(f"Orijinal\n{label}")
    axes[0].axis("off")

    axes[1].imshow(clahe, cmap="gray")
    axes[1].set_title("CLAHE + Blur")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Beyin Maskesi\nOtsu={otsu_t:.1f}")
    axes[2].axis("off")

    axes[3].imshow(gray, cmap="gray")
    axes[3].imshow(overlay, alpha=0.45)
    axes[3].set_title("3 Seviyeli Doku Ayrımı")
    axes[3].axis("off")

    axes[4].hist(clahe.ravel(), bins=48, color="#4b5563", alpha=0.85)
    axes[4].axvline(otsu_t, color="#ef4444", linestyle="--", linewidth=2, label="Otsu")
    axes[4].axvline(tissue_t[0], color="#2563eb", linestyle=":", linewidth=2, label="T1")
    axes[4].axvline(tissue_t[1], color="#f97316", linestyle=":", linewidth=2, label="T2")
    axes[4].set_title("Histogram")
    axes[4].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """4 metrik için sınıflar arası karşılaştırma bar grafiği."""
    metrics = [
        ("brain_area_ratio", "Beyin Alan Oranı"),
        ("csf_like_ratio", "CSF-benzeri Alan Oranı"),
        ("silhouette_score", "Silhouette Skoru"),
        ("edge_alignment_score", "Kenar Uyum Skoru"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metrics):
        ax.bar(summary_df["label_name"], summary_df[col], color="#2563eb")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Ana fonksiyon

def main() -> int:
    rng = np.random.default_rng(SEED)

    # Çıktı klasörlerini oluştur
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples_dir = OUTPUT_DIR / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Veriyi yükle ve dengeli alt küme seç
    print("Veri yükleniyor...")
    dataset, data_source = load_project_dataset(SPLIT)
    subset = select_balanced_subset(dataset, SAMPLES_PER_CLASS)
    print(f"Veri kaynağı: {data_source}")
    print(f"Toplam örnek: {len(subset)}")

    # Her görüntüyü işle
    metrics_rows = []
    example_records: dict[str, dict] = {}

    for idx, sample in enumerate(subset):
        label_id = int(sample["label"])
        label_name = LABEL_NAMES[label_id]

        print(f"  [{idx + 1}/{len(subset)}] {label_name} işleniyor...")

        # Ön-işleme
        grayscale, clahe, blurred = preprocess_image(sample["image"])

        # Otsu ile beyin maskesi
        brain_mask, otsu_threshold = segment_brain_otsu(blurred)

        # Multi-Otsu ile doku sınıflandırması
        tissue_map, tissue_thresholds = segment_tissues(blurred, brain_mask)

        # Renkli overlay
        overlay = build_overlay(tissue_map, brain_mask)

        # Metrik hesapla
        row = compute_metrics(blurred, brain_mask, tissue_map, tissue_thresholds, rng)
        row.update({
            "dataset_index": idx,
            "label_id": label_id,
            "label_name": label_name,
            "otsu_threshold": otsu_threshold,
        })
        metrics_rows.append(row)

        # Her sınıftan ilk örneği kaydet (görsel panel için)
        if label_name not in example_records:
            example_records[label_name] = {
                "grayscale": grayscale,
                "clahe": blurred,
                "brain_mask": brain_mask,
                "overlay": overlay,
                "otsu_threshold": otsu_threshold,
                "tissue_thresholds": tissue_thresholds,
                "label_name": label_name,
            }

    # Metrikleri kaydet
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # Sınıf bazlı özet
    summary_df = (
        metrics_df.groupby("label_name", as_index=False)
        .agg(
            sample_count=("dataset_index", "count"),
            brain_area_ratio=("brain_area_ratio", "mean"),
            csf_like_ratio=("csf_like_ratio", "mean"),
            gray_matter_like_ratio=("gray_matter_like_ratio", "mean"),
            white_matter_like_ratio=("white_matter_like_ratio", "mean"),
            silhouette_score=("silhouette_score", "mean"),
            edge_alignment_score=("edge_alignment_score", "mean"),
            connected_components=("connected_components", "mean"),
            otsu_threshold=("otsu_threshold", "mean"),
            tissue_threshold_1=("tissue_threshold_1", "mean"),
            tissue_threshold_2=("tissue_threshold_2", "mean"),
        )
        .sort_values("label_name")
    )
    summary_path = OUTPUT_DIR / "class_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Görsel çıktılar
    print("\nÖrnek paneller kaydediliyor...")
    for label_name, record in example_records.items():
        save_example_panel(record, examples_dir / f"{label_name}.png")

    print("Özet grafik kaydediliyor...")
    save_summary_plot(summary_df, OUTPUT_DIR / "summary_metrics.png")



    # Sonuçları ekrana yazdır
    print(f"\nDetay metrikleri: {metrics_path}")
    print(f"Sınıf özeti:      {summary_path}")
    print()
    print(summary_df.round(4).to_string(index=False))
    print("\nTamamlandı!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
