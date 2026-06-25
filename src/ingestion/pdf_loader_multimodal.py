import os
import fitz


MIN_CLUSTER_WIDTH = 80
MIN_CLUSTER_HEIGHT = 80


def _get_column(block, page_width: float) -> int:
    """Assign text block to left/right column for reading-order sorting."""
    x0 = block[0]
    return 0 if x0 < page_width / 2 else 1


def _find_caption(blocks, image_bbox, page_width: float) -> str:
    """
    Find caption text near image bbox using proximity detection.
    Looks for text blocks below or above the image within a threshold.
    """
    img_x0, img_y0, img_x1, img_y1 = image_bbox
    threshold = 60  # pixels

    candidates = []
    for block in blocks:
        if block[6] != 0:  # samo text blokovi
            continue
        bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
        text = block[4].strip()

        # ispod slike
        if abs(by0 - img_y1) < threshold and bx0 < img_x1 and bx1 > img_x0:
            candidates.append((abs(by0 - img_y1), text))
        # iznad slike
        elif abs(img_y0 - by1) < threshold and bx0 < img_x1 and bx1 > img_x0:
            candidates.append((abs(img_y0 - by1), text))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0])
    caption = candidates[0][1]

    if any(caption.lower().startswith(p) for p in ["figure", "fig.", "fig ", "table"]):
        return caption

    return ""


def _extract_vector_figures(page, blocks, page_width: float) -> list[dict]:
    """
    Detect vector graphics clusters on page using cluster_drawings().
    Returns list of image dicts compatible with raster image format.
    """
    try:
        clusters = page.cluster_drawings()
    except Exception:
        return []

    vector_images = []
    # Kept for compatibility with earlier overlap-filtering approach.
    existing_raster_bboxes = [
        fitz.Rect(img.get("bbox", [0, 0, 0, 0]))
        for img in []
    ]

    for i, rect in enumerate(clusters):
        width = rect.width
        height = rect.height

            # skip clusters that are too small
        caption = _find_caption(blocks, rect, page_width)

        # renderuj region stranice kao PNG
        try:
            mat = fitz.Matrix(2, 2)  # 2x zoom
            pix = page.get_pixmap(matrix=mat, clip=rect)
            img_bytes = pix.tobytes("png")
        except Exception:
            continue

        vector_images.append({
            "img_index": f"vec_{i}",
            "xref": None,
            "bbox": list(rect),
            "width": width,
            "height": height,
            "caption": caption,
            "is_vector": True,
            "img_bytes": img_bytes,
        })

    return vector_images


def load_pdf(pdf_path: str) -> list[dict]:
    """Load one PDF and return per-page text + image metadata.

    Output format is designed for downstream multimodal chunking:
    each page contains plain text and a list of image dictionaries.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        blocks = page.get_text("blocks")
        page_width = page.rect.width

        blocks_sorted = sorted(blocks, key=lambda b: (_get_column(b, page_width), b[1]))
        page_text = " ".join(
            block[4].strip()
            for block in blocks_sorted
            if block[6] == 0
        )

        # --- RASTER slike ---
        images = []
        raster_bboxes = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue

            bbox = rects[0]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width < 50 or height < 50:
                continue

            caption = _find_caption(blocks, bbox, page_width)
            raster_bboxes.append(fitz.Rect(bbox))

            images.append({
                "img_index": img_index,
                "xref": xref,
                "bbox": list(bbox),
                "width": width,
                "height": height,
                "caption": caption,
                "is_vector": False,
            })

        # --- VECTOR slike ---
        vector_images = _extract_vector_figures(page, blocks, page_width)

        # izbaci vector clustere koji se preklapaju sa raster slikama
        for vec_img in vector_images:
            vec_rect = fitz.Rect(vec_img["bbox"])
            overlaps = any(
                vec_rect.intersects(r) for r in raster_bboxes
            )
            if not overlaps:
                images.append(vec_img)

        pages.append({
            "page_num": page.number,
            "text": page_text,
            "images": images,
        })

    doc.close()
    return pages


def extract_images_to_disk(pdf_path: str, pages: list[dict], output_dir: str) -> list[dict]:
    """
    Save extracted images to disk and add image_path to each image dict.
    Handles both raster (xref) and vector (rendered PNG bytes) images.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    for page_data in pages:
        for img in page_data["images"]:
            try:
                if img.get("is_vector"):
                    # vector — already have PNG bytes
                    filename = f"page{page_data['page_num']}_vec{img['img_index']}.png"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(img["img_bytes"])
                    img["image_path"] = filepath
                    del img["img_bytes"]  # ukloni bytes iz memorije
                else:
                    # raster — ekstraktuj iz PDF
                    xref = img["xref"]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    ext = base_image["ext"]
                    filename = f"page{page_data['page_num']}_img{img['img_index']}.{ext}"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)
                    img["image_path"] = filepath

            except Exception as e:
                # Missing path is tolerated; downstream chunking can still proceed.
                img["image_path"] = None

    doc.close()
    return pages