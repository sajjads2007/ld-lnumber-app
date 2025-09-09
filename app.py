"""
L-Number Lookup — Brave Search Engine Mode
------------------------------------------
Version: 2.2.1 (backend-only secret, inline thumbnails)

- NEVER calls londondrugs.com directly (no scraping, no mirrors).
- Uses Brave Search API (web + image) to find a best-match title + thumbnail per L-number.
- Exports Excel, CSV, and a PDF with embedded thumbnails.
- Shows thumbnails inline in the HTML table via data URIs (no hotlinking).
- API key is read ONLY from backend (Streamlit secrets or environment variables).
  * env var: BRAVE_SEARCH_KEY
  * optional env var: BRAVE_SEARCH_ENDPOINT (default: https://api.search.brave.com)
"""

import io
import os
import re
import html
import base64
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors

try:
    import pdfplumber
except Exception:
    pdfplumber = None

import httpx

# -------------------- App config --------------------
APP_VERSION = "2.2.1"

st.set_page_config(page_title="L-Number Lookup (Brave Search Mode)", layout="wide")
st.markdown(
    f"""
    <style>
      table, th, td {{ border: 1px solid #ddd !important; }}
      th, td {{ padding: 8px !important; }}
      th {{ background: #f7f7f7 !important; }}
      .version-tag {{ text-align:right;color:#999;font-size:0.9rem;margin-bottom:-1rem; }}
      code {{ white-space: pre-wrap; }}
    </style>
    <div class="version-tag">App version: {APP_VERSION}</div>
    """,
    unsafe_allow_html=True,
)
st.title("L-Number Lookup — Brave Search Mode")
st.caption("No contact with the source website. Uses Brave Search results (titles + thumbnails).")

# -------------------- Secret loading (backend only) --------------------
def _get_secret(name: str, default: str = "") -> str:
    if hasattr(st, "secrets"):
        try:
            v = st.secrets.get(name, "")
            if v:
                return str(v)
        except Exception:
            pass
    return os.getenv(name, default)

BRAVE_SEARCH_KEY = _get_secret("BRAVE_SEARCH_KEY").strip()
BRAVE_SEARCH_ENDPOINT = _get_secret("BRAVE_SEARCH_ENDPOINT", "https://api.search.brave.com").rstrip("/")

# -------------------- Regex helpers --------------------
LNUM_RX = re.compile(r"\bL\d{7}\b", re.I)

# -------------------- Sidebar (no secrets shown) --------------------
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload a PDF / Excel / CSV / TXT",
                                type=["pdf", "xls", "xlsx", "csv", "txt"])
    manual_input = st.text_area("Or paste L-numbers (optional)",
                                placeholder="L2161628, L2221406, ...")
    img_width = st.slider("Image width (px)", 80, 240, 120, 10)

    st.divider()
    st.subheader("Search settings")
    prefer_domain = st.text_input("Prefer results from domain (optional)", value="londondrugs.com")
    max_results = st.slider("Max results to try per query", 1, 10, 3, 1)

    run_btn = st.button("▶️ Run lookup")
    st.divider()
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

# -------------------- Utils --------------------
def extract_lnums(text: str) -> List[str]:
    raw = sorted(set(LNUM_RX.findall(text or "")))
    seen, out = set(), []
    for x in raw:
        x = x.strip().upper()
        digits = "".join(ch for ch in x[1:] if ch.isdigit()) if x.startswith("L") else ""
        if not digits:
            continue
        norm = "L" + digits.zfill(7)
        if norm not in seen:
            seen.add(norm); out.append(norm)
    return out

@st.cache_data(show_spinner=False)
def _cache_get(key: str) -> Optional[Dict]:
    return None

@st.cache_data(show_spinner=False)
def _cache_put(key: str, val: Dict) -> Dict:
    return val

def _score_domain(url: Optional[str], prefer: str) -> int:
    if not url or not prefer:
        return 0
    return 10 if prefer.lower() in url.lower() else 0

# -------------------- Brave API --------------------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

HTTPX_COMMON = dict(
    headers={"User-Agent": UA},
    follow_redirects=True,
    timeout=httpx.Timeout(12.0, read=12.0),
    http2=False,
)

WEB_ENDPOINT   = f"{BRAVE_SEARCH_ENDPOINT}/res/v1/web/search"
IMAGE_ENDPOINT = f"{BRAVE_SEARCH_ENDPOINT}/res/v1/images/search"

async def brave_web_search(client: httpx.AsyncClient, token: str, query: str, count: int) -> List[Dict]:
    if not token:
        return []
    headers = {"X-Subscription-Token": token, "Accept": "application/json"}
    params = {"q": query, "count": count, "country": "CA", "search_lang": "en", "safesearch": "moderate"}
    try:
        r = await client.get(WEB_ENDPOINT, params=params, headers=headers)
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        web = (data or {}).get("web", {})
        for item in web.get("results", []):
            results.append({
                "url": item.get("url"),
                "title": item.get("title") or "",
                "description": item.get("description") or "",
                "thumbnail": ((item.get("thumbnail") or {}).get("src")) if isinstance(item.get("thumbnail"), dict) else None,
            })
        return results
    except Exception:
        return []

async def brave_image_search(client: httpx.AsyncClient, token: str, query: str, count: int) -> List[Dict]:
    if not token:
        return []
    headers = {"X-Subscription-Token": token, "Accept": "application/json"}
    params = {"q": query, "count": count, "country": "CA", "search_lang": "en", "safesearch": "moderate"}
    try:
        r = await client.get(IMAGE_ENDPOINT, params=params, headers=headers)
        if r.status_code != 200:
            return []
        data = r.json()
        norm = []
        for it in (data or {}).get("results", []):
            thumb = it.get("thumbnail")
            thumb_src = thumb.get("src") if isinstance(thumb, dict) else None
            norm.append({
                "thumbnailUrl": thumb_src or it.get("url") or "",
                "hostPageUrl": it.get("page_url") or it.get("source", ""),
                "name": it.get("title") or "",
            })
        return norm
    except Exception:
        return []

async def resolve_one(client: httpx.AsyncClient, token: str, lnum: str, prefer: str, count: int) -> Dict[str, str]:
    queries = []
    if prefer:
        queries.append(f"site:{prefer} {lnum}")
    queries.append(f"{lnum} product")
    queries.append(lnum)

    best_title = "NOT FOUND"
    best_image = "NOT FOUND"     # from image search
    best_web_thumb = None        # from web results (fallback)
    best_url   = "—"

    for q in queries:
        web = await brave_web_search(client, token, q, count)
        img = await brave_image_search(client, token, q, count)

        if web:
            ranked = sorted(web, key=lambda w: _score_domain(w.get("url"), prefer), reverse=True)
            chosen = ranked[0]
            best_title = chosen.get("title") or chosen.get("description") or best_title
            best_url = chosen.get("url") or best_url
            # capture web thumbnail as a fallback
            if not best_web_thumb:
                best_web_thumb = chosen.get("thumbnail")

        if img and best_image == "NOT FOUND":
            img_sorted = sorted(
                img,
                key=lambda it: (1 if prefer and prefer.lower() in (it.get("hostPageUrl","").lower()) else 0),
                reverse=True
            )
            best_image = img_sorted[0].get("thumbnailUrl") or best_image

        if best_title != "NOT FOUND" and (best_image != "NOT FOUND" or best_web_thumb):
            break

    chosen_thumb = best_image if best_image != "NOT FOUND" else (best_web_thumb or "NOT FOUND")

    return {
        "L-Number": lnum,
        "Product Name": best_title,
        "Product Image": chosen_thumb,
        "Source URL": best_url,
    }

async def download_image(client: httpx.AsyncClient, url: str, max_w: int) -> Optional[bytes]:
    if not url or url == "NOT FOUND":
        return None
    try:
        r = await client.get(url)
        if r.status_code != 200 or not r.content:
            return None
        img = PILImage.open(io.BytesIO(r.content)).convert("RGB")
        if img.width > max_w:
            ratio = max_w / float(img.width)
            img = img.resize((max_w, int(img.height * ratio)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None

def build_pdf(rows: List[Dict[str, str]], img_width_px: int, images: Dict[str, Optional[bytes]]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=24, rightMargin=24, topMargin=36, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("L-Number Lookup (Brave Search Mode)", styles["Title"]))
    story.append(Spacer(1, 8))

    data = [["L-Number", "Thumbnail", "Title / Best Match"]]
    for r in rows:
        lnum = r["L-Number"]
        pname = r["Product Name"]
        pimg = r["Product Image"]
        cell_img = "NOT FOUND"
        if pimg != "NOT FOUND":
            blob = images.get(pimg)
            if blob:
                cell_img = RLImage(io.BytesIO(blob))
        data.append([lnum, cell_img, pname])

    table = Table(data, colWidths=[90, img_width_px + 12, 340])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    doc.build(story)
    return buf.getvalue()

def as_data_uri(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

# -------------------- Ingest input --------------------
lnums: List[str] = []
should_run = False

if uploaded is not None:
    should_run = True
if manual_input and manual_input.strip() and run_btn:
    should_run = True

if should_run:
    text = ""
    if uploaded is not None:
        fb = uploaded.read()
        ext = Path(uploaded.name).suffix.lower()
        try:
            if ext in {".xls", ".xlsx"}:
                df_in = pd.read_excel(io.BytesIO(fb), header=None)
                text = "\n".join(df_in.astype(str).fillna("").values.ravel().tolist())
            elif ext == ".csv":
                df_in = pd.read_csv(io.BytesIO(fb), header=None)
                text = "\n".join(df_in.astype(str).fillna("").values.ravel().tolist())
            elif ext == ".txt":
                text = fb.decode(errors="ignore")
            elif ext == ".pdf" and pdfplumber:
                with pdfplumber.open(io.BytesIO(fb)) as pdf:
                    for page in pdf.pages:
                        text += "\n" + (page.extract_text() or "")
            else:
                text = fb.decode(errors="ignore")
        except Exception:
            text = ""
    if manual_input and manual_input.strip():
        text += "\n" + manual_input.strip()

    lnums = extract_lnums(text)

st.subheader(f"Found {len(lnums)} L-numbers")
st.write(", ".join(lnums) or "—")

# -------------------- Run lookups --------------------
if lnums:
    if not BRAVE_SEARCH_KEY:
        st.error("Server is missing BRAVE_SEARCH_KEY. Add it as a Streamlit secret or environment variable on the backend.")
    else:
        st.info("Running in **Brave Search Mode**. This app does not contact the source site.")
        progress = st.progress(0.0)
        status = st.empty()
        status.write("Resolving via Brave Web + Image Search…")

        async def resolve_all(items: List[str]) -> List[Dict[str, str]]:
            sem = asyncio.Semaphore(8)
            async with httpx.AsyncClient(**HTTPX_COMMON) as client:
                async def worker(ln: str):
                    async with sem:
                        key = f"{ln}|{prefer_domain}|{max_results}"
                        cached = _cache_get(key)
                        if cached:
                            return cached
                        res = await resolve_one(client, BRAVE_SEARCH_KEY, ln, prefer_domain.strip(), max_results)
                        _cache_put(key, res)
                        return res
                tasks = [worker(x) for x in items]
                results: List[Dict[str, str]] = []
                done = 0
                for coro in asyncio.as_completed(tasks):
                    r = await coro
                    results.append(r)
                    done += 1
                    progress.progress(done / len(items))
                return results

        results = asyncio.run(resolve_all(lnums))
        results.sort(key=lambda r: r["L-Number"])

        df = pd.DataFrame([{
            "L-Number": r["L-Number"],
            "Product Name": r["Product Name"],
            "Product Image": r["Product Image"],
            "Source URL": r["Source URL"],
        } for r in results])

        status.write("Preparing downloads…")

        # Excel
        excel_buf = io.BytesIO()
        df.to_excel(excel_buf, index=False, engine="openpyxl")
        st.download_button("⬇️ Download results (Excel)", data=excel_buf.getvalue(),
                           file_name="lnumber_brave_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # CSV
        st.download_button("⬇️ Download results (CSV)", data=df.to_csv(index=False),
                           file_name="lnumber_brave_results.csv", mime="text/csv")

        # Download thumbnails (for PDF and inline HTML)
        status.write("Fetching thumbnails…")
        async def fetch_thumbs():
            urls = sorted(set(u for u in df["Product Image"].tolist() if u and u != "NOT FOUND"))
            blobs: Dict[str, Optional[bytes]] = {}
            sem = asyncio.Semaphore(8)
            async with httpx.AsyncClient(**HTTPX_COMMON) as client:
                async def dl(u: str):
                    async with sem:
                        blobs[u] = await download_image(client, u, img_width)
                await asyncio.gather(*[dl(u) for u in urls])
            return blobs

        blobs_map = asyncio.run(fetch_thumbs())

        # PDF
        status.write("Building PDF (with thumbnails)…")
        pdf_bytes = build_pdf(results, img_width, blobs_map)
        st.download_button("⬇️ Download results (PDF with images)", data=pdf_bytes,
                           file_name="lnumber_brave_results.pdf", mime="application/pdf")

        # HTML preview with INLINE images to avoid hotlinking failures
        def table_html(rows: List[Dict[str, str]], w: int, blobs: Dict[str, Optional[bytes]]) -> str:
            trs = []
            for r in rows:
                src = r["Product Image"]
                if src != "NOT FOUND" and blobs.get(src):
                    data_uri = as_data_uri(blobs[src])
                    img_html = f'<img src="{data_uri}" width="{w}">'
                elif src != "NOT FOUND":
                    # fallback to external (may still work for some hosts)
                    img_html = f'<img src="{html.escape(src)}" width="{w}">'
                else:
                    img_html = "<b>NOT FOUND</b>"
                link_html = (f'<a href="{html.escape(r["Source URL"])}" target="_blank">open</a>'
                             if r["Source URL"] not in (None, "", "—") else "—")
                trs.append(
                    f"<tr><td>{html.escape(r['L-Number'])}</td>"
                    f"<td>{img_html}</td>"
                    f"<td>{html.escape(r['Product Name'])}</td>"
                    f"<td>{link_html}</td></tr>"
                )
            return ("<table style='border-collapse:collapse;width:100%'>"
                    "<thead><tr>"
                    "<th>L-Number</th><th>Thumbnail</th><th>Title</th><th>Source</th>"
                    "</tr></thead><tbody>" + "".join(trs) + "</tbody></table>")

        st.markdown("### Results (inline thumbnails)")
        st.markdown(table_html(results, img_width, blobs_map), unsafe_allow_html=True)

        hits = sum(1 for r in results if (r["Product Name"] != "NOT FOUND") or (r["Product Image"] != "NOT FOUND"))
        st.success(f"Completed: {hits}/{len(results)} items returned at least a title or an image.")
else:
    st.info("Upload a file or paste L-numbers in the sidebar, then press **Run lookup**.")
