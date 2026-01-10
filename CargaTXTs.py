from pathlib import Path

def load_txt_dir(txt_dir: str, encoding="utf-8"):
    paths = sorted(Path(txt_dir).glob("*.txt"))
    docs = []
    ids = []
    for p in paths:
        text = p.read_text(encoding=encoding, errors="ignore").strip()
        if text:
            docs.append(text)
            ids.append(p.name)
    return ids, docs

# Cambiar por tu ruta
txt_dir = "IVEDip"
doc_ids, docs = load_txt_dir(txt_dir)

print(f"Docs cargados: {len(docs)}")
print("Ejemplo ID:", doc_ids[0])
print("Ejemplo texto (primeros 300 chars):", docs[0][:300])