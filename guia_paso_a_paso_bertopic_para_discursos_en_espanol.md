# Guía paso a paso: BERTopic para discursos en español

Esta guía muestra **cómo aplicar BERTopic** a un conjunto de **discursos en español** almacenados como archivos `.txt` en un directorio.

Está pensada para:
- ~100 documentos
- textos no excesivamente largos
- sin metadatos
- comparación directa contra resultados obtenidos con MALLET

---

## 0) Preparación del entorno

## Crear Entorno Virtual

python -m venv venv

## Activar EV

venv\Scripts\Activate.ps1


## Actualizar PIP

pip install --upgrade pip


## Inicializar Git localmente

git init


## Crear archivo .gitignore

New-Item .gitignore

# Entorno virtual
venv/
env/
ENV/

# Archivos de Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Archivos de configuración
.env
*.log

# Archivos de IDE
.vscode/
.idea/

# Archivos de sistema
.DS_Store
Thumbs.db

##  Hacer primer commit

git add .
git commit -m "Initial commit"

## Crear repo en GitHub desde VS Code

Ve a la barra lateral izquierda → Source Control (ícono de bifurcación)
Click en "Publish to GitHub"
Elige:

Public o Private
Nombre del repositorio


VS Code se encarga de todo automáticamente ✅


### Instalación de dependencias

```bash
pip install -U bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas
```

Opcional (para visualizaciones interactivas):

```bash
pip install -U plotly
```

---

## 1) Cargar discursos desde un directorio

Cada archivo `.txt` se considera un documento.

```python
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
txt_dir = "RUTA/A/TU/DIRECTORIO"
doc_ids, docs = load_txt_dir(txt_dir)

print(f"Docs cargados: {len(docs)}")
print("Ejemplo ID:", doc_ids[0])
print("Ejemplo texto (primeros 300 chars):", docs[0][:300])
```

Si ves problemas de caracteres, probá `encoding="latin-1"`.

---

## 2) Preprocesado liviano (recomendado)

Para BERTopic **no conviene** un preprocesado agresivo. Los embeddings capturan semántica contextual.

Se recomienda solo:
- normalizar espacios
- eliminar URLs y emails

```python
import re

def clean_text_es(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

docs_clean = [clean_text_es(t) for t in docs]
```

---

## 3) Modelo de embeddings (español)

Se usa un modelo **Sentence-Transformer multilingüe**, robusto y liviano:

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)
```

---

## 4) Configuración de BERTopic (crítica para corpus chico)

Con ~100 documentos buscamos:
- estabilidad
- pocos tópicos bien definidos
- evitar micro-tópicos ruidosos

### UMAP (reducción dimensional)

```python
from umap import UMAP

umap_model = UMAP(
    n_neighbors=12,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)
```

### HDBSCAN (clustering)

```python
import hdbscan

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=6,
    min_samples=3,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)
```

### Vectorizer (c-TF-IDF)

Incluye:
- stopwords en español
- unigramas + bigramas
- filtros por frecuencia

```python
from sklearn.feature_extraction.text import CountVectorizer

spanish_stopwords = [
    "de","la","que","el","en","y","a","los","del","se","las","por","un",
    "para","con","no","una","su","al","lo","como","más","pero","sus","le",
    "ya","o","este","sí","porque","esta","entre","cuando","muy","sin","sobre",
    "también","me","hasta","hay","donde","quien","desde","todo","nos","durante",
    "todos","uno","les","ni","contra","otros","ese","eso","ante","ellos","e",
    "esto","mí","antes","algunos","qué","unos","yo","otro","otras","otra","él",
    "tanto","esa","estos","mucho","quienes","nada","muchos","cual","poco","ella",
    "estar","estas","algunas","algo",
]

vectorizer_model = CountVectorizer(
    stop_words=spanish_stopwords,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.7
)
```

### Crear el modelo BERTopic

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)
```

---

## 5) Entrenamiento

```python
topics, probs = topic_model.fit_transform(docs_clean)

print("Cantidad de tópicos (incluyendo -1):", len(set(topics)))
print("Outliers (-1):", sum(t == -1 for t in topics))
```

`-1` indica documentos no asignados (normal en corpus chicos).

---

## 6) Inspección de resultados

### Información general de tópicos

```python
topic_info = topic_model.get_topic_info()
print(topic_info.head(15))
```

### Palabras de un tópico

```python
print(topic_model.get_topic(0))
```

### Documentos representativos

```python
rep_docs = topic_model.get_representative_docs(0)
for i, d in enumerate(rep_docs[:3], 1):
    print(f"\n--- Doc representativo {i} ---\n{d[:800]}")
```

---

## 7) Reducción / fusión de tópicos (muy recomendado)

En corpus chicos casi siempre mejora la interpretabilidad.

```python
topic_model_reduced = topic_model.reduce_topics(
    docs_clean,
    nr_topics=10
)

topics_red, probs_red = topic_model_reduced.transform(docs_clean)

print(topic_model_reduced.get_topic_info().head(20))
```

---

## 8) Exportar resultados a CSV

```python
import pandas as pd

df = pd.DataFrame({
    "doc_id": doc_ids,
    "topic": topics_red,
    "text": docs_clean,
})

if probs_red is not None:
    df["topic_prob"] = probs_red.max(axis=1)

df.to_csv("bertopic_resultados.csv", index=False, encoding="utf-8")
print("Guardado: bertopic_resultados.csv")
```

---

## 9) Visualizaciones (opcional)

```python
fig = topic_model_reduced.visualize_topics()
fig.write_html("topics_map.html")

fig2 = topic_model_reduced.visualize_barchart(top_n_topics=10)
fig2.write_html("topics_barchart.html")

print("Visualizaciones guardadas")
```

---

## 10) Ajustes rápidos

### Demasiados tópicos pequeños
- subir `min_cluster_size` (8–12)
- subir `n_neighbors` (15)
- reducir tópicos (`nr_topics=6–8`)

### Muchos outliers (-1)
- bajar `min_cluster_size`
- bajar `min_samples`

### Palabras poco interpretables
- ampliar stopwords de dominio (retórica política)
- subir `min_df` a 3
- probar `ngram_range=(1,3)` con cuidado

---

## Nota final

Si BERTopic **no mejora claramente** a MALLET en este escenario, lo más probable es que el límite sea el **tamaño del corpus**, no el algoritmo.

En discursos, la mayor ganancia suele venir de **embeddings semánticos** bien configurados y **reducción controlada de tópicos**.

---

Fin del documento.

