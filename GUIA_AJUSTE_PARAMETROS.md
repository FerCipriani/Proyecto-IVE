# 🎛️ Guía de Ajuste de Parámetros BERTopic

Esta guía explica cómo ajustar los parámetros del análisis según los resultados obtenidos.

---

## 📊 Diagnóstico: ¿Qué problemas tengo?

### Problema 1: **Demasiados tópicos pequeños** (muchos tópicos con pocos documentos)

**Síntomas:**
- 15+ tópicos
- Muchos tópicos con solo 2-5 documentos
- Tópicos muy similares entre sí

**Solución:**

```python
# En create_bertopic_model()

# 1️⃣ Aumentar tamaño mínimo de clusters
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,     # ⬆️ Subir de 6 a 10
    min_samples=5,           # ⬆️ Subir de 3 a 5
    ...
)

# 2️⃣ Aumentar vecinos en UMAP
umap_model = UMAP(
    n_neighbors=15,          # ⬆️ Subir de 12 a 15
    ...
)

# 3️⃣ Reducir más agresivamente
# En analizar_documentos()
topic_model_reduced = topic_model.reduce_topics(
    docs_clean,
    nr_topics=6              # ⬇️ Bajar de 10 a 6-8
)
```

---

### Problema 2: **Muchos documentos sin tópico** (tópico -1)

**Síntomas:**
- >30% de documentos con tópico -1
- Documentos relevantes marcados como outliers

**Solución:**

```python
# En create_bertopic_model()

# 1️⃣ Clusters más pequeños
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=4,      # ⬇️ Bajar de 6 a 4
    min_samples=2,           # ⬇️ Bajar de 3 a 2
    ...
)

# 2️⃣ UMAP más flexible
umap_model = UMAP(
    n_neighbors=8,           # ⬇️ Bajar de 12 a 8
    min_dist=0.1,            # ⬆️ Aumentar de 0.0 a 0.1
    ...
)
```

---

### Problema 3: **Palabras poco interpretables** (palabras raras o sin sentido)

**Síntomas:**
- Palabras muy raras dominan los tópicos
- Palabras genéricas sin significado específico
- Fragmentos de palabras

**Solución:**

```python
# En create_bertopic_model()

# 1️⃣ Agregar más stopwords específicas del dominio
spanish_stopwords = [
    # ... stopwords base ...
    # Agregar palabras del contexto parlamentario:
    "señor", "señora", "señoría", "honorable",
    "diputado", "diputada", "presidente", "presidenta",
    "cámara", "congreso", "comisión", "proyecto",
    "artículo", "inciso", "ley", "norma",
]

# 2️⃣ Filtrar palabras más raras
vectorizer_model = CountVectorizer(
    stop_words=spanish_stopwords,
    ngram_range=(1, 2),
    min_df=3,                # ⬆️ Subir de 2 a 3 (palabra aparece en 3+ docs)
    max_df=0.6               # ⬇️ Bajar de 0.7 a 0.6 (máximo 60% docs)
)
```

---

### Problema 4: **Tópicos demasiado amplios** (poco específicos)

**Síntomas:**
- 3-5 tópicos gigantes
- Documentos muy diferentes en el mismo tópico
- Palabras muy genéricas

**Solución:**

```python
# En create_bertopic_model()

# 1️⃣ Clusters más pequeños
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=4,      # ⬇️ Bajar de 6 a 4
    ...
)

# 2️⃣ Menos reducción de tópicos
# En analizar_documentos()
topic_model_reduced = topic_model.reduce_topics(
    docs_clean,
    nr_topics=15             # ⬆️ Subir de 10 a 15
)

# 3️⃣ UMAP más detallado
umap_model = UMAP(
    n_neighbors=8,           # ⬇️ Bajar de 12 a 8
    n_components=10,         # ⬆️ Subir de 5 a 10
    ...
)
```

---

## 🎯 Configuraciones Recomendadas por Tamaño de Corpus

### Corpus muy pequeño (50-100 documentos)

```python
# UMAP
umap_model = UMAP(
    n_neighbors=8,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

# HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=4,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# Reducción final
nr_topics=6
```

### Corpus mediano (100-300 documentos)

```python
# UMAP
umap_model = UMAP(
    n_neighbors=12,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

# HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=6,
    min_samples=3,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# Reducción final
nr_topics=10
```

### Corpus grande (300+ documentos)

```python
# UMAP
umap_model = UMAP(
    n_neighbors=15,
    n_components=10,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

# HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# Reducción final
nr_topics=15
```

---

## 🔄 Proceso Iterativo Recomendado

### Iteración 1: Configuración inicial (conservadora)

```python
min_cluster_size=6
min_samples=3
n_neighbors=12
nr_topics=10
```

**→ Ejecutar y revisar resultados**

---

### Iteración 2: Ajustar según diagnóstico

**Si muchos outliers (-1):**
```python
min_cluster_size=4  # ⬇️
min_samples=2       # ⬇️
```

**Si muchos tópicos pequeños:**
```python
min_cluster_size=8  # ⬆️
nr_topics=6         # ⬇️
```

**→ Ejecutar y revisar resultados**

---

### Iteración 3: Refinamiento fino

**Ajustar stopwords + min_df según palabras observadas**

**→ Ejecutar y revisar resultados finales**

---

## 📝 Checklist de Calidad

Usa esta lista para evaluar si tus resultados son buenos:

- [ ] **Número de tópicos razonable** (5-12 para ~100 docs)
- [ ] **<30% outliers** (tópico -1)
- [ ] **Palabras clave interpretables** (tienen sentido en el dominio)
- [ ] **Documentos coherentes** dentro de cada tópico
- [ ] **Tópicos distintos** entre sí (no redundantes)
- [ ] **Balance** (no un tópico gigante y muchos mini-tópicos)

---

## 🎓 Parámetros Explicados

### HDBSCAN

| Parámetro | Qué hace | Cuándo subirlo | Cuándo bajarlo |
|-----------|----------|----------------|----------------|
| `min_cluster_size` | Mínimo de docs por tópico | Muchos tópicos pequeños | Muchos outliers |
| `min_samples` | Densidad mínima requerida | Ruido en tópicos | Muchos outliers |

### UMAP

| Parámetro | Qué hace | Cuándo subirlo | Cuándo bajarlo |
|-----------|----------|----------------|----------------|
| `n_neighbors` | Contexto local vs global | Tópicos muy granulares | Tópicos muy amplios |
| `n_components` | Dimensiones finales | Corpus grande (300+) | Corpus pequeño (<100) |
| `min_dist` | Separación entre puntos | Muchos outliers | Tópicos se mezclan |

### Vectorizer

| Parámetro | Qué hace | Cuándo subirlo | Cuándo bajarlo |
|-----------|----------|----------------|----------------|
| `min_df` | Frecuencia mínima | Palabras raras | Vocabulario muy limitado |
| `max_df` | Frecuencia máxima | Palabras demasiado comunes | Pierdes términos importantes |
| `ngram_range` | Longitud de frases | Frases importantes | Vocabulario explota |

---

## 💡 Tips Finales

1. **Cambiar UN parámetro a la vez** para entender su efecto
2. **Guardar cada versión** con nombre descriptivo (ej: `resultados_mincluster8.xlsx`)
3. **Documentar cambios** en un archivo de notas
4. **Comparar visualmente** los resultados entre iteraciones
5. **Validar manualmente** revisando documentos de cada tópico

---

¡Buena suerte con el ajuste! 🚀
