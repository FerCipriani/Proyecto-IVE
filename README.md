# 🎯 Análisis BERTopic para Discursos Políticos

Script automatizado para análisis de tópicos en discursos parlamentarios usando BERTopic.

## 📋 Requisitos

- Python 3.8 o superior
- Archivos `.txt` con discursos en español
- ~100 documentos (optimizado para corpus pequeño)

## 🚀 Instalación

### 1. Clonar/descargar el proyecto

```bash
git clone https://github.com/FerCipriani/Proyecto-IVE.git
cd Proyecto-IVE
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno virtual

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si fallas por timeout, usa:
```bash
pip install --timeout=300 -r requirements.txt
```

## 📂 Estructura de Archivos

```
Proyecto-IVE/
│
├── IVEDip/                          # Directorio con archivos .txt
│   ├── AFIRMATIVO_discurso1.txt
│   ├── NEGATIVO_discurso2.txt
│   ├── ABSTENCION_discurso3.txt
│   └── ...
│
├── analisis_bertopic_discursos.py  # Script principal
├── requirements.txt                # Dependencias
└── README.md                       # Este archivo
```

## ▶️ Uso

### Ejecución básica

```bash
python analisis_bertopic_discursos.py
```

### Salida

El script genera un archivo Excel con timestamp:
```
bertopic_resultados_YYYYMMDD_HHMMSS.xlsx
```

### Hojas del Excel

1. **Resultados**: Cada documento con su tópico asignado y probabilidad
2. **Resumen_Topicos**: Estadísticas por tópico
3. **Votos_x_Topico**: Distribución de votos por tópico
4. **Info_Topicos_Detallada**: Información completa de cada tópico

## 🔧 Ajuste de Parámetros

### Cambiar directorio de entrada

En `analisis_bertopic_discursos.py`, línea 20:

```python
TXT_DIR = "IVEDip"  # Cambiar por tu directorio
```

### Ajustar clustering (si hay problemas)

En la función `create_bertopic_model()`:

#### 🔹 **Problema: Demasiados tópicos pequeños**

```python
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,     # ⬆️ Aumentar (default: 6)
    min_samples=5,           # ⬆️ Aumentar (default: 3)
    ...
)

umap_model = UMAP(
    n_neighbors=15,          # ⬆️ Aumentar (default: 12)
    ...
)
```

#### 🔹 **Problema: Muchos documentos sin tópico (-1)**

```python
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=4,      # ⬇️ Reducir (default: 6)
    min_samples=2,           # ⬇️ Reducir (default: 3)
    ...
)
```

#### 🔹 **Ajustar número de tópicos finales**

En la función `analizar_documentos()`, línea ~150:

```python
topic_model_reduced = topic_model.reduce_topics(
    docs_clean,
    nr_topics=10  # Cambiar según necesidad (5-15)
)
```

### Ajustar stopwords

En `create_bertopic_model()`, agregar palabras específicas del dominio:

```python
spanish_stopwords = [
    # ... stopwords existentes ...
    # Agregar palabras propias del contexto:
    "señor", "señora", "diputado", "diputada", "honorable",
]
```

## 📊 Interpretación de Resultados

### Columnas principales del Excel

| Columna | Descripción |
|---------|-------------|
| `Archivo` | Nombre del archivo .txt |
| `Tipo_Voto` | AFIRMATIVO / NEGATIVO / ABSTENCION |
| `Topico` | Número del tópico asignado (-1 = sin tópico) |
| `Probabilidad_Topico` | Confianza de la asignación (0-1) |
| `Palabras_Clave_Topico` | Top 10 palabras del tópico |
| `Resumen_Texto` | Primeros 200 caracteres |
| `Texto_Completo` | Texto completo del documento |

### Entendiendo los tópicos

- **Tópico -1**: Documentos "outliers" (no asignados)
- **Probabilidad alta** (>0.7): Asignación confiable
- **Probabilidad baja** (<0.3): Revisar manualmente

## 🎨 Visualizaciones (Opcional)

Descomentar al final de `analizar_documentos()`:

```python
# Generar visualizaciones HTML
fig = topic_model_reduced.visualize_topics()
fig.write_html("topics_map.html")

fig2 = topic_model_reduced.visualize_barchart(top_n_topics=10)
fig2.write_html("topics_barchart.html")
```

## ⚠️ Solución de Problemas

### Error: "No se encontraron documentos"

- Verificar que `TXT_DIR` apunte al directorio correcto
- Verificar que los archivos tengan extensión `.txt`

### Error: Timeout al instalar paquetes

```bash
pip install --timeout=300 -r requirements.txt
```

### Error: Memoria insuficiente

Reducir el tamaño del modelo de embeddings:

```python
embedding_model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"  # Modelo más ligero
)
```

### Resultados poco interpretables

1. **Revisar stopwords**: Agregar palabras frecuentes del dominio
2. **Aumentar min_df**: Filtrar palabras raras
3. **Reducir tópicos**: Usar `nr_topics` más bajo (6-8)

## 📝 Notas

- **Corpus pequeño** (~100 docs): Es normal tener algunos outliers (-1)
- **Primera ejecución**: Puede tardar 5-10 minutos descargando modelos
- **Ejecuciones posteriores**: Más rápidas (modelos en caché)

## 🤝 Contacto

Para dudas o mejoras:
- Abrir un issue en GitHub
- Contactar a [tu email]

---

**Versión**: 1.0  
**Última actualización**: Enero 2026
