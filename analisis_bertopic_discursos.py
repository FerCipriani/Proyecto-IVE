"""
Análisis BERTopic para Discursos Políticos
Procesa archivos .txt y genera Excel con tópicos asignados
"""

import re
from pathlib import Path
import pandas as pd
from datetime import datetime

# BERTopic y componentes
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Directorio con los documentos
TXT_DIR = "IVEDip"

# Archivo de salida
OUTPUT_FILE = f"bertopic_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

# ============================================================================
# 1) CARGAR DOCUMENTOS
# ============================================================================

def extraer_tipo_voto(filename):
    """
    Extrae el tipo de voto del nombre del archivo
    Returns: ABSTENCION, AFIRMATIVO, NEGATIVO, o DESCONOCIDO
    """
    filename_upper = filename.upper()
    
    if "ABSTENCION" in filename_upper:
        return "ABSTENCION"
    elif "AFIRMATIVO" in filename_upper:
        return "AFIRMATIVO"
    elif "NEGATIVO" in filename_upper:
        return "NEGATIVO"
    else:
        return "DESCONOCIDO"


def load_txt_dir(txt_dir, encoding="utf-8"):
    """
    Carga todos los archivos .txt de un directorio
    Returns: doc_ids, docs, tipos_voto
    """
    paths = sorted(Path(txt_dir).glob("*.txt"))
    docs = []
    doc_ids = []
    tipos_voto = []
    
    print(f"📂 Buscando archivos .txt en: {txt_dir}")
    
    for p in paths:
        try:
            text = p.read_text(encoding=encoding, errors="ignore").strip()
            if text:
                docs.append(text)
                doc_ids.append(p.name)
                tipos_voto.append(extraer_tipo_voto(p.name))
                
        except Exception as e:
            print(f"⚠️ Error al leer {p.name}: {e}")
    
    print(f"✅ Documentos cargados: {len(docs)}")
    
    # Mostrar estadísticas por tipo de voto
    from collections import Counter
    conteo = Counter(tipos_voto)
    print("\n📊 Distribución por tipo de voto:")
    for tipo, cantidad in conteo.items():
        print(f"  - {tipo}: {cantidad}")
    
    return doc_ids, docs, tipos_voto


# ============================================================================
# 2) PREPROCESADO LIVIANO
# ============================================================================

def clean_text_es(text):
    """
    Limpieza suave para preservar semántica
    """
    # Normalizar espacios
    text = text.replace("\u00a0", " ")
    
    # Eliminar URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # Eliminar emails
    text = re.sub(r"\S+@\S+", " ", text)
    
    # Normalizar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ============================================================================
# 3) CONFIGURACIÓN DE BERTOPIC
# ============================================================================

def create_bertopic_model():
    """
    Crea el modelo BERTopic configurado para corpus pequeño en español
    """
    
    print("\n🔧 Configurando modelo BERTopic...")
    
    # --- Modelo de embeddings multilingüe ---
    print("  📥 Cargando modelo de embeddings...")
    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # --- UMAP: Reducción dimensional ---
    umap_model = UMAP(
        n_neighbors=12,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    
    # --- HDBSCAN: Clustering ---
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=6,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    
    # --- Stopwords en español ---
    spanish_stopwords = [
        "de","la","que","el","en","y","a","los","del","se","las","por","un",
        "para","con","no","una","su","al","lo","como","más","pero","sus","le",
        "ya","o","este","sí","porque","esta","entre","cuando","muy","sin","sobre",
        "también","me","hasta","hay","donde","quien","desde","todo","nos","durante",
        "todos","uno","les","ni","contra","otros","ese","eso","ante","ellos","e",
        "esto","mí","antes","algunos","qué","unos","yo","otro","otras","otra","él",
        "tanto","esa","estos","mucho","quienes","nada","muchos","cual","poco","ella",
        "estar","estas","algunas","algo","ser","ha","sido","puede","pueden","han",
        "hacer","tiene","tienen","debe","deben","hoy","ahora","aquí","allí",
    ]
    
    # --- Vectorizer: c-TF-IDF ---
    vectorizer_model = CountVectorizer(
        stop_words=spanish_stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.7
    )
    
    # --- Crear modelo BERTopic ---
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True
    )
    
    print("✅ Modelo configurado correctamente")
    
    return topic_model


# ============================================================================
# 4) ENTRENAMIENTO Y ANÁLISIS
# ============================================================================

def analizar_documentos(doc_ids, docs, tipos_voto):
    """
    Ejecuta el análisis completo de BERTopic
    """
    
    # Preprocesar textos
    print("\n🧹 Limpiando textos...")
    docs_clean = [clean_text_es(t) for t in docs]
    print(f"✅ {len(docs_clean)} documentos preprocesados")
    
    # Crear modelo
    topic_model = create_bertopic_model()
    
    # Entrenar
    print("\n🎯 Entrenando modelo BERTopic...")
    topics, probs = topic_model.fit_transform(docs_clean)
    
    print(f"\n📊 Resultados iniciales:")
    print(f"  - Tópicos encontrados: {len(set(topics)) - (1 if -1 in topics else 0)}")
    print(f"  - Outliers (tópico -1): {sum(t == -1 for t in topics)}")
    
    # Mostrar información de tópicos
    print("\n📋 Información de tópicos:")
    topic_info = topic_model.get_topic_info()
    print(topic_info.head(15).to_string())
    
    # Reducir tópicos (recomendado para corpus pequeño)
    print("\n🔄 Reduciendo tópicos para mejorar interpretabilidad...")
    topic_model_reduced = topic_model.reduce_topics(
        docs_clean,
        nr_topics=10  # Ajustar según necesidad
    )
    
    topics_red, probs_red = topic_model_reduced.transform(docs_clean)
    
    print(f"\n📊 Después de reducción:")
    print(f"  - Tópicos finales: {len(set(topics_red)) - (1 if -1 in topics_red else 0)}")
    print(f"  - Outliers: {sum(t == -1 for t in topics_red)}")
    
    # Información reducida
    print("\n📋 Tópicos finales:")
    topic_info_red = topic_model_reduced.get_topic_info()
    print(topic_info_red.to_string())
    
    return topic_model_reduced, topics_red, probs_red, docs_clean


# ============================================================================
# 5) EXPORTAR A EXCEL
# ============================================================================

def exportar_a_excel(doc_ids, topics, probs, docs_clean, tipos_voto, 
                     topic_model, output_file):
    """
    Exporta resultados a Excel con formato profesional
    """
    
    print(f"\n💾 Exportando resultados a Excel...")
    
    # Crear DataFrame principal
    df = pd.DataFrame({
        "Archivo": doc_ids,
        "Tipo_Voto": tipos_voto,
        "Topico": topics,
        "Texto_Completo": docs_clean,
    })
    
    # Agregar probabilidad del tópico asignado
    if probs is not None:
        df["Probabilidad_Topico"] = probs.max(axis=1)
        df["Probabilidad_Topico"] = df["Probabilidad_Topico"].round(4)
    
    # Agregar palabras clave del tópico
    df["Palabras_Clave_Topico"] = df["Topico"].apply(
        lambda x: ", ".join([word for word, _ in topic_model.get_topic(x)[:10]]) 
        if x != -1 else "Sin tópico asignado"
    )
    
    # Crear resumen por texto (primeros 200 caracteres)
    df["Resumen_Texto"] = df["Texto_Completo"].str[:200] + "..."
    
    # Reordenar columnas
    df = df[[
        "Archivo",
        "Tipo_Voto", 
        "Topico",
        "Probabilidad_Topico",
        "Palabras_Clave_Topico",
        "Resumen_Texto",
        "Texto_Completo"
    ]]
    
    # Crear Excel con múltiples hojas
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Hoja 1: Resultados principales
        df.to_excel(writer, sheet_name='Resultados', index=False)
        
        # Hoja 2: Resumen por tópico
        topic_summary = df.groupby('Topico').agg({
            'Archivo': 'count',
            'Probabilidad_Topico': 'mean',
            'Tipo_Voto': lambda x: x.value_counts().to_dict()
        }).reset_index()
        topic_summary.columns = ['Topico', 'Num_Documentos', 'Prob_Promedio', 'Distribucion_Votos']
        
        # Agregar palabras clave
        topic_summary["Palabras_Clave"] = topic_summary["Topico"].apply(
            lambda x: ", ".join([word for word, _ in topic_model.get_topic(x)[:15]])
            if x != -1 else "Sin tópico"
        )
        
        topic_summary.to_excel(writer, sheet_name='Resumen_Topicos', index=False)
        
        # Hoja 3: Distribución por tipo de voto
        voto_summary = df.groupby(['Tipo_Voto', 'Topico']).size().reset_index(name='Cantidad')
        voto_summary.to_excel(writer, sheet_name='Votos_x_Topico', index=False)
        
        # Hoja 4: Información detallada de tópicos
        topic_info = topic_model.get_topic_info()
        topic_info.to_excel(writer, sheet_name='Info_Topicos_Detallada', index=False)
    
    print(f"✅ Archivo guardado: {output_file}")
    print(f"\n📊 Hojas incluidas en el Excel:")
    print(f"  1. Resultados: Documento por documento con tópico asignado")
    print(f"  2. Resumen_Topicos: Estadísticas por tópico")
    print(f"  3. Votos_x_Topico: Relación entre tipo de voto y tópicos")
    print(f"  4. Info_Topicos_Detallada: Información completa de cada tópico")


# ============================================================================
# 6) FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta el análisis completo
    """
    
    print("="*70)
    print("🎯 ANÁLISIS BERTOPIC - DISCURSOS POLÍTICOS")
    print("="*70)
    
    # 1. Cargar documentos
    doc_ids, docs, tipos_voto = load_txt_dir(TXT_DIR)
    
    if len(docs) == 0:
        print("❌ No se encontraron documentos. Verifica el directorio.")
        return
    
    # Mostrar ejemplo
    print(f"\n📄 Ejemplo de documento:")
    print(f"  ID: {doc_ids[0]}")
    print(f"  Tipo: {tipos_voto[0]}")
    print(f"  Texto (primeros 300 chars):")
    print(f"  {docs[0][:300]}...")
    
    # 2. Analizar
    topic_model, topics, probs, docs_clean = analizar_documentos(
        doc_ids, docs, tipos_voto
    )
    
    # 3. Exportar
    exportar_a_excel(
        doc_ids, topics, probs, docs_clean, tipos_voto,
        topic_model, OUTPUT_FILE
    )
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\n📁 Archivo de salida: {OUTPUT_FILE}")
    print("\n💡 Próximos pasos:")
    print("  1. Revisar el Excel generado")
    print("  2. Analizar la coherencia de los tópicos")
    print("  3. Ajustar parámetros si es necesario")
    print("\n🔧 Parámetros ajustables en create_bertopic_model():")
    print("  - min_cluster_size: controla tamaño mínimo de tópicos")
    print("  - n_neighbors: afecta la formación de clusters")
    print("  - nr_topics: número de tópicos finales después de reducción")


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
