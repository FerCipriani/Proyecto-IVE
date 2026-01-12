"""
Script de prueba rápida
Verifica que todas las librerías estén instaladas correctamente
"""

print("🔍 Verificando instalación de librerías...\n")

librerías = [
    ("BERTopic", "bertopic"),
    ("Sentence Transformers", "sentence_transformers"),
    ("UMAP", "umap"),
    ("HDBSCAN", "hdbscan"),
    ("Scikit-learn", "sklearn"),
    ("Pandas", "pandas"),
    ("OpenPyXL", "openpyxl"),
    ("Plotly", "plotly"),
]

errores = []

for nombre, modulo in librerías:
    try:
        __import__(modulo)
        print(f"✅ {nombre:25} - OK")
    except ImportError as e:
        print(f"❌ {nombre:25} - FALTA")
        errores.append(nombre)

print("\n" + "="*60)

if errores:
    print(f"\n⚠️ FALTAN {len(errores)} LIBRERÍAS:")
    for lib in errores:
        print(f"  - {lib}")
    print("\n💡 Ejecuta: pip install -r requirements.txt")
else:
    print("\n🎉 ¡TODAS LAS LIBRERÍAS INSTALADAS CORRECTAMENTE!")
    print("\n✅ Puedes ejecutar:")
    print("   python analisis_bertopic_discursos.py")

print("="*60)
