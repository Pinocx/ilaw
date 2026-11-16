"""Shared fixtures and test data for AI segmentation tests."""

from pathlib import Path
from typing import List

# Root paths
TEST_ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_ROOT / "fixtures"

# Ensure fixtures directory exists
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def load_fixture(fixture_name: str) -> str:
    """Load a fixture file and return its content as text."""
    path = FIXTURES_DIR / fixture_name
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Fixture not found: {path}")


# Peruvian law with numbered articles
PERUVIAN_LAW_TEXT = """
LEY N° 29973
LEY QUE PROTEGE LOS DATOS PERSONALES

Artículo 1.- Objeto de la ley
La presente ley tiene por objeto proteger el derecho al intimidad y los datos personales.

Artículo 2.- Principio de extraterritorialidad
La presente ley se aplica a los tratamientos de datos personales efectuados en todo el territorio de la República del Perú.

Artículo 3.- Principio de consentimiento
Todo tratamiento de datos personales requiere el consentimiento del titular.

Artículo 4.- Principio de finalidad
Los datos personales deben ser recolectados con fines específicos.

Artículo Único.- Disposiciones finales
Esta ley entra en vigor al día siguiente de su publicación.
"""

# Regulation with multiple hierarchy levels
REGULATION_TEXT = """
DECRETO SUPREMO N° 014-2019-JUS
REGLAMENTO DE LA LEY DE PROTECCION DE DATOS PERSONALES

TÍTULO I
DISPOSICIONES GENERALES

CAPÍTULO I
Del objeto y alcance

Artículo 1.- Objeto
El presente reglamento desarrolla...

Artículo 2.- Alcance
Las disposiciones...

CAPÍTULO II
De las autoridades competentes

Artículo 3.- Autoridad de protección
La Autoridad Nacional de Protección de Datos Personales...

Artículo 4.- Funciones
Las funciones son...

TÍTULO II
Del tratamiento de datos personales

Artículo 5.- Principios
El tratamiento se rige por...

Artículo 6.- Consentimiento
El consentimiento debe ser...

DISPOSICIONES TRANSITORIAS
Disposición transitoria primera
Esta disposición aplica...
"""

# Legal textbook with table of contents
LEGAL_TEXTBOOK_TEXT = """
CODIGO PENAL COMENTADO - VOLUMEN I

COMENTARISTAS DE ESTE TOMO
José Luis Castillo Alva
José Urquizo Dlaeehea Dino

PRIMERA EDICIÓN SETIEMBRE 2004

TABLA DE CONTENIDOS

Art. 1  Principio de territorialidad ........ 38
Art. 2  Principio de extraterritorialidad ........ 45
Art. 3  Principio de protección ........ 52

CAPÍTULO I
DISPOSICIONES PRELIMINARES

Artículo 1.- Principio de territorialidad
Todo hecho cometido en territorio peruano es delito.

Artículo 2.- Principio de extraterritorialidad
Se castigan delitos cometidos en el extranjero cuando...

COMENTARIO SOBRE EL ARTÍCULO 2
El autor comenta que el artículo 2 establece la aplicación del Código Penal...
Según la doctrina citada, el artículo 2 tiene alcance limitado.
"""

# Legal commentary discussing several articles
LEGAL_COMMENTARY_TEXT = """
COMENTARIO AL CÓDIGO PENAL PERUANO
Por Dr. Juan Pérez López

INTRODUCCIÓN

El Artículo 1 del Código Penal establece el principio de territorialidad.
Sin embargo, esta disposición presenta ciertas deficiencias en su redacción.
Como señala el artículo 3 del Código Penal, existen excepciones.

ANÁLISIS DEL ARTÍCULO 5
El artículo 5 trata sobre la responsabilidad penal. Según diversos autores,
la interpretación del artículo 5 ha variado con el tiempo.
Artículo 5°: No obstante lo dispuesto...

CONCLUSIONES
El artículo 1 requiere una revisión urgente.
"""

# Judicial decision with references to laws
JUDICIAL_DECISION_TEXT = """
SENTENCIA N° 123-2024

VISTOS:
Se presenta recurso de casación contra sentencia del juez.

CONSIDERANDO:
PRIMERO: Que el artículo 2 del Código Penal establece...
SEGUNDO: Que conforme al artículo 5 citado, existe excepción...
TERCERO: Que se aplica el artículo 12 en este caso.

PARTE RESOLUTIVA:
RESUELVE:
1. Declarar fundado el recurso de casación.
2. Casar la sentencia apelada.
"""

# Jurisprudence compilation
JURISPRUDENCE_COMPILATION = """
1000 CRITERIOS JURISPRUDENCIALES PENALES

CRITERIO 1:
Sentencia N° 001-2023
Artículo 2 del Código Penal - Principio de extraterritorialidad
VISTOS: Se considera el artículo 2 en este caso.
CONSIDERANDO: Que el artículo 2 se aplica.
RESUELVE: Declara procedente.

CRITERIO 2:
Sentencia N° 002-2023
Artículo 5 del Código Penal - Responsabilidad penal
Artículo 5 es relevante para el análisis.
"""

# Document with OCR errors
OCR_ERRORS_TEXT = """
LEY N° 29973
LEY QVO PRQTEGE LOS DATOS PERS0NALES

Artículo 1.- Objeto de la ley
La presene ley tije por objeto proteger el derecho al intimidad.

Artículo 2.- Principio de extraterritorialidad
La presena ley se aplica a los tratamientos de datos p3rsonales.
"""

# Document with missing page boundaries
MISSING_PAGE_TEXT = """
LEY N° 9999
Artículo 1.- Primera disposición
Artículo 2.- Segunda disposición
Artículo 3.- Tercera disposición sin salto de página claro.
Artículo 4.- Cuarta disposición continua.
"""

# Document with annexes and tables
ANNEX_TABLE_TEXT = """
LEY N° 5555
Artículo 1.- Objeto
Artículo 2.- Definiciones

ANEXO I

Cuadro resumen de sanciones:

| Artículo | Sanción | Multa |
|----------|---------|-------|
| Art. 1   | Leve    | 100   |
| Art. 2   | Grave   | 500   |

ANEXO II
Formulario de solicitud
"""

# Document with ambiguous classification
AMBIGUOUS_TEXT = """
MANUAL PRÁCTICO DEL ABOGADO

Este manual contiene explicaciones sobre el artículo 1 del Código Penal.
También incluye el texto del artículo 2 para referencia.
El artículo 3 se explica con ejemplos prácticos.

Guía de pasos a seguir:
1. Identificar el artículo aplicable.
2. Redactar la demanda.
3. Presentar ante el juzgado.

ANEXO
Formato de demanda.
"""

__all__ = [
    "PERUVIAN_LAW_TEXT",
    "REGULATION_TEXT",
    "LEGAL_TEXTBOOK_TEXT",
    "LEGAL_COMMENTARY_TEXT",
    "JUDICIAL_DECISION_TEXT",
    "JURISPRUDENCE_COMPILATION",
    "OCR_ERRORS_TEXT",
    "MISSING_PAGE_TEXT",
    "ANNEX_TABLE_TEXT",
    "AMBIGUOUS_TEXT",
]
