"""Esquema de entrada de la API (Pydantic v2).

Reglas:
- 8 parámetros capturados por el usuario + departamento.
- Sin valores por defecto silenciosos: un campo faltante o inválido es 422.
- Campos extra también son rechazados (``extra="forbid"``).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from model_service import departamentos

#: Enum de los 33 departamentos, generado desde columnas_edutrack.pkl.
Departamento = Enum(
    "Departamento",
    {nombre: nombre for nombre in departamentos()},
    type=str,
    module=__name__,
)


class PrediccionRequest(BaseModel):
    """Cuerpo esperado por ``POST /predecir``."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "matricula": 210,
                "docentes": 8,
                "orientadores": 0,
                "niveles": 2,
                "jornadas": 1,
                "zona": "rural",
                "bilingue": False,
                "etnoeducativa": False,
                "departamento": "Amazonas",
            }
        },
    )

    matricula: int = Field(
        ge=0, le=100_000, description="Estudiantes matriculados en la sede"
    )
    docentes: int = Field(
        ge=1, le=5_000, description="Docentes de aula (divisor de EST_POR_DOCENTE)"
    )
    orientadores: int = Field(
        ge=0, le=500, description="Docentes orientadores / psicosociales"
    )
    niveles: int = Field(ge=1, le=6, description="Niveles educativos ofrecidos")
    jornadas: int = Field(ge=1, le=5, description="Jornadas operativas")
    zona: Literal["rural", "urbana"] = Field(description="Zona de ubicación")
    bilingue: bool = Field(description="¿La sede es bilingüe?")
    etnoeducativa: bool = Field(description="¿La sede es etnoeducativa?")
    departamento: Departamento = Field(description="Departamento (33 valores)")
