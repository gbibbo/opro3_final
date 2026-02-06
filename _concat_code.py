#!/usr/bin/env python3
"""Script para concatenar todo el código en un único archivo .txt"""

import os
from pathlib import Path

ROOT = Path("/mnt/fast/nobackup/users/gb0048/opro3_final")
OUTPUT = ROOT / "TODO_EL_CODIGO_PARA_GEMINI.txt"

# Carpetas a incluir
FOLDERS = ["scripts", "slurm", "src"]

# Extensiones válidas (solo código/texto)
VALID_EXTENSIONS = {".py", ".sh", ".job", ".md", ".txt", ".yaml", ".yml", ".json"}

# Excluir
EXCLUDE_DIRS = {"__pycache__", ".git", "data", "results", "checkpoints"}


def should_include(path: Path) -> bool:
    """Determina si un archivo debe incluirse."""
    # Excluir directorios no deseados
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    # Solo extensiones válidas
    return path.suffix.lower() in VALID_EXTENSIONS


def main():
    all_files = []

    # Recopilar archivos de las carpetas especificadas
    for folder in FOLDERS:
        folder_path = ROOT / folder
        if folder_path.exists():
            for file_path in folder_path.rglob("*"):
                if file_path.is_file() and should_include(file_path):
                    all_files.append(file_path)

    # También incluir CLAUDE.md del root
    claude_md = ROOT / "CLAUDE.md"
    if claude_md.exists():
        all_files.append(claude_md)

    # Ordenar por ruta
    all_files.sort()

    # Escribir el archivo de salida
    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("CONTENIDO COMPLETO DEL PROYECTO OPRO3_FINAL\n")
        out.write(f"Total de archivos: {len(all_files)}\n")
        out.write("=" * 80 + "\n\n")

        for i, file_path in enumerate(all_files, 1):
            rel_path = file_path.relative_to(ROOT)
            out.write("\n" + "=" * 80 + "\n")
            out.write(f"[{i}/{len(all_files)}] ARCHIVO: {rel_path}\n")
            out.write(f"RUTA COMPLETA: {file_path}\n")
            out.write("=" * 80 + "\n\n")

            try:
                content = file_path.read_text(encoding="utf-8")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except Exception as e:
                out.write(f"[ERROR al leer: {e}]\n")

            out.write("\n")

    print(f"Archivo generado: {OUTPUT}")
    print(f"Archivos incluidos: {len(all_files)}")
    for f in all_files:
        print(f"  - {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
