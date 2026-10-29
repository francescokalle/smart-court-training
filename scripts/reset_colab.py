#!/usr/bin/env python3
"""Reset completo dell'ambiente Colab.

Rimuove tutti i dati, modelli e codice in /content, pulisce la cache pip
e reinstalla le dipendenze base. Dopo l'esecuzione, /content sarà pulito.

Uso:
    python reset_colab.py [--keep-deps]
"""
import argparse
import os
import shutil
import subprocess
import sys


def run(cmd, check=True, capture=False):
    print(f"$ {' '.join(cmd)}")
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check)


def remove_dir(path):
    if os.path.exists(path):
        print(f"🗑️  Rimozione {path}...")
        shutil.rmtree(path, ignore_errors=True)
    else:
        print(f"   {path} non esiste, skip.")


def main():
    parser = argparse.ArgumentParser(description="Reset completo Colab")
    parser.add_argument("--keep-deps", action="store_true",
                        help="Non reinstallare le dipendenze base")
    args = parser.parse_args()

    print("=" * 60)
    print("  RESET COMPLETO COLAB")
    print("=" * 60)

    # 1. Rimuovi directory di lavoro
    for d in ["/content/data", "/content/models", "/content/smart-court-training",
              "/content/runs", "/content/datasets"]:
        remove_dir(d)

    # 2. Pulisci cache pip
    print("\n🧹 Pulizia cache pip...")
    run([sys.executable, "-m", "pip", "cache", "purge"], check=False)

    # 3. Reinstalla dipendenze base (se non --keep-deps)
    if not args.keep_deps:
        print("\n📦 Reinstallazione dipendenze base...")
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

    # 4. Verifica spazio
    print("\n📊 Spazio disco:")
    run(["df", "-h", "/content"], check=False)

    print("\n" + "=" * 60)
    print("  ✅ RESET COMPLETATO")
    print("  /content è pulito. Ora esegui setup_full.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
