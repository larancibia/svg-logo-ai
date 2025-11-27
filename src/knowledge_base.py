"""
Sistema de base de conocimiento para generación de logos vectoriales con IA
Usa ChromaDB para almacenar y recuperar información sobre modelos y técnicas
"""

import chromadb
from typing import List, Dict
import json


class SVGKnowledgeBase:
    """Gestiona el almacenamiento y recuperación de conocimiento sobre generación SVG"""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Inicializa la base de conocimiento con ChromaDB

        Args:
            persist_directory: Directorio donde se persiste la base de datos
        """
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Colección para papers de investigación
        self.papers_collection = self.client.get_or_create_collection(
            name="svg_research_papers",
            metadata={"description": "Papers sobre generación vectorial con IA"}
        )

        # Colección para técnicas y métodos
        self.techniques_collection = self.client.get_or_create_collection(
            name="svg_techniques",
            metadata={"description": "Técnicas y métodos de generación SVG"}
        )

        # Colección para modelos específicos
        self.models_collection = self.client.get_or_create_collection(
            name="svg_models",
            metadata={"description": "Modelos de IA para generación vectorial"}
        )

    def add_paper(self, title: str, authors: str, summary: str,
                  key_findings: List[str], url: str = None):
        """Agrega un paper de investigación a la base de conocimiento"""
        doc_id = title.lower().replace(" ", "_")

        # Texto completo para embeddings
        full_text = f"{title}. {authors}. {summary} " + " ".join(key_findings)

        metadata = {
            "type": "research_paper",
            "title": title,
            "authors": authors,
            "url": url or "",
            "key_findings_count": len(key_findings)
        }

        self.papers_collection.add(
            documents=[full_text],
            metadatas=[metadata],
            ids=[doc_id]
        )

        print(f"✓ Paper agregado: {title}")

    def add_model(self, name: str, description: str, capabilities: List[str],
                  limitations: List[str], implementation: str = None):
        """Agrega información sobre un modelo de IA específico"""
        doc_id = name.lower().replace(" ", "_").replace("-", "_")

        full_text = f"{name}. {description}. Capabilities: " + ", ".join(capabilities)
        full_text += ". Limitations: " + ", ".join(limitations)

        metadata = {
            "type": "model",
            "name": name,
            "has_implementation": implementation is not None,
            "capabilities_count": len(capabilities)
        }

        self.models_collection.add(
            documents=[full_text],
            metadatas=[metadata],
            ids=[doc_id]
        )

        print(f"✓ Modelo agregado: {name}")

    def add_technique(self, name: str, description: str, category: str,
                     difficulty: str, use_cases: List[str]):
        """Agrega una técnica o método de generación SVG"""
        doc_id = f"{category}_{name}".lower().replace(" ", "_")

        full_text = f"{name} ({category}). {description}. Use cases: " + ", ".join(use_cases)

        metadata = {
            "type": "technique",
            "name": name,
            "category": category,
            "difficulty": difficulty
        }

        self.techniques_collection.add(
            documents=[full_text],
            metadatas=[metadata],
            ids=[doc_id]
        )

        print(f"✓ Técnica agregada: {name}")

    def search_papers(self, query: str, n_results: int = 5) -> List[Dict]:
        """Busca papers relevantes"""
        results = self.papers_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return self._format_results(results)

    def search_models(self, query: str, n_results: int = 5) -> List[Dict]:
        """Busca modelos relevantes"""
        results = self.models_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return self._format_results(results)

    def search_techniques(self, query: str, n_results: int = 5) -> List[Dict]:
        """Busca técnicas relevantes"""
        results = self.techniques_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return self._format_results(results)

    def search_all(self, query: str, n_results: int = 3) -> Dict[str, List]:
        """Busca en todas las colecciones"""
        return {
            "papers": self.search_papers(query, n_results),
            "models": self.search_models(query, n_results),
            "techniques": self.search_techniques(query, n_results)
        }

    def _format_results(self, results) -> List[Dict]:
        """Formatea los resultados de ChromaDB"""
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "metadata": results['metadatas'][0][i],
                "document": results['documents'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        return formatted

    def get_stats(self) -> Dict:
        """Retorna estadísticas de la base de conocimiento"""
        return {
            "papers": self.papers_collection.count(),
            "models": self.models_collection.count(),
            "techniques": self.techniques_collection.count()
        }


if __name__ == "__main__":
    # Demo de uso
    kb = SVGKnowledgeBase()
    print("Base de conocimiento inicializada")
    print(f"Estadísticas: {kb.get_stats()}")
