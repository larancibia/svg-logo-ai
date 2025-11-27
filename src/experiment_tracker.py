"""
Experimental Tracking System using ChromaDB
Registers every step, decision, and result for full traceability
"""

import chromadb
from chromadb.config import Settings
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
from pathlib import Path


class ExperimentTracker:
    """
    Comprehensive experiment tracking system using ChromaDB.
    Records every step, decision, and result with full metadata.
    """

    def __init__(self, experiment_name: str, base_dir: str = "/home/luis/svg-logo-ai"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize ChromaDB client
        chroma_path = self.base_dir / "chroma_experiments"
        chroma_path.mkdir(exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create collections for different types of data
        self.log_collection = self._get_or_create_collection(
            "experiment_logs",
            metadata={"description": "Step-by-step experiment execution logs"}
        )

        self.decision_collection = self._get_or_create_collection(
            "decisions",
            metadata={"description": "Key decisions and their rationale"}
        )

        self.results_collection = self._get_or_create_collection(
            "results",
            metadata={"description": "Experimental results and metrics"}
        )

        # Track experiment start
        self.log_step(
            step_type="experiment_start",
            description=f"Starting experiment: {experiment_name}",
            metadata={
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat()
            }
        )

    def _get_or_create_collection(self, name: str, metadata: Dict = None):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(name=name)
        except:
            return self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )

    def log_step(
        self,
        step_type: str,
        description: str,
        metadata: Dict[str, Any] = None,
        data: Any = None
    ) -> str:
        """
        Log a single step in the experiment

        Args:
            step_type: Type of step (e.g., 'generation', 'evaluation', 'mutation')
            description: Human-readable description
            metadata: Additional metadata
            data: Any associated data (will be JSON serialized)

        Returns:
            step_id: Unique identifier for this step
        """
        step_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Prepare metadata
        full_metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "step_type": step_type,
            "timestamp": timestamp,
            **(metadata or {})
        }

        # Prepare document (description + data)
        document = description
        if data:
            document += f"\n\nData: {json.dumps(data, indent=2)}"

        # Store in ChromaDB
        self.log_collection.add(
            ids=[step_id],
            documents=[document],
            metadatas=[full_metadata]
        )

        print(f"[{timestamp}] [{step_type}] {description}")
        return step_id

    def log_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Log a key decision point

        Args:
            decision: The decision made
            rationale: Why this decision was made
            alternatives: Other options considered
            metadata: Additional context
        """
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        document = f"DECISION: {decision}\n\nRATIONALE: {rationale}"
        if alternatives:
            document += f"\n\nALTERNATIVES CONSIDERED:\n" + "\n".join(f"- {alt}" for alt in alternatives)

        full_metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": timestamp,
            "decision_type": "key_decision",
            **(metadata or {})
        }

        self.decision_collection.add(
            ids=[decision_id],
            documents=[document],
            metadatas=[full_metadata]
        )

        print(f"[DECISION] {decision}")
        return decision_id

    def log_result(
        self,
        result_type: str,
        metrics: Dict[str, float],
        description: str = "",
        metadata: Dict[str, Any] = None,
        artifacts: Dict[str, str] = None
    ) -> str:
        """
        Log experimental results

        Args:
            result_type: Type of result (e.g., 'fitness_score', 'comparison')
            metrics: Numerical metrics
            description: Human-readable description
            metadata: Additional context
            artifacts: Paths to generated files/artifacts
        """
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        document = f"RESULT TYPE: {result_type}\n\n{description}\n\nMETRICS:\n"
        document += "\n".join(f"- {key}: {value}" for key, value in metrics.items())

        if artifacts:
            document += "\n\nARTIFACTS:\n"
            document += "\n".join(f"- {key}: {path}" for key, path in artifacts.items())

        full_metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": timestamp,
            "result_type": result_type,
            **metrics,  # Store metrics in metadata for easy querying
            **(metadata or {})
        }

        self.results_collection.add(
            ids=[result_id],
            documents=[document],
            metadatas=[full_metadata]
        )

        print(f"[RESULT] {result_type}: {metrics}")
        return result_id

    def query_logs(
        self,
        query_text: str = None,
        step_type: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query experiment logs

        Args:
            query_text: Semantic search query
            step_type: Filter by step type
            limit: Maximum number of results
        """
        where_filter = {}
        if step_type:
            where_filter["step_type"] = step_type

        if query_text:
            results = self.log_collection.query(
                query_texts=[query_text],
                n_results=limit,
                where=where_filter if where_filter else None
            )
        else:
            results = self.log_collection.get(
                limit=limit,
                where=where_filter if where_filter else None
            )

        return results

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of current experiment"""

        # Count logs by type
        all_logs = self.log_collection.get(
            where={"experiment_id": self.experiment_id}
        )

        step_types = {}
        for metadata in all_logs['metadatas']:
            step_type = metadata.get('step_type', 'unknown')
            step_types[step_type] = step_types.get(step_type, 0) + 1

        # Get all decisions
        decisions = self.decision_collection.get(
            where={"experiment_id": self.experiment_id}
        )

        # Get all results
        results = self.results_collection.get(
            where={"experiment_id": self.experiment_id}
        )

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "total_logs": len(all_logs['ids']),
            "logs_by_type": step_types,
            "total_decisions": len(decisions['ids']),
            "total_results": len(results['ids']),
            "latest_logs": [
                {
                    "type": all_logs['metadatas'][i].get('step_type'),
                    "timestamp": all_logs['metadatas'][i].get('timestamp'),
                    "description": all_logs['documents'][i][:100] + "..."
                }
                for i in range(min(5, len(all_logs['ids'])))
            ]
        }

    def export_trace(self, output_path: Optional[str] = None) -> str:
        """
        Export complete experimental trace to JSON

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.base_dir / "experiments" / f"{self.experiment_id}_trace.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Get all data
        logs = self.log_collection.get(
            where={"experiment_id": self.experiment_id}
        )

        decisions = self.decision_collection.get(
            where={"experiment_id": self.experiment_id}
        )

        results = self.results_collection.get(
            where={"experiment_id": self.experiment_id}
        )

        trace = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "logs": [
                {
                    "id": logs['ids'][i],
                    "document": logs['documents'][i],
                    "metadata": logs['metadatas'][i]
                }
                for i in range(len(logs['ids']))
            ],
            "decisions": [
                {
                    "id": decisions['ids'][i],
                    "document": decisions['documents'][i],
                    "metadata": decisions['metadatas'][i]
                }
                for i in range(len(decisions['ids']))
            ],
            "results": [
                {
                    "id": results['ids'][i],
                    "document": results['documents'][i],
                    "metadata": results['metadatas'][i]
                }
                for i in range(len(results['ids']))
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(trace, f, indent=2)

        self.log_step(
            step_type="export",
            description=f"Exported experimental trace to {output_path}",
            metadata={"output_path": str(output_path)}
        )

        return str(output_path)

    def finalize(self):
        """Finalize experiment and export trace"""
        self.log_step(
            step_type="experiment_end",
            description=f"Finalizing experiment: {self.experiment_name}",
            metadata={"timestamp": datetime.now().isoformat()}
        )

        summary = self.get_experiment_summary()
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(json.dumps(summary, indent=2))
        print("="*80 + "\n")

        trace_path = self.export_trace()
        print(f"Full experimental trace exported to: {trace_path}")

        return trace_path
