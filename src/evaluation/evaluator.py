from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models.llms import BaseLLM
import json
import os
from datetime import datetime

class RAGEvaluator:
    """
    A class for evaluating RAG system performance, tracking metrics,
    and detecting hallucinations or errors in responses.
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        log_dir: str = "./eval_logs",
        metrics_file: str = "rag_metrics.json"
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            llm: Language model for evaluations
            log_dir: Directory to store evaluation logs
            metrics_file: File to store metrics
        """
        self.llm = llm or ChatOpenAI(model_name="llama3-8b-8192", temperature=0)
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, metrics_file)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load existing metrics from file or create new metrics object."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading metrics file. Creating new one.")
                
        # Initialize new metrics object
        return {
            "queries": 0,
            "relevance_scores": [],
            "avg_relevance": 0.0,
            "faithfulness_scores": [],
            "avg_faithfulness": 0.0,
            "hallucination_rate": 0.0,
            "response_times": [],
            "avg_response_time": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_metrics(self):
        """Save metrics to file."""
        self.metrics["last_updated"] = datetime.now().isoformat()
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_query(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: str,
        response_time: float,
        session_id: Optional[str] = None
    ):
        """
        Log a query and its results for evaluation.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            answer: Generated answer
            response_time: Time taken to generate the answer (seconds)
            session_id: Optional session identifier
        """
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or "anonymous",
            "query": query,
            "retrieved_docs": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in retrieved_docs
            ],
            "answer": answer,
            "response_time": response_time
        }
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_{timestamp}.json"
        log_path = os.path.join(self.log_dir, filename)
        
        # Save log to file
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
            
        # Update metrics
        self.metrics["queries"] += 1
        self.metrics["response_times"].append(response_time)
        self.metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        
        # Save updated metrics
        self._save_metrics()
        
        return log_path
    
    def evaluate_relevance(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: str
    ) -> float:
        """
        Evaluate the relevance of retrieved documents to the query.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            answer: Generated answer
            
        Returns:
            Relevance score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
            
        # Prompt for evaluating relevance
        prompt = f"""You are an objective evaluator for RAG systems.
        
        Query: "{query}"
        
        Answer: "{answer}"
        
        Retrieved Documents:
        {chr(10).join([f"Document {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(retrieved_docs)])}
        
        On a scale of 0 to 10, rate how relevant the retrieved documents are to the query.
        Consider:
        - Do the documents contain information needed to answer the query?
        - Are the documents on topic?
        - Is there unnecessary or irrelevant information?
        
        Provide only a numerical score from 0 to 10:"""
        
        # Get relevance score
        try:
            response = self.llm.predict(prompt).strip()
            score = float(response) / 10.0  # Convert to 0-1 scale
            
            # Update metrics
            self.metrics["relevance_scores"].append(score)
            self.metrics["avg_relevance"] = sum(self.metrics["relevance_scores"]) / len(self.metrics["relevance_scores"])
            self._save_metrics()
            
            return score
        except ValueError:
            print(f"Error parsing relevance score: {response}")
            return 0.5  # Default middle score
    
    def evaluate_faithfulness(
        self,
        retrieved_docs: List[Document],
        answer: str
    ) -> float:
        """
        Evaluate the faithfulness of the answer to the retrieved documents.
        Detects potential hallucinations or unsupported claims.
        
        Args:
            retrieved_docs: Retrieved documents
            answer: Generated answer
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
            
        # Combine document contents
        combined_docs = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prompt for evaluating faithfulness
        prompt = f"""You are an expert evaluator for detecting AI hallucinations and unsupported claims.
        
        Answer generated by AI: "{answer}"
        
        Source Documents:
        {combined_docs[:2000]}  # Limit to avoid exceeding context window
        
        On a scale of 0 to 10, rate how faithful the answer is to the source documents.
        Consider:
        - Does the answer only contain information found in the documents?
        - Are there any claims or statements not supported by the documents?
        - Does the answer contradict the documents?
        
        Provide only a numerical score from 0 to 10:"""
        
        # Get faithfulness score
        try:
            response = self.llm.predict(prompt).strip()
            score = float(response) / 10.0  # Convert to 0-1 scale
            
            # Update metrics
            self.metrics["faithfulness_scores"].append(score)
            self.metrics["avg_faithfulness"] = sum(self.metrics["faithfulness_scores"]) / len(self.metrics["faithfulness_scores"])
            
            # Calculate hallucination rate (1 - faithfulness)
            self.metrics["hallucination_rate"] = 1 - self.metrics["avg_faithfulness"]
            
            self._save_metrics()
            
            return score
        except ValueError:
            print(f"Error parsing faithfulness score: {response}")
            return 0.5  # Default middle score
    
    def detect_hallucinations(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: str
    ) -> Dict[str, Any]:
        """
        Detect potential hallucinations or unsupported claims in the answer.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            answer: Generated answer
            
        Returns:
            Dictionary with hallucination analysis
        """
        # Combine document contents
        combined_docs = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Prompt for detecting hallucinations
        prompt = f"""You are an expert fact-checker tasked with identifying potential hallucinations or unsupported claims in AI-generated answers.
        
        Query: "{query}"
        
        Answer generated by AI: "{answer}"
        
        Source Documents:
        {combined_docs[:2000]}  # Limit to avoid exceeding context window
        
        Carefully analyze the answer for any statements not supported by the source documents.
        Identify specific claims that appear to be hallucinations or that go beyond the information provided.
        
        Your response should be in this JSON format:
        {{
            "has_hallucinations": true/false,
            "hallucinated_statements": ["list of", "unsupported statements"],
            "confidence": 0.8  # your confidence in this assessment (0-1)
        }}
        
        Return only the JSON object, nothing else:"""
        
        # Get hallucination analysis
        try:
            response = self.llm.predict(prompt).strip()
            analysis = json.loads(response)
            return analysis
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing hallucination analysis: {e}")
            return {
                "has_hallucinations": None,
                "hallucinated_statements": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for the RAG system.
        
        Returns:
            Dictionary with performance metrics and analysis
        """
        # Load the latest metrics
        metrics = self._load_metrics()
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": metrics["queries"],
            "retrieval_quality": {
                "avg_relevance": metrics["avg_relevance"],
                "relevance_trend": self._calculate_trend(metrics["relevance_scores"][-10:]) if len(metrics["relevance_scores"]) >= 10 else "insufficient_data"
            },
            "answer_quality": {
                "avg_faithfulness": metrics["avg_faithfulness"],
                "hallucination_rate": metrics["hallucination_rate"],
                "faithfulness_trend": self._calculate_trend(metrics["faithfulness_scores"][-10:]) if len(metrics["faithfulness_scores"]) >= 10 else "insufficient_data"
            },
            "performance": {
                "avg_response_time": metrics["avg_response_time"],
                "response_time_trend": self._calculate_trend(metrics["response_times"][-10:]) if len(metrics["response_times"]) >= 10 else "insufficient_data"
            },
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return report
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend (improving, stable, declining) from a list of values."""
        if not values or len(values) < 3:
            return "insufficient_data"
            
        # Simple trend analysis
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = second_half - first_half
        if abs(diff) < 0.05:  # Threshold for "stable"
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "declining"
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # Recommendations based on relevance
        if metrics.get("avg_relevance", 0) < 0.7:
            recommendations.append("Consider improving document retrieval by adjusting chunk size or using a hybrid search approach.")
        
        # Recommendations based on faithfulness
        if metrics.get("hallucination_rate", 0) > 0.3:
            recommendations.append("Reduce hallucinations by adding more relevant documents to the knowledge base or adjusting the LLM temperature parameter.")
        
        # Recommendations based on response time
        if metrics.get("avg_response_time", 0) > 5.0:
            recommendations.append("Optimize response time by using a smaller model, caching common queries, or optimizing vector search parameters.")
        
        # Add a general recommendation if none specific
        if not recommendations:
            recommendations.append("System is performing well. Consider A/B testing different retrieval strategies to further optimize performance.")
        
        return recommendations
    
    # TruLens-style evaluation 
    def evaluate_with_trulens(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline using TruLens-inspired metrics.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            answer: Generated answer
            
        Returns:
            Dictionary with TruLens-style evaluation results
        """
        # Calculate relevance
        relevance = self.evaluate_relevance(query, retrieved_docs, answer)
        
        # Calculate faithfulness
        faithfulness = self.evaluate_faithfulness(retrieved_docs, answer)
        
        # Answer relevance (is the answer addressing the question?)
        prompt = f"""You are an objective evaluator for question-answering systems.
        
        Query: "{query}"
        
        Answer: "{answer}"
        
        On a scale of 0 to 10, rate how directly the answer addresses the query.
        Consider:
        - Does the answer respond to what was asked?
        - Is the answer on topic?
        - Does the answer provide the information requested?
        
        Provide only a numerical score from 0 to 10:"""
        
        try:
            response = self.llm.predict(prompt).strip()
            answer_relevance = float(response) / 10.0
        except ValueError:
            answer_relevance = 0.5
        
        # Contextual precision (ratio of relevant to irrelevant information)
        contextual_precision = sum([self._calculate_doc_relevance(query, doc) for doc in retrieved_docs]) / len(retrieved_docs) if retrieved_docs else 0
        
        # Combine into TruLens-style metrics
        trulens_metrics = {
            "context_relevance": relevance,
            "answer_relevance": answer_relevance,
            "answer_faithfulness": faithfulness,
            "contextual_precision": contextual_precision,
            "rag_triad": (relevance + faithfulness + answer_relevance) / 3,
            "timestamp": datetime.now().isoformat()
        }
        
        return trulens_metrics
    
    def _calculate_doc_relevance(self, query: str, doc: Document) -> float:
        """Calculate relevance of a single document to the query on a 0-1 scale."""
        prompt = f"""You are an objective evaluator for document retrieval.
        
        Query: "{query}"
        
        Document: "{doc.page_content[:500]}..."
        
        On a scale of 0 to 1, is this document relevant to answering the query?
        Return only a number between 0 and 1:"""
        
        try:
            response = self.llm.predict(prompt).strip()
            return float(response)
        except ValueError:
            return 0.5
