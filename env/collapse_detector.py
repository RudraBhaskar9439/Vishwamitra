"""
VIDYA Collapse Detection Module

Early warning system for educational system collapse using:
- Critical slowing down indicators
- Correlation network analysis  
- LSTM-based ensemble prediction
- Shock propagation modeling
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass
class CollapseWarning:
    """Structure for collapse warnings."""
    probability: float
    method: str
    confidence: float
    time_to_collapse: Optional[int]
    indicators: Dict[str, float]
    recommendation: str


class CollapseDetector(ABC):
    """Base class for collapse detection methods."""
    
    @abstractmethod
    def predict(self, state_history: List[np.ndarray]) -> CollapseWarning:
        """Predict collapse probability from state history."""
        pass
    
    @abstractmethod
    def fit(self, normal_data: List[np.ndarray], collapse_data: List[np.ndarray]):
        """Train detector on historical data."""
        pass


class CriticalSlowingDownDetector(CollapseDetector):
    """
    Detect critical slowing down before phase transitions.
    
    Based on theory that complex systems lose resilience before collapse,
    exhibiting increased variance, autocorrelation, and skewness.
    """
    
    def __init__(self, window_size: int = 20, n_indicators: int = 5):
        self.window_size = window_size
        self.n_indicators = n_indicators
        self.baseline_stats = None
        
    def fit(self, normal_data: List[np.ndarray], collapse_data: List[np.ndarray]):
        """Learn normal operating statistics."""
        normal_array = np.array(normal_data)
        self.baseline_stats = {
            'mean': np.mean(normal_array, axis=0),
            'std': np.std(normal_array, axis=0),
            'variance': np.var(normal_array, axis=0),
        }
        
        # Learn collapse signatures
        collapse_array = np.array(collapse_data)
        self.collapse_stats = {
            'mean': np.mean(collapse_array, axis=0),
            'std': np.std(collapse_array, axis=0),
        }
        
    def predict(self, state_history: List[np.ndarray]) -> CollapseWarning:
        """
        Detect critical slowing down indicators.
        
        Returns warning with probability and key indicators.
        """
        if len(state_history) < self.window_size:
            return CollapseWarning(
                probability=0.0,
                method='critical_slowing_down',
                confidence=0.0,
                time_to_collapse=None,
                indicators={},
                recommendation="Insufficient data for detection"
            )
            
        recent_states = np.array(state_history[-self.window_size:])
        
        # Compute critical slowing down indicators
        indicators = {}
        
        # 1. Variance increase
        recent_var = np.var(recent_states, axis=0)
        if self.baseline_stats is not None:
            var_ratio = recent_var / (self.baseline_stats['variance'] + 1e-8)
            indicators['variance_increase'] = float(np.mean(var_ratio))
        else:
            indicators['variance_increase'] = float(np.mean(recent_var))
            
        # 2. Autocorrelation increase (system has memory)
        autocorrs = []
        for i in range(recent_states.shape[1]):
            if len(recent_states) > 1:
                autocorr = np.corrcoef(
                    recent_states[:-1, i], 
                    recent_states[1:, i]
                )[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
        indicators['autocorrelation'] = float(np.mean(autocorrs)) if autocorrs else 0.0
        
        # 3. Skewness increase
        from scipy import stats
        skewness = np.mean([stats.skew(recent_states[:, i]) 
                           for i in range(recent_states.shape[1])])
        indicators['skewness'] = float(skewness)
        
        # 4. Recovery rate (return time after perturbation)
        if len(state_history) >= 2 * self.window_size:
            older_states = np.array(state_history[-2*self.window_size:-self.window_size])
            recovery_indicator = self._compute_recovery_rate(recent_states, older_states)
            indicators['recovery_rate'] = recovery_indicator
        else:
            indicators['recovery_rate'] = 1.0  # Normal recovery
            
        # Compute collapse probability from indicators
        # Higher variance, autocorrelation, skewness -> higher collapse risk
        # Lower recovery rate -> higher collapse risk
        
        var_score = min(indicators['variance_increase'] / 2.0, 1.0)
        autocorr_score = max(0, indicators['autocorrelation'])  # Positive autocorr is bad
        skew_score = min(abs(indicators['skewness']) / 2.0, 1.0)
        recovery_score = max(0, 1 - indicators['recovery_rate'])
        
        # Weighted combination
        collapse_prob = (
            0.3 * var_score + 
            0.3 * autocorr_score + 
            0.2 * skew_score + 
            0.2 * recovery_score
        )
        
        # Estimate time to collapse
        if collapse_prob > 0.7:
            time_estimate = max(5, int(20 * (1 - collapse_prob)))
        else:
            time_estimate = None
            
        recommendation = self._generate_recommendation(indicators, collapse_prob)
        
        return CollapseWarning(
            probability=float(collapse_prob),
            method='critical_slowing_down',
            confidence=min(1.0, len(state_history) / 100),
            time_to_collapse=time_estimate,
            indicators=indicators,
            recommendation=recommendation
        )
        
    def _compute_recovery_rate(
        self, 
        recent: np.ndarray, 
        older: np.ndarray
    ) -> float:
        """Compute how quickly system returns to equilibrium after perturbation."""
        # Compute mean deviation from trend
        recent_trend = np.polyfit(range(len(recent)), recent.mean(axis=1), 1)[0]
        older_trend = np.polyfit(range(len(older)), older.mean(axis=1), 1)[0]
        
        # Slower recovery = flatter trend
        if abs(older_trend) < 1e-6:
            return 1.0
        recovery_rate = abs(recent_trend / older_trend)
        return min(recovery_rate, 2.0)  # Cap at 2x
        
    def _generate_recommendation(
        self, 
        indicators: Dict[str, float], 
        collapse_prob: float
    ) -> str:
        """Generate intervention recommendation based on indicators."""
        if collapse_prob < 0.3:
            return "System stable. Continue monitoring."
        elif collapse_prob < 0.6:
            return "Early warning signs detected. Consider preventive measures."
        elif collapse_prob < 0.8:
            return "Critical slowing down evident. Immediate intervention recommended."
        else:
            return "Imminent collapse risk. Emergency response required."


class CorrelationNetworkDetector(CollapseDetector):
    """
    Detect collapse through correlation network analysis.
    
    As systems approach collapse, components become more correlated
    (dominated by shared stressors rather than local dynamics).
    """
    
    def __init__(self, threshold: float = 0.8, window_size: int = 15):
        self.threshold = threshold
        self.window_size = window_size
        self.baseline_correlation = None
        
    def fit(self, normal_data: List[np.ndarray], collapse_data: List[np.ndarray]):
        """Learn baseline correlation structure."""
        normal_array = np.array(normal_data[-self.window_size:])
        self.baseline_correlation = np.corrcoef(normal_array.T)
        
    def predict(self, state_history: List[np.ndarray]) -> CollapseWarning:
        """Analyze correlation network structure."""
        if len(state_history) < self.window_size:
            return CollapseWarning(
                probability=0.0,
                method='correlation_network',
                confidence=0.0,
                time_to_collapse=None,
                indicators={},
                recommendation="Insufficient data"
            )
            
        recent = np.array(state_history[-self.window_size:])
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(recent.T)
        
        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        indicators = {}
        
        # 1. Mean absolute correlation
        # Exclude diagonal
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        mean_corr = np.mean(np.abs(corr_matrix[mask]))
        indicators['mean_correlation'] = float(mean_corr)
        
        # 2. Correlation increase from baseline
        if self.baseline_correlation is not None:
            corr_change = mean_corr - np.mean(np.abs(self.baseline_correlation[mask]))
            indicators['correlation_increase'] = float(corr_change)
        else:
            indicators['correlation_increase'] = 0.0
            
        # 3. Network density (proportion of strong correlations)
        strong_corr = np.sum(np.abs(corr_matrix[mask]) > self.threshold)
        total_pairs = np.sum(mask)
        network_density = strong_corr / total_pairs if total_pairs > 0 else 0
        indicators['network_density'] = float(network_density)
        
        # 4. Spectral properties (eigenvalue distribution)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        indicators['spectral_entropy'] = float(self._compute_spectral_entropy(eigenvalues))
        
        # Compute collapse probability
        # Higher correlation, density, lower entropy -> higher collapse risk
        corr_score = min(mean_corr / 0.8, 1.0)
        density_score = network_density
        entropy_score = max(0, 1 - indicators['spectral_entropy'] / 2.0)
        
        collapse_prob = 0.4 * corr_score + 0.3 * density_score + 0.3 * entropy_score
        
        recommendation = self._generate_recommendation(indicators, collapse_prob)
        
        return CollapseWarning(
            probability=float(collapse_prob),
            method='correlation_network',
            confidence=min(1.0, len(state_history) / 100),
            time_to_collapse=max(5, int(15 * (1 - collapse_prob))) if collapse_prob > 0.6 else None,
            indicators=indicators,
            recommendation=recommendation
        )
        
    def _compute_spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """Compute spectral entropy of correlation matrix."""
        # Normalize eigenvalues
        ev_norm = eigenvalues / (eigenvalues.sum() + 1e-8)
        # Shannon entropy
        entropy = -np.sum(ev_norm * np.log(ev_norm + 1e-8))
        return float(entropy)
        
    def _generate_recommendation(self, indicators: Dict[str, float], prob: float) -> str:
        if indicators.get('network_density', 0) > 0.5:
            return "High system coupling detected. Diversify interventions."
        elif prob > 0.7:
            return "System components highly synchronized. Risk of cascade failure."
        else:
            return "Monitor inter-component correlations."


class LSTMEnsembleDetector(CollapseDetector):
    """
    Deep learning-based collapse detection using LSTM ensemble.
    
    Trains multiple LSTM models with different architectures and
    combines predictions for robust detection.
    """
    
    def __init__(
        self, 
        sequence_length: int = 30,
        n_models: int = 3,
        warning_steps: int = 10
    ):
        self.sequence_length = sequence_length
        self.n_models = n_models
        self.warning_steps = warning_steps
        self.models = []
        self.is_fitted = False
        
    def fit(self, normal_data: List[np.ndarray], collapse_data: List[np.ndarray]):
        """Train ensemble of LSTM models."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            warnings.warn("PyTorch not available. LSTM detector will not work.")
            return
            
        # Prepare training data
        X, y = self._prepare_sequences(normal_data, collapse_data)
        
        if len(X) < 10:
            warnings.warn("Insufficient data for LSTM training")
            return
            
        # Train ensemble
        self.models = []
        for i in range(self.n_models):
            model = self._build_lstm_model(X.shape[-1], hidden_size=64 + i*32)
            self._train_model(model, X, y)
            self.models.append(model)
            
        self.is_fitted = True
        
    def predict(self, state_history: List[np.ndarray]) -> CollapseWarning:
        """Generate ensemble prediction."""
        if not self.is_fitted or len(self.models) == 0:
            return CollapseWarning(
                probability=0.0,
                method='lstm_ensemble',
                confidence=0.0,
                time_to_collapse=None,
                indicators={},
                recommendation="Model not trained"
            )
            
        if len(state_history) < self.sequence_length:
            return CollapseWarning(
                probability=0.0,
                method='lstm_ensemble',
                confidence=0.0,
                time_to_collapse=None,
                indicators={},
                recommendation="Insufficient sequence length"
            )
            
        # Prepare input sequence
        import torch
        sequence = np.array(state_history[-self.sequence_length:])
        X = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dim
        
        # Get predictions from ensemble
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(X))
                predictions.append(pred.item())
                
        # Ensemble statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        indicators = {
            'mean_prediction': float(mean_pred),
            'prediction_std': float(std_pred),
            'model_agreement': 1 - std_pred / max(mean_pred, 0.01)
        }
        
        confidence = 1 - std_pred
        time_estimate = self.warning_steps if mean_pred > 0.7 else None
        
        recommendation = "LSTM prediction complete. "
        if mean_pred > 0.8:
            recommendation += "High collapse probability predicted."
        elif mean_pred > 0.5:
            recommendation += "Elevated risk detected."
        else:
            recommendation += "System appears stable."
            
        return CollapseWarning(
            probability=float(mean_pred),
            method='lstm_ensemble',
            confidence=float(confidence),
            time_to_collapse=time_estimate,
            indicators=indicators,
            recommendation=recommendation
        )
        
    def _prepare_sequences(
        self,
        normal_data: List[np.ndarray],
        collapse_data: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences with labels."""
        sequences = []
        labels = []
        
        # Normal sequences (label 0)
        for i in range(len(normal_data) - self.sequence_length):
            seq = normal_data[i:i + self.sequence_length]
            sequences.append(seq)
            labels.append(0)
            
        # Pre-collapse sequences (label 1)
        for i in range(max(0, len(collapse_data) - self.sequence_length)):
            seq = collapse_data[i:i + self.sequence_length]
            sequences.append(seq)
            labels.append(1)
            
        return np.array(sequences), np.array(labels)
        
    def _build_lstm_model(self, input_dim: int, hidden_size: int):
        """Build LSTM classifier."""
        import torch
        import torch.nn as nn
        
        class LSTMClassifier(nn.Module):
            def __init__(self, input_dim, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(0.3)
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Take last output
                last_out = lstm_out[:, -1, :]
                last_out = self.dropout(last_out)
                return self.fc(last_out)
                
        return LSTMClassifier(input_dim, hidden_size)
        
    def _train_model(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray,
        epochs: int = 50
    ):
        """Train single LSTM model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()


class ShockPropagationDetector(CollapseDetector):
    """
    Model how shocks propagate through the educational system.
    
    Tracks how local perturbations spread and amplify through
    the network of interacting components.
    """
    
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.adjacency_matrix = None
        
    def fit(self, normal_data: List[np.ndarray], collapse_data: List[np.ndarray]):
        """Learn shock propagation network from data."""
        if len(normal_data) < 10:
            return
            
        # Compute Granger causality-like influence matrix
        data_array = np.array(normal_data)
        n_vars = data_array.shape[1]
        
        influence_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Simple correlation-based influence
                    if len(data_array) > 1:
                        corr = np.corrcoef(data_array[:-1, i], data_array[1:, j])[0, 1]
                        influence_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
                        
        self.adjacency_matrix = influence_matrix
        
    def predict(self, state_history: List[np.ndarray]) -> CollapseWarning:
        """Analyze shock propagation potential."""
        if len(state_history) < 5:
            return CollapseWarning(
                probability=0.0,
                method='shock_propagation',
                confidence=0.0,
                time_to_collapse=None,
                indicators={},
                recommendation="Insufficient data"
            )
            
        recent = np.array(state_history[-10:])
        
        indicators = {}
        
        # 1. System stress (deviation from equilibrium)
        mean_state = np.mean(recent, axis=0)
        stress = np.abs(mean_state - 0.5)  # Assume 0.5 is equilibrium
        indicators['system_stress'] = float(np.mean(stress))
        
        # 2. Amplification factor (how much shocks grow)
        if len(state_history) >= 2:
            diffs = np.diff(recent, axis=0)
            amplification = np.mean(np.abs(diffs[1:])) / (np.mean(np.abs(diffs[:-1])) + 1e-8)
            indicators['amplification_factor'] = float(amplification)
        else:
            indicators['amplification_factor'] = 1.0
            
        # 3. Cascade potential (based on network structure)
        if self.adjacency_matrix is not None:
            # Compute cascade potential using network centrality
            eigenvalues = np.linalg.eigvals(self.adjacency_matrix)
            spectral_radius = np.max(np.abs(eigenvalues))
            indicators['cascade_potential'] = float(spectral_radius)
        else:
            indicators['cascade_potential'] = 0.5
            
        # 4. Vulnerable nodes (high stress + high connectivity)
        if self.adjacency_matrix is not None:
            node_vulnerability = stress * self.adjacency_matrix.sum(axis=1)
            indicators['max_vulnerability'] = float(np.max(node_vulnerability))
        else:
            indicators['max_vulnerability'] = float(np.max(stress))
            
        # Compute collapse probability
        stress_score = min(indicators['system_stress'] * 2, 1.0)
        amp_score = min(max(0, indicators['amplification_factor'] - 1) * 2, 1.0)
        cascade_score = min(indicators['cascade_potential'] / 2, 1.0)
        vuln_score = min(indicators['max_vulnerability'], 1.0)
        
        collapse_prob = (
            0.25 * stress_score + 
            0.25 * amp_score + 
            0.25 * cascade_score + 
            0.25 * vuln_score
        )
        
        recommendation = "Shock propagation analysis: "
        if indicators['amplification_factor'] > 1.5:
            recommendation += "Shocks amplifying. System vulnerable to cascades."
        elif indicators['cascade_potential'] > 1.0:
            recommendation += "High cascade potential in network structure."
        else:
            recommendation += "Shock propagation appears controlled."
            
        return CollapseWarning(
            probability=float(collapse_prob),
            method='shock_propagation',
            confidence=0.7,
            time_to_collapse=max(3, int(10 * (1 - collapse_prob))) if collapse_prob > 0.6 else None,
            indicators=indicators,
            recommendation=recommendation
        )


class EnsembleCollapseDetector(CollapseDetector):
    """
    Combine multiple collapse detection methods for robust prediction.
    """
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None
    ):
        self.methods = methods or ['critical_slowing_down', 'correlation_network', 'shock_propagation']
        self.weights = weights
        
        self.detectors = {
            'critical_slowing_down': CriticalSlowingDownDetector(),
            'correlation_network': CorrelationNetworkDetector(),
            'shock_propagation': ShockPropagationDetector(),
        }
        
    def fit(self, normal_data: List[np.ndarray], collapse_data: List[np.ndarray]):
        """Train all detection methods."""
        for method in self.methods:
            self.detectors[method].fit(normal_data, collapse_data)
            
    def predict(self, state_history: List[np.ndarray]) -> CollapseWarning:
        """Generate ensemble prediction."""
        warnings_list = []
        
        for method in self.methods:
            warning = self.detectors[method].predict(state_history)
            warnings_list.append(warning)
            
        # Weighted average of probabilities
        probs = np.array([w.probability for w in warnings_list])
        confidences = np.array([w.confidence for w in warnings_list])
        
        if self.weights is not None:
            weights = self.weights
        else:
            # Weight by confidence
            weights = confidences / (confidences.sum() + 1e-8)
            
        ensemble_prob = float(np.average(probs, weights=weights))
        
        # Aggregate indicators
        all_indicators = {}
        for w in warnings_list:
            for k, v in w.indicators.items():
                all_indicators[f"{w.method}_{k}"] = v
                
        # Determine consensus
        high_risk_count = sum(1 for w in warnings_list if w.probability > 0.7)
        
        if high_risk_count >= 2:
            recommendation = "CONSENSUS ALERT: Multiple detection methods indicate high collapse risk!"
            time_estimate = min([w.time_to_collapse for w in warnings_list if w.time_to_collapse is not None] or [10])
        elif ensemble_prob > 0.6:
            recommendation = "Elevated risk detected by ensemble. Recommend intervention."
            time_estimate = 15
        else:
            recommendation = "Ensemble indicates stable system."
            time_estimate = None
            
        return CollapseWarning(
            probability=ensemble_prob,
            method='ensemble',
            confidence=float(np.mean(confidences)),
            time_to_collapse=time_estimate,
            indicators=all_indicators,
            recommendation=recommendation
        )
