from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class HealthScoreCallback(BaseCallback):
    """Logs custom metrics: health score, dropout rate, budget."""

    def __init__(self):
        super().__init__(verbose=0)
        self._health_scores = []

    def _on_step(self) -> bool:
        # infos is a list of dicts, one per parallel env
        for info in self.locals.get("infos", []):
            if "health_score" in info:
                self._health_scores.append(info["health_score"])

        if len(self._health_scores) >= 100:
            mean_h = np.mean(self._health_scores[-100:])
            self.logger.record("custom/mean_health_score", mean_h)
            self.logger.record("custom/health_score_std", np.std(self._health_scores[-100:]))

        return True  # continue training