"""ONNX policy loader and inference."""

import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    """Loads and runs an ONNX policy exported from rsl_rl training.

    Observation normalization is baked into the ONNX graph, so raw
    observations can be passed directly.
    """

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run policy inference.

        Args:
            obs: (106,) observation vector.

        Returns:
            actions: (29,) raw action output (before scaling).
        """
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        result = self.session.run(
            [self.output_name],
            {self.input_name: obs_batch},
        )
        return result[0].flatten()
