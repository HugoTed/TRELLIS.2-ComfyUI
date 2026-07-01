from trellis2_client.worker_client import preload_model


class Trellis2LoadModel:
    """Preload TRELLIS.2 weights into the worker GPU memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "load"
    CATEGORY = "3d/trellis2"

    def load(self):
        result = preload_model()
        return (f"Model loaded: {result.get('model_id', 'unknown')}",)
