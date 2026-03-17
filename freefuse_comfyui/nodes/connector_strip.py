"""
FreeFuse Connector Input Strip - A vertical pass-through node for easy workflow connections.

Provides inputs and outputs for connecting FreeFuse workflows to other
LTX-Video or ComfyUI workflows.
"""


class FreeFuseConnectorInputStrip:
    """
    A connector strip with named inputs and outputs.
    
    Acts as a pass-through node to organize workflow connections.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_in": ("MODEL", {"default": None}),
                "clip_in": ("CLIP", {"default": None}),
                "latent_in": ("LATENT", {"default": None}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "LATENT")
    RETURN_NAMES = ("model_in", "clip_in", "latent_in")
    FUNCTION = "pass_through"
    CATEGORY = "FreeFuse/utils"
    OUTPUT_NODE = True

    DESCRIPTION = """A connector strip for organizing workflow connections.

Inputs/Outputs: model_in, clip_in, latent_in

Use this to cleanly connect FreeFuse workflows to other LTX-Video
or ComfyUI workflows."""

    def pass_through(self, model_in, clip_in, latent_in):
        """Pass all inputs straight to outputs."""
        return (model_in, clip_in, latent_in)


class FreeFuseConnectorOutputStrip:
    """
    A connector strip for output connections.
    
    Acts as a pass-through node to organize workflow connections.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_out": ("MODEL",),
                "prompt_out": ("*",),
            }
        }

    RETURN_TYPES = ("MODEL", "*")
    RETURN_NAMES = ("model_out", "prompt_out")
    FUNCTION = "pass_through"
    CATEGORY = "FreeFuse/utils"
    OUTPUT_NODE = True

    DESCRIPTION = """A connector strip for organizing workflow connections.

Inputs/Outputs: model_out, prompt_out

Use this to cleanly connect FreeFuse workflows to other LTX-Video
or ComfyUI workflows."""

    def pass_through(self, model_out, prompt_out):
        """Pass all inputs straight to outputs."""
        return (model_out, prompt_out)


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseConnectorInputStrip": FreeFuseConnectorInputStrip,
    "FreeFuseConnectorOutputStrip": FreeFuseConnectorOutputStrip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseConnectorInputStrip": "FreeFuse Connector Input Strip",
    "FreeFuseConnectorOutputStrip": "FreeFuse Connector Output Strip",
}
