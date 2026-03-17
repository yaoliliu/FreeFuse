"""
FreeFuse Named Reroute - A reroute node with an editable label.

Like ComfyUI's Reroute node but with a customizable name in the middle.
"""


class FreeFuseNamedReroute:
    """
    A reroute node with an editable label.
    
    Acts as a pass-through node with a custom name for organizing workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {"default": None}),
            },
            "optional": {
                "label": ("STRING", {"default": "", "placeholder": "enter label"}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "pass_through"
    CATEGORY = "FreeFuse/utils"
    OUTPUT_NODE = True

    DESCRIPTION = """A reroute node with an editable label.

Use this to organize and label connections in your workflow.
The label appears in the middle of the node for easy identification."""

    def pass_through(self, input, label=""):
        """Pass input straight to output."""
        return (input,)


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseNamedReroute": FreeFuseNamedReroute,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseNamedReroute": "FreeFuse Named Reroute",
}
