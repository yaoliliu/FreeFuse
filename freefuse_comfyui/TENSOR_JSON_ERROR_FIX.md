# Fix for "Object of type Tensor is not JSON serializable" Error

## Problem

When saving images with FreeFuse, you get this error:
```
TypeError: Object of type Tensor is not JSON serializable
```

## Root Cause

ComfyUI's SaveImage node automatically stores **all node inputs/outputs** in workflow metadata (`extra_pnginfo`). When the image is saved, this metadata is serialized to JSON.

The `FREEFUSE_MASKS` data type contains PyTorch tensors, which cannot be JSON-serialized. If `FREEFUSE_MASKS` is connected to any node in your workflow, its tensor data gets stored and causes the serialization error.

## Solution

### Option 1: Fix Your Workflow (RECOMMENDED)

Check your workflow connections. The `FREEFUSE_MASKS` output should **ONLY** be connected to:
- Other FreeFuse nodes that expect `FREEFUSE_MASKS` input
- Preview/debug nodes that don't pass data to SaveImage

**DO NOT connect `FREEFUSE_MASKS` to:**
- SaveImage node inputs
- Any node that passes data to SaveImage
- STRING inputs (ComfyUI stores all string inputs in metadata)

### Typical Problematic Pattern

```
FreeFusePhase1Sampler (FREEFUSE_MASKS output) 
    → SomeNode (STRING output) 
    → SaveImage (pnginfo input)
```

This causes the tensor data to be stored in PNG metadata.

### Correct Pattern

```
FreeFusePhase1Sampler (FREEFUSE_MASKS output) 
    → FreeFuseMaskApplicator (FREEFUSE_MASKS input)
    → Model processing...
    
FreeFusePhase1Sampler (preview IMAGE output)
    → Preview node (for display only, not saved)
```

### Option 2: Use the Serialization Utility

If you need to store FreeFuse data in metadata (for debugging), use the new serialization utility:

```python
from freefuse_core import make_freefuse_masks_json_serializable, safe_json_dumps

# Convert masks to JSON-serializable format
serializable_masks = make_freefuse_masks_json_serializable(masks_data)

# Or use the safe JSON dumps function
json_string = safe_json_dumps(masks_data)
```

### Option 3: Disable Metadata Saving

If you don't need workflow metadata in saved images:

1. In ComfyUI settings, disable "Save metadata in images"
2. Or use a custom save node that doesn't store metadata

## Checking Your Workflow

1. Open your workflow in ComfyUI
2. Look for any connections from `FREEFUSE_MASKS` outputs
3. Trace where those connections lead
4. If they reach SaveImage, disconnect them

## Common Mistakes

1. **Connecting FREEFUSE_MASKS to a debug node's STRING output**
   - The debug node converts tensor info to text, but the original tensors might still be stored
   
2. **Connecting FREEFUSE_DATA with tensor-containing fields**
   - FREEFUSE_DATA should only contain strings, lists, and dicts
   - If you add tensor data to it, it will fail serialization

3. **Using custom nodes that store all inputs**
   - Some custom nodes store all inputs in their output metadata
   - Avoid connecting FreeFuse tensor outputs to these nodes

## Technical Details

The error occurs in ComfyUI's `nodes.py`:
```python
metadata.add_text(x, json.dumps(extra_pnginfo[x]))
```

ComfyUI stores all node inputs in `extra_pnginfo`. When `FREEFUSE_MASKS` contains tensors, `json.dumps()` fails.

## Files Modified

- `freefuse_core/json_serialization.py` - New utility for JSON serialization
- `freefuse_core/mask_utils.py` - Added `make_freefuse_masks_json_serializable()`
- `freefuse_core/__init__.py` - Exported serialization utilities

## Still Having Issues?

1. Check your workflow connections carefully
2. Look for any path from FreeFuse outputs to SaveImage inputs
3. Consider using the standard FreeFuse workflow template (`FreeFuse-standard.json`)
4. Report the issue with your workflow JSON attached
