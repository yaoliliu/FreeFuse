"""
FreeFuse mask editing utility nodes.

These nodes provide an optional debug/advanced path:
- FreeFuseMaskTap: expose/edit per-concept masks through MaskEditor image refs
- FreeFuseMaskReassemble: normalize and forward edited mask bank
"""

import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import folder_paths
except Exception:
    folder_paths = None


_REF_WITH_TYPE_RE = re.compile(r"^(?P<path>.+?)(?:\s*\[(?P<type>[a-zA-Z0-9_]+)\])?$")
_MASK_BINARY_THRESHOLD = 0.5
_MASK_SIGNAL_EPS = 1e-6
_DEFAULT_EDITOR_REF_PREFIXES = (
    "freefuse-masktap-slot-",
    "freefuse-masktap-blank-",
)
_USER_EDIT_REF_PREFIX = "freefuse-masktap-edit-"
_USER_EDIT_SEED_RE = re.compile(r"^freefuse-masktap-edit-s(?P<seed>\d+)-")


def _to_2d_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Normalize mask tensor to 2D float32 [H, W]."""
    if mask is None or not isinstance(mask, torch.Tensor):
        return None
    if mask.dim() == 2:
        return mask.float()
    if mask.dim() == 3:
        return mask[0].float()
    if mask.dim() == 4:
        return mask[0, 0].float()
    return None


def _mask_hw(mask: Optional[torch.Tensor]) -> Optional[Tuple[int, int]]:
    m = _to_2d_mask(mask)
    if m is None:
        return None
    return int(m.shape[0]), int(m.shape[1])


def _resize_2d(
    mask_2d: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    if mask_2d.shape[0] == target_h and mask_2d.shape[1] == target_w:
        return mask_2d
    kwargs = {
        "size": (target_h, target_w),
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(mask_2d.unsqueeze(0).unsqueeze(0), **kwargs).squeeze(0).squeeze(0)


def _binarize_2d(mask_2d: torch.Tensor, threshold: float = _MASK_BINARY_THRESHOLD) -> torch.Tensor:
    return (mask_2d.float() >= float(threshold)).float()


def _parse_image_ref(value) -> Tuple[Optional[str], Optional[str]]:
    """Parse image-ref style widget values into path + type."""
    if value is None:
        return None, None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None
        match = _REF_WITH_TYPE_RE.match(text)
        if not match:
            return text, None
        path_like = match.group("path").strip()
        ref_type = match.group("type")
        return path_like or None, ref_type

    if isinstance(value, dict):
        filename = value.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return None, None
        subfolder = value.get("subfolder")
        path_like = filename.strip()
        if isinstance(subfolder, str) and subfolder.strip():
            path_like = os.path.join(subfolder.strip(), path_like)
        ref_type = value.get("type")
        if not isinstance(ref_type, str):
            ref_type = None
        return path_like, ref_type

    return None, None


def _is_default_editor_ref(value) -> bool:
    if value is None:
        return False

    filename: Optional[str] = None
    if isinstance(value, dict):
        raw = value.get("filename")
        if isinstance(raw, str):
            filename = raw
    elif isinstance(value, str):
        path_like, _ = _parse_image_ref(value)
        if isinstance(path_like, str):
            filename = os.path.basename(path_like)

    if not filename:
        return False
    name = filename.strip().lower()
    if not name:
        return False
    return any(name.startswith(prefix) for prefix in _DEFAULT_EDITOR_REF_PREFIXES)


def _extract_ref_filename(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("filename")
        if isinstance(raw, str):
            text = raw.strip()
            return text or None
        return None
    if isinstance(value, str):
        path_like, _ = _parse_image_ref(value)
        if isinstance(path_like, str):
            text = os.path.basename(path_like).strip()
            return text or None
    return None


def _extract_user_edit_seed(value) -> Optional[int]:
    filename = _extract_ref_filename(value)
    if not filename:
        return None
    match = _USER_EDIT_SEED_RE.match(filename.lower())
    if not match:
        return None
    try:
        return int(match.group("seed"))
    except Exception:
        return None


def _is_user_edit_ref(value) -> bool:
    filename = _extract_ref_filename(value)
    if not filename:
        return False
    return filename.lower().startswith(_USER_EDIT_REF_PREFIX)


def _extract_phase1_seed(mask_bank: Dict) -> Optional[int]:
    if not isinstance(mask_bank, dict):
        return None
    metadata = mask_bank.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("phase1_seed")
    try:
        if raw is None:
            return None
        return int(raw)
    except Exception:
        return None


def _should_ignore_edited_ref(image_ref, current_seed: Optional[int]) -> bool:
    if _is_default_editor_ref(image_ref):
        return True
    if not _is_user_edit_ref(image_ref):
        return False
    if current_seed is None:
        return False

    # Legacy edited refs without seed tag should not carry across seeded reruns.
    ref_seed = _extract_user_edit_seed(image_ref)
    if ref_seed is None:
        return True
    return ref_seed != int(current_seed)


def _resolve_image_ref_path(path_like: Optional[str], ref_type: Optional[str]) -> Optional[str]:
    if not isinstance(path_like, str) or not path_like.strip():
        return None

    candidate = path_like.strip()

    if os.path.isabs(candidate) and os.path.isfile(candidate):
        return candidate

    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    if folder_paths is None:
        return None

    try:
        annotated = candidate
        if ref_type in {"input", "output", "temp"}:
            annotated = f"{candidate} [{ref_type}]"
        return folder_paths.get_annotated_filepath(annotated)
    except Exception:
        return None


def _load_mask_from_image_ref(
    image_ref,
    target_h: int,
    target_w: int,
) -> Optional[torch.Tensor]:
    if Image is None:
        return None

    path_like, ref_type = _parse_image_ref(image_ref)
    resolved_path = _resolve_image_ref_path(path_like, ref_type)
    if resolved_path is None or not os.path.isfile(resolved_path):
        return None

    try:
        with Image.open(resolved_path) as img:
            gray = np.array(img.convert("L")).astype(np.float32) / 255.0
            mask_2d = torch.from_numpy(gray)
            if "A" in img.getbands():
                alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                alpha_mask = 1.0 - torch.from_numpy(alpha)
                # Prefer visible grayscale content. Fallback to alpha when RGB is flat.
                if float(np.var(gray)) <= _MASK_SIGNAL_EPS and float(np.var(alpha)) > _MASK_SIGNAL_EPS:
                    mask_2d = alpha_mask
        mask_2d = mask_2d.float()
        if mask_2d.dim() != 2:
            return None
        mask_2d = _resize_2d(mask_2d, target_h, target_w, mode="nearest")
        return _binarize_2d(mask_2d)
    except Exception as e:
        print(f"[FreeFuseMaskTap] Warning: failed to load edited mask '{resolved_path}': {e}")
        return None


def _format_2d_for_target(mask_2d: torch.Tensor, target_mask: Optional[torch.Tensor]) -> torch.Tensor:
    m2d = mask_2d.float()
    target_hw = _mask_hw(target_mask)
    if target_hw is not None:
        m2d = _resize_2d(m2d, target_hw[0], target_hw[1])

    if isinstance(target_mask, torch.Tensor):
        m2d = m2d.to(device=target_mask.device, dtype=torch.float32)
        if target_mask.dim() == 2:
            return m2d
        if target_mask.dim() == 3:
            return m2d.unsqueeze(0)
        if target_mask.dim() == 4:
            return m2d.unsqueeze(0).unsqueeze(0)
    return m2d


def _build_mask_editor_ui_images(slot_masks_2d: List[torch.Tensor]) -> List[Dict[str, str]]:
    if folder_paths is None or Image is None:
        return []

    try:
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
    except Exception:
        return []

    stamp = uuid.uuid4().hex[:12]
    refs: List[Dict[str, str]] = []
    for idx, mask in enumerate(slot_masks_2d):
        m = _binarize_2d(mask).clamp(0.0, 1.0).cpu().numpy()
        h, w = m.shape[0], m.shape[1]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        gray = np.clip(m * 255.0, 0, 255).astype(np.uint8)
        rgba[:, :, 0] = gray
        rgba[:, :, 1] = gray
        rgba[:, :, 2] = gray
        rgba[:, :, 3] = np.clip((1.0 - m) * 255.0, 0, 255).astype(np.uint8)

        filename = f"freefuse-masktap-slot-{stamp}-{idx:02d}.png"
        path = os.path.join(temp_dir, filename)
        Image.fromarray(rgba, mode="RGBA").save(path, compress_level=1)
        refs.append({"filename": filename, "subfolder": "", "type": "temp"})

    return refs


class FreeFuseMaskTap:
    """
    Prepare deterministic per-adapter mask slots for interactive editing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_bank": ("FREEFUSE_MASKS",),
            },
            "optional": {
                "freefuse_data": ("FREEFUSE_DATA",),
                "mask_image_00": ("STRING", {"default": "", "multiline": False}),
                "mask_image_01": ("STRING", {"default": "", "multiline": False}),
                "mask_image_02": ("STRING", {"default": "", "multiline": False}),
                "mask_image_03": ("STRING", {"default": "", "multiline": False}),
                "mask_image_04": ("STRING", {"default": "", "multiline": False}),
                "mask_image_05": ("STRING", {"default": "", "multiline": False}),
                "mask_image_06": ("STRING", {"default": "", "multiline": False}),
                "mask_image_07": ("STRING", {"default": "", "multiline": False}),
                "mask_image_08": ("STRING", {"default": "", "multiline": False}),
                "mask_image_09": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "FREEFUSE_MASKS",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "mask_bank",
        "slot_names",
        "slot_mask_images",
    )
    FUNCTION = "tap_masks"
    CATEGORY = "FreeFuse/Utils"

    @staticmethod
    def _resolve_order(
        mask_dict: Dict[str, torch.Tensor],
        freefuse_data: Optional[Dict],
        metadata_names: Optional[List[str]] = None,
    ) -> List[str]:
        ordered: List[str] = []
        seen = set()

        # 1) metadata adapter order (if present)
        if isinstance(metadata_names, list):
            for name in metadata_names:
                if isinstance(name, str) and name in mask_dict and name not in seen:
                    ordered.append(name)
                    seen.add(name)

        # 2) adapter order from freefuse_data
        adapters = []
        if isinstance(freefuse_data, dict):
            adapters = freefuse_data.get("adapters", [])
        if isinstance(adapters, list):
            for adapter in adapters:
                if isinstance(adapter, dict):
                    name = adapter.get("name")
                else:
                    name = adapter
                if isinstance(name, str) and name in mask_dict and name not in seen:
                    ordered.append(name)
                    seen.add(name)

        # 3) remaining keys in deterministic order
        for name in sorted(mask_dict.keys()):
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        return ordered

    def tap_masks(self, mask_bank, freefuse_data=None, **kwargs):
        if not isinstance(mask_bank, dict):
            mask_bank = {"masks": {}, "similarity_maps": {}}

        mask_dict = mask_bank.get("masks", {})
        if not isinstance(mask_dict, dict):
            mask_dict = {}
        new_mask_bank = dict(mask_bank)
        new_masks = dict(mask_dict)

        metadata = mask_bank.get("metadata", {}) if isinstance(mask_bank.get("metadata", {}), dict) else {}
        metadata_names = metadata.get("adapter_names", [])
        ordered_names = self._resolve_order(
            mask_dict,
            freefuse_data if isinstance(freefuse_data, dict) else None,
            metadata_names if isinstance(metadata_names, list) else None,
        )
        current_seed = _extract_phase1_seed(mask_bank)

        # Reference size for empty slots.
        ref_h, ref_w = 64, 64
        for mask in mask_dict.values():
            hw = _mask_hw(mask)
            if hw is not None:
                ref_h, ref_w = hw
                break

        slot_masks: List[torch.Tensor] = []
        slot_images: List[torch.Tensor] = []
        slot_lines: List[str] = []
        for i in range(10):
            adapter_name: Optional[str] = None
            target_mask: Optional[torch.Tensor] = None
            if i < len(ordered_names):
                adapter_name = ordered_names[i]
                target_mask = mask_dict.get(adapter_name)
                slot_lines.append(f"{i:02d}:{adapter_name}")
                m2d = _to_2d_mask(target_mask)
                if m2d is None:
                    m2d = torch.zeros((ref_h, ref_w), dtype=torch.float32)
            else:
                slot_lines.append(f"{i:02d}:")
                m2d = torch.zeros((ref_h, ref_w), dtype=torch.float32)

            edited_ref = kwargs.get(f"mask_image_{i:02d}")
            if _should_ignore_edited_ref(edited_ref, current_seed):
                edited_ref = None
            edited_mask = _load_mask_from_image_ref(
                image_ref=edited_ref,
                target_h=int(m2d.shape[0]),
                target_w=int(m2d.shape[1]),
            )
            if edited_mask is not None:
                m2d = edited_mask
                if adapter_name is not None:
                    new_masks[adapter_name] = _format_2d_for_target(m2d, target_mask)

            slot_masks.append(m2d)
            slot_m2d = _binarize_2d(m2d)
            img = (
                slot_m2d.float()
                .clamp(0.0, 1.0)
                .detach()
                .cpu()
                .unsqueeze(-1)
                .repeat(1, 1, 3)
            )
            slot_images.append(img)

        slot_names = "\n".join(slot_lines)
        new_mask_bank["masks"] = new_masks
        metadata = new_mask_bank.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["adapter_names"] = ordered_names
        new_mask_bank["metadata"] = metadata

        slot_mask_images = torch.stack(slot_images, dim=0)
        result = (new_mask_bank, slot_names, slot_mask_images)
        ui_images = _build_mask_editor_ui_images(slot_masks)
        if ui_images:
            return {"ui": {"images": ui_images}, "result": result}
        return result


class FreeFuseMaskReassemble:
    """
    Merge edited masks back into FREEFUSE_MASKS.

    Legacy mask_00..mask_09 kwargs are still accepted for compatibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_mask_bank": ("FREEFUSE_MASKS",),
            },
            "optional": {
                "freefuse_data": ("FREEFUSE_DATA",),
                "slot_names": ("STRING", {"default": ""}),
                "clamp_01": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("FREEFUSE_MASKS",)
    RETURN_NAMES = ("new_mask_bank",)
    FUNCTION = "reassemble_masks"
    CATEGORY = "FreeFuse/Utils"

    @staticmethod
    def _parse_slot_names(slot_names: str) -> List[str]:
        names: List[str] = []
        if not isinstance(slot_names, str) or not slot_names.strip():
            return names
        for line in slot_names.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                _, rhs = line.split(":", 1)
                rhs = rhs.strip()
                if rhs:
                    names.append(rhs)
            else:
                names.append(line)
        return names

    @staticmethod
    def _resolve_order(
        original_masks: Dict[str, torch.Tensor],
        original_bank: Dict,
        freefuse_data: Optional[Dict],
        slot_names: str,
    ) -> List[str]:
        names: List[str] = []
        seen = set()

        for name in FreeFuseMaskReassemble._parse_slot_names(slot_names):
            if name not in seen:
                names.append(name)
                seen.add(name)

        metadata = original_bank.get("metadata", {})
        if isinstance(metadata, dict):
            for name in metadata.get("adapter_names", []) or []:
                if isinstance(name, str) and name in original_masks and name not in seen:
                    names.append(name)
                    seen.add(name)

        if isinstance(freefuse_data, dict):
            adapters = freefuse_data.get("adapters", [])
            if isinstance(adapters, list):
                for adapter in adapters:
                    if isinstance(adapter, dict):
                        name = adapter.get("name")
                    else:
                        name = adapter
                    if isinstance(name, str) and name in original_masks and name not in seen:
                        names.append(name)
                        seen.add(name)

        for name in original_masks.keys():
            if name not in seen:
                names.append(name)
                seen.add(name)

        return names

    @staticmethod
    def _format_for_target(
        edited_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor],
        clamp_01: bool,
    ) -> torch.Tensor:
        m2d = _to_2d_mask(edited_mask)
        if m2d is None:
            raise ValueError("Edited mask is not a valid tensor.")

        target_hw = _mask_hw(target_mask)
        if target_hw is not None:
            m2d = _resize_2d(m2d, target_hw[0], target_hw[1])

        if clamp_01:
            m2d = m2d.clamp(0.0, 1.0)

        if isinstance(target_mask, torch.Tensor):
            if target_mask.dim() == 2:
                return m2d
            if target_mask.dim() == 3:
                return m2d.unsqueeze(0)
            if target_mask.dim() == 4:
                return m2d.unsqueeze(0).unsqueeze(0)
        return m2d

    def reassemble_masks(
        self,
        original_mask_bank,
        freefuse_data=None,
        slot_names="",
        clamp_01=True,
        **kwargs,
    ):
        if not isinstance(original_mask_bank, dict):
            original_mask_bank = {"masks": {}, "similarity_maps": {}}

        new_mask_bank = dict(original_mask_bank)
        original_masks = original_mask_bank.get("masks", {})
        if not isinstance(original_masks, dict):
            original_masks = {}
        new_masks = dict(original_masks)

        ordered_names = self._resolve_order(
            original_masks=original_masks,
            original_bank=original_mask_bank,
            freefuse_data=freefuse_data if isinstance(freefuse_data, dict) else None,
            slot_names=slot_names,
        )

        for i in range(10):
            key = f"mask_{i:02d}"
            edited = kwargs.get(key)
            if edited is None or i >= len(ordered_names):
                continue

            adapter_name = ordered_names[i]
            target_mask = original_masks.get(adapter_name)
            try:
                formatted = self._format_for_target(
                    edited_mask=edited,
                    target_mask=target_mask,
                    clamp_01=bool(clamp_01),
                )
                new_masks[adapter_name] = formatted
            except Exception as e:
                print(f"[FreeFuseMaskReassemble] Warning: failed to apply {key} -> {adapter_name}: {e}")

        new_mask_bank["masks"] = new_masks

        metadata = new_mask_bank.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["adapter_names"] = ordered_names
        new_mask_bank["metadata"] = metadata

        return (new_mask_bank,)


NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskTap": FreeFuseMaskTap,
    "FreeFuseMaskReassemble": FreeFuseMaskReassemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskTap": "FreeFuse Mask Tap",
    "FreeFuseMaskReassemble": "FreeFuse Mask Reassemble",
}
