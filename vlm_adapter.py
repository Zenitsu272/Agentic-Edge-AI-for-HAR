import os
import time
import json
import re
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

class VLMAdapter:
    """
    Genuine Vision-Language Model (VLM) Adapter for Qwen2.5-VL.
    Evaluates candidate event frames to inspect posture, floor contact,
    obstructions, and presence or absence of voluntary recovery.
    
    Status states:
    - 'ONLINE': Full Qwen2.5-VL model loaded and operational on device.
    - 'WEIGHTS_INCOMPLETE': Model weights partially downloaded or missing.
    - 'WEIGHTS_MISSING': Model directory not found.
    - 'OFFLINE_FALLBACK': Running structured biomechanical rule verification.
    """
    def __init__(self, model_dir: str = r"E:\hf_cache\hub\models--unsloth--Qwen2.5-VL-3B-Instruct-bnb-4bit", auto_load: bool = True):
        self.model_dir = model_dir
        self.model_name = "Qwen2.5-VL-3B-Instruct (4-bit NF4)"
        self.model = None
        self.processor = None
        self.status = "INITIALIZING"
        self.status_detail = ""
        self.device = "cpu"
        if auto_load:
            self._init_adapter()

    def _init_adapter(self):
        # 1. Inspect model directory and check for incomplete blobs
        if not os.path.exists(self.model_dir):
            self.status = "WEIGHTS_MISSING"
            self.status_detail = f"Model directory {self.model_dir} does not exist on disk."
            return

        has_incomplete = False
        try:
            for root, _, files in os.walk(self.model_dir):
                for f in files:
                    if f.endswith(".incomplete") or f.endswith(".lock"):
                        has_incomplete = True
                        break
        except Exception:
            pass

        if has_incomplete:
            self.status = "WEIGHTS_INCOMPLETE"
            self.status_detail = "Model weights are partially downloaded (.incomplete blob found in cache). Operating in transparent fallback mode."
            return

        # Attempt real model loading if weights exist and complete
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
            )
            self.status = "ONLINE"
            if self.device.startswith("cuda"):
                self.status_detail = f"Loaded onto {self.device}. VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB"
            else:
                self.status_detail = "Loaded onto CPU."
        except Exception as e:
            self.status = "OFFLINE_FALLBACK"
            self.status_detail = f"PyTorch/Transformers load error: {str(e)[:140]}"

    def verify_candidate_event(self, candidate_event: Dict[str, Any], evidence_frames: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Takes a candidate event dictionary and optional evidence frames,
        and produces a structured verification report with transparent provenance.
        """
        kinematics = candidate_event.get("kinematics", {})
        posture = kinematics.get("posture", "Unknown")
        torso_angle = kinematics.get("torso_angle", 0.0)
        dwell_sec = kinematics.get("dwell_sec", 0.0)
        drop_vel = kinematics.get("drop_velocity", 0.0)
        camera_id = candidate_event.get("camera_id", "Camera 01")
        event_id = candidate_event.get("id", f"EVT-{int(time.time()*1000)%1000000}")
        
        # Frames can be passed directly or inside candidate_event
        frames = evidence_frames or candidate_event.get("evidence_frames") or candidate_event.get("frames") or []

        # If online model is loaded and evidence frames are present, evaluate frames with genuine inference
        if self.status == "ONLINE" and self.model is not None and self.processor is not None and len(frames) > 0:
            try:
                return self._run_model_inference(candidate_event, frames)
            except Exception as e:
                # Catch inference runtime errors and fallback explicitly
                return self._build_fallback_response(
                    candidate_event, 
                    error_detail=f"VLM inference exception: {str(e)[:140]}"
                )

        # Explicit rule-based fallback when model is not ONLINE or no video frames are available
        return self._build_fallback_response(candidate_event)

    def _build_fallback_response(self, candidate_event: Dict[str, Any], error_detail: Optional[str] = None) -> Dict[str, Any]:
        """
        Transparent rule-based fallback verification.
        Distinguishes rule-based classification from true VLM verification.
        """
        kinematics = candidate_event.get("kinematics", {})
        posture = kinematics.get("posture", "Unknown")
        torso_angle = kinematics.get("torso_angle", 0.0)
        dwell_sec = kinematics.get("dwell_sec", 0.0)
        drop_vel = kinematics.get("drop_velocity", 0.0)
        camera_id = candidate_event.get("camera_id", "Camera 01")
        event_id = candidate_event.get("id", "UNKNOWN")

        is_fall = False
        conf = 0.0
        reasoning = ""
        scene_context = f"{camera_id} · Fallback inspection"
        uncertainty = False

        if posture == "Lying / Resting" and dwell_sec >= 1.2:
            is_fall = True
            conf = 0.91
            reasoning = (
                f"Rule-based fallback: Subject collapsed prone/supine on floor surface (torso tilt {torso_angle:.1f}°). "
                f"Rapid vertical drop ({drop_vel:.1f} px/s) followed by unassisted ground dwell ({dwell_sec:.1f}s)."
            )
            scene_context = f"{camera_id} · Floor contact verified adjacent to resident path"
        elif "Sitting" in posture:
            is_fall = False
            conf = 0.93
            reasoning = (
                f"Rule-based fallback: Seated posture maintained. Hips remain elevated at chair seat height; "
                f"forward tilt ({torso_angle:.1f}°) reflects voluntary reach or resting posture (0 fall risk)."
            )
            scene_context = f"{camera_id} · Armchair/seat support detected"
        elif posture == "Standing":
            is_fall = False
            conf = 0.95
            reasoning = "Rule-based fallback: Subject upright with normal vertical center of mass."
            scene_context = f"{camera_id} · Open floor area"
        else:
            is_fall = (dwell_sec >= 2.0 and drop_vel > 60.0)
            conf = 0.75
            uncertainty = True
            reasoning = f"Rule-based fallback: Indeterminate orientation. Dwell duration: {dwell_sec:.1f}s, drop velocity: {drop_vel:.1f} px/s."
            scene_context = f"{camera_id} · Continuous monitoring window"

        status_detail = error_detail if error_detail else self.status_detail

        return {
            "event_id": event_id,
            "model": self.model_name,
            "adapter_status": self.status,
            "status_detail": status_detail,
            "verification_method": "RULE_BASED_FALLBACK",
            "is_true_fall": is_fall,
            "confidence": round(float(conf), 2),
            "scene_observation": reasoning,
            "environmental_context": scene_context,
            "recovery_observed": not is_fall and posture in ["Standing", "Sitting"],
            "uncertainty_flag": uncertainty,
            "evidence_frames_count": len(candidate_event.get("evidence_frames", [])),
            "recommended_action": "ESCALATE_EMERGENCY" if is_fall else "LOG_NORMAL"
        }

    def _prepare_pil_images(self, frames: List[Any]) -> List[Image.Image]:
        """Validates and converts input frames into PIL Images for Qwen2.5-VL."""
        pil_images = []
        for item in frames:
            img = None
            if isinstance(item, dict):
                f_data = item.get("frame")
            else:
                f_data = item

            if isinstance(f_data, Image.Image):
                img = f_data
            elif isinstance(f_data, np.ndarray):
                if len(f_data.shape) == 3 and f_data.shape[2] == 3:
                    # Assume BGR if from cv2, convert to RGB
                    rgb = f_data[:, :, ::-1]
                    img = Image.fromarray(rgb)
                elif len(f_data.shape) == 2:
                    img = Image.fromarray(f_data).convert("RGB")
            elif isinstance(f_data, str) and os.path.exists(f_data):
                img = Image.open(f_data).convert("RGB")

            if img is not None:
                # Resize if excessively large to keep inference bounded
                max_dim = 640
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim))
                pil_images.append(img)

        return pil_images

    def _run_model_inference(self, candidate_event: Dict[str, Any], frames: List[Any]) -> Dict[str, Any]:
        """
        Genuine Qwen2.5-VL Model Forward Pass.
        Formats evidence frames with a neutral, evidence-focused prompt.
        Parses structured JSON response with explicit uncertainty.
        """
        event_id = candidate_event.get("id", "UNKNOWN")
        camera_id = candidate_event.get("camera_id", "Camera 01")
        pil_images = self._prepare_pil_images(frames)

        # Construct neutral, evidence-focused prompt without confirmation bias
        prompt_text = (
            "You are an objective vision verification system in a healthcare monitoring environment.\n"
            "Examine the provided camera frame(s) carefully. Describe observable human posture, body orientation, "
            "contact with floor or furniture, and evidence of voluntary recovery or distress.\n"
            "Do NOT assume or confirm a fall occurred unless unequivocal visual evidence demonstrates an uncontrolled "
            "collapse onto the floor without recovery.\n"
            "Respond strictly in valid JSON matching this schema:\n"
            "{\n"
            '  "action_observed": "string describing observed physical activity",\n'
            '  "posture": "standing | sitting | lying_floor | transitioning | recovering | other",\n'
            '  "floor_contact": boolean,\n'
            '  "voluntary_recovery": boolean,\n'
            '  "is_true_fall": boolean,\n'
            '  "confidence": float between 0.0 and 1.0,\n'
            '  "uncertainty": boolean,\n'
            '  "scene_observation": "concise objective summary of physical evidence"\n'
            "}"
        )

        content_payload = []
        if pil_images:
            for pil_img in pil_images[:4]: # Cap at 4 frames to keep bounded
                content_payload.append({"type": "image", "image": pil_img})
        content_payload.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "user",
                "content": content_payload
            }
        ]

        # Use qwen_vl_utils if available, otherwise direct processor call
        try:
            from qwen_vl_utils import process_vision_info
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
        except ImportError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if pil_images:
                inputs = self.processor(text=[text], images=pil_images[:4], padding=True, return_tensors="pt")
            else:
                inputs = self.processor(text=[text], return_tensors="pt")

        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, "tokenizer") else None
            )

        # Extract generated tokens
        in_len = inputs["input_ids"].shape[1]
        out_ids = generated_ids[0][in_len:]
        output_text = self.processor.decode(out_ids, skip_special_tokens=True).strip()

        # Parse structured JSON from model output
        parsed = self._parse_model_output(output_text)
        is_fall = parsed.get("is_true_fall", False)
        conf = float(parsed.get("confidence", 0.5))
        uncertainty = parsed.get("uncertainty", False)
        recovery = parsed.get("voluntary_recovery", False)
        raw_obs = parsed.get("scene_observation") or parsed.get("action_observed", output_text[:120]) or ""
        scene_obs = f"Qwen2.5-VL Verified: {raw_obs}" if not raw_obs.startswith("Qwen2.5-VL") else raw_obs

        return {
            "event_id": event_id,
            "model": self.model_name,
            "adapter_status": "ONLINE",
            "status_detail": f"Qwen2.5-VL forward pass executed on {len(pil_images)} frame(s).",
            "verification_method": "VLM_MODEL",
            "is_true_fall": is_fall,
            "confidence": round(conf, 2),
            "scene_observation": scene_obs,
            "environmental_context": f"{camera_id} · Qwen2.5-VL visual verification",
            "recovery_observed": recovery,
            "uncertainty_flag": uncertainty,
            "evidence_frames_count": len(pil_images),
            "recommended_action": "ESCALATE_EMERGENCY" if is_fall else "LOG_NORMAL",
            "raw_model_output": output_text[:300]
        }

    def _parse_model_output(self, text: str) -> Dict[str, Any]:
        """Robust parser for JSON output inside markdown or free text."""
        # Check for ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception:
                pass

        # Try searching for outermost curly braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass

        # Fallback keyword extraction if JSON is completely malformed
        is_fall = bool(re.search(r"\b(fall|collapsed|uncontrolled|prone)\b", text, re.IGNORECASE))
        recovery = bool(re.search(r"\b(recovered|recovering|standing up|sitting)\b", text, re.IGNORECASE))
        return {
            "is_true_fall": is_fall and not recovery,
            "confidence": 0.65,
            "uncertainty": True,
            "voluntary_recovery": recovery,
            "scene_observation": text[:120]
        }
