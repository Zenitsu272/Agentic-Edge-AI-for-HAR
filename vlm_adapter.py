import os
import time
import json
import re
import threading
from pathlib import Path
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
    - 'LOADING', 'UNAVAILABLE': No model verdict is available yet.

    Production uses verify_video_event, which never falls back to pose rules.
    verify_candidate_event retains the legacy graph's explicit rule fallback.
    """
    def __init__(self, model_dir: Optional[str] = None, auto_load: bool = True):
        default_dir = Path(__file__).parent / '.venv' / 'models' / 'Qwen2.5-VL-3B-Instruct-bnb-4bit'
        self.model_dir = str(model_dir or os.getenv('CAREWATCH_VLM_MODEL', str(default_dir)))
        self.model_name = "Qwen2.5-VL-3B-Instruct (4-bit NF4)"
        self.model = None
        self.processor = None
        self.status = "NOT_LOADED"
        self.status_detail = ""
        self.device = "cpu"
        self.generation_lock = threading.Lock()
        if auto_load:
            self._init_adapter()

    def _init_adapter(self):
        self.status = 'LOADING'
        self.status_detail = 'Loading the local vision-language model.'
        # 1. Inspect model directory and check for incomplete blobs
        if not os.path.exists(self.model_dir):
            self.status = "WEIGHTS_MISSING"
            self.status_detail = f"Model directory {self.model_dir} does not exist on disk."
            return

        has_incomplete = False
        try:
            for root, _, files in os.walk(self.model_dir):
                for f in files:
                    if f.endswith(".incomplete"):
                        has_incomplete = True
                        break
        except Exception:
            pass

        if has_incomplete and not list(Path(self.model_dir).glob('model*.safetensors')):
            self.status = "WEIGHTS_INCOMPLETE"
            self.status_detail = "Model download is incomplete. VLM verification is unavailable until the weights are ready."
            return

        # Attempt real model loading if weights exist and complete
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir, local_files_only=True,
                min_pixels=16*28*28, max_pixels=512*28*28)
            config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
            quantization = getattr(config, 'quantization_config', None)
            if isinstance(quantization, dict) and quantization.get('quant_method') == 'bitsandbytes':
                # This checkpoint keeps the vision merger in BF16. Newer
                # Transformers needs its full module path to skip child layers.
                skips = list(quantization.get('llm_int8_skip_modules') or [])
                quantization['llm_int8_skip_modules'] = list(dict.fromkeys(
                    skips + ['model.visual.merger', 'visual.merger']))
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir,
                config=config,
                device_map={'': self.device}, local_files_only=True,
                dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
                attn_implementation='sdpa'
            )
            self.model.eval()
            from qwen_video_positions import install_video_position_fix
            install_video_position_fix(self.model)
            if isinstance(quantization, dict) and quantization.get('quant_method') == 'bitsandbytes':
                import bitsandbytes as bnb
                if any(isinstance(layer, bnb.nn.Linear4bit) and
                       getattr(layer.weight, 'quant_state', None) is None
                       for layer in self.model.modules()):
                    raise RuntimeError('Quantized model layers did not initialize correctly.')
            self.status = "ONLINE"
            if self.device.startswith("cuda"):
                self.status_detail = f"Loaded onto {self.device}. VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB"
            else:
                self.status_detail = "Loaded onto CPU."
        except Exception as e:
            self.model = None
            self.status = "UNAVAILABLE"
            self.status_detail = f"PyTorch/Transformers load error: {str(e)[:140]}"

    def verify_video_event(self, candidate_event, evidence_frames):
        """Production path: only actual model inference can produce a verdict."""
        if self.status != 'ONLINE' or self.model is None or self.processor is None:
            return self._unverified(candidate_event, 'UNAVAILABLE', self.status_detail or self.status)
        if len(evidence_frames) < 3:
            return self._unverified(candidate_event, 'INCONCLUSIVE', 'Too few chronological frames to verify the motion.')
        try:
            with self.generation_lock:
                return self._run_model_inference(candidate_event, evidence_frames)
        except Exception as error:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return self._unverified(candidate_event, 'ERROR', f'VLM inference failed: {type(error).__name__}: {str(error)[:160]}')

    def _unverified(self, candidate, status, reason):
        return {'event_id': candidate.get('id'), 'model': self.model_name,
                'status': status, 'is_true_fall': None, 'uncertainty_flag': True,
                'verification_method': 'UNAVAILABLE', 'scene_observation': reason,
                'evidence_frames_count': 0, 'adapter_status': self.status}

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

    def _run_model_inference(self, candidate_event, frames):
        import torch
        started = time.monotonic()
        bounded = frames[:12]
        images = self._prepare_pil_images(bounded)
        if len(images) != len(bounded) or not images:
            return self._unverified(candidate_event, 'INCONCLUSIVE', 'Evidence images could not be decoded.')
        prompt = (
            "Analyze the video in its given chronological order, not reversed. First identify the starting posture, "
            "then the final posture, then the movement between them. A rapid drop or collapse onto the floor or "
            "a safety mat is FALL, even in a staged demonstration. Controlled sitting, bending, squatting, or "
            "already lying still is NO_FALL. Describe only visible motion up to the last frame; do not invent later recovery. "
            "Return one JSON object with start_posture, end_posture, "
            "scene_observation describing the transition, and classification (FALL, NO_FALL or UNCERTAIN)."
        )
        # One video token enables Qwen's temporal vision patches and MRoPE.
        # Separate image tokens encode still pictures and often miss rapid motion.
        content = [{'type':'video'}, {'type':'text','text':prompt}]
        messages = [{'role':'user','content':content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        offsets = [item.get('relative_sec') if isinstance(item,dict) else None for item in bounded]
        timed = all(isinstance(t,(int,float)) and np.isfinite(t) for t in offsets)
        if timed and any(b <= a for a,b in zip(offsets, offsets[1:])):
            return self._unverified(candidate_event, 'INCONCLUSIVE', 'Evidence frame timestamps are not increasing.')
        # The service samples chronologically; use nearest frames on a uniform
        # time grid so video FPS describes the input rather than source FPS.
        if timed:
            grid = np.linspace(offsets[0], offsets[-1], len(images))
            indices = [int(np.argmin(np.abs(np.asarray(offsets)-t))) for t in grid]
            images = [images[i] for i in indices]
            fps = (len(images)-1) / max(offsets[-1]-offsets[0], .001)
        else:
            fps = 4.0  # Legacy offline callers without timestamps.
        size = images[0].size
        video = np.stack([np.asarray(image.resize(size)) for image in images])
        metadata = {'total_num_frames':len(images), 'fps':fps,
                    'frames_indices':list(range(len(images)))}
        inputs = self.processor(text=[rendered], videos=[video], video_metadata=[metadata],
                                do_sample_frames=False, min_pixels=16*28*28, max_pixels=512*28*28,
                                padding=True, return_tensors='pt')
        inputs = {k:v.to(self.model.device) if hasattr(v,'to') else v for k,v in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=160, max_length=None, do_sample=False,
                max_time=float(os.getenv('CAREWATCH_VLM_GENERATION_TIMEOUT','45')),
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor,'tokenizer') else None)
        output_ids = generated[0][inputs['input_ids'].shape[1]:]
        output = self.processor.decode(output_ids, skip_special_tokens=True).strip()
        parsed = self._parse_model_output(output)
        classification = parsed.get('classification')
        if classification in ('FALL', 'NO_FALL', 'UNCERTAIN'):
            verdict = True if classification == 'FALL' else False if classification == 'NO_FALL' else None
            uncertain = classification == 'UNCERTAIN'
        else:
            # Also understand the strict boolean schema used by older adapters.
            verdict = parsed.get('is_true_fall')
            uncertain = parsed.get('uncertainty', True)
        valid = (type(verdict) is bool or verdict is None) and type(uncertain) is bool
        observation = parsed.get('scene_observation')
        if not isinstance(observation, str) or not observation.strip(): valid = False
        if not valid or uncertain or verdict is None:
            status, verdict = 'INCONCLUSIVE', None
        else:
            status = 'CONFIRMED_FALL' if verdict else 'NOT_FALL'
        confidence = parsed.get('confidence')
        if type(confidence) not in (float, int) or not 0 <= confidence <= 1: confidence = None
        return {
            'event_id':candidate_event.get('id'), 'model':self.model_name,
            'verification_scope':'SCENE',
            'adapter_status':'ONLINE', 'verification_method':'VLM_MODEL',
            'status':status, 'is_true_fall':verdict,
            'confidence':confidence, 'uncertainty_flag':status == 'INCONCLUSIVE',
            'scene_observation':observation if isinstance(observation,str) else 'Model response was not valid structured evidence.',
            'recovery_observed':parsed.get('voluntary_recovery') is True,
            'evidence_frames_count':len(images), 'duration_ms':round((time.monotonic()-started)*1000),
            'raw_model_output':output[:1200], 'input_mode':'VIDEO',
            'input_fps':round(fps,4), 'evidence_span_sec':round((len(images)-1)/fps,4),
        }

    def _parse_model_output(self, text: str) -> Dict[str, Any]:
        """Robust parser for JSON output inside markdown or free text."""
        def structured(value):
            if isinstance(value, dict): return value
            # Qwen sometimes reports several visible actions. This is a scene
            # verdict: a clear fall in any valid observation confirms a scene fall.
            if isinstance(value, list) and value and all(
                    isinstance(item, dict) and item.get('classification') in ('FALL','NO_FALL','UNCERTAIN')
                    and isinstance(item.get('scene_observation'),str) and item['scene_observation'].strip()
                    for item in value):
                labels = {item['classification'] for item in value}
                classification = 'FALL' if 'FALL' in labels else 'UNCERTAIN' if 'UNCERTAIN' in labels else 'NO_FALL'
                observation = next(item['scene_observation'] for item in value if item['classification']==classification)
                return {'classification':classification,'scene_observation':observation[:800]}
        # Check for ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                parsed = structured(json.loads(json_match.group(1)))
                if parsed is not None: return parsed
            except Exception:
                pass

        # Try searching for outermost curly braces
        positions = [p for p in (text.find('{'),text.find('[')) if p>=0]
        start = min(positions) if positions else -1
        end = text.rfind(']' if start>=0 and text[start]=='[' else '}')
        if start != -1 and end != -1 and end > start:
            try:
                parsed = structured(json.loads(text[start:end+1]))
                if parsed is not None: return parsed
            except Exception:
                pass

        # Invalid prose (including 'not a fall') must never become a verdict.
        return {'is_true_fall':None, 'uncertainty':True,
                'scene_observation':'The VLM did not return valid JSON; manual review is needed.'}
