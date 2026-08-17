import torch
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import numpy as np
import json
import subprocess
import os
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cosine

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/hubert-base-ls960"
BASE_DIR = Path("/home/node/.openclaw/liam/src/PinkTrumpet/research_representation")
ENGINE_PATH = BASE_DIR / "engine.py"
OUTPUT_DATA = BASE_DIR / "results_v2.json"

class RepresentationExtractor:
    def __init__(self):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self.model = HubertModel.from_pretrained(MODEL_ID).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, audio):
        import librosa
        audio_16k = librosa.resample(audio, orig_sr=44100, target_sr=16000)
        inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

def call_engine(params, duration=0.2):
    payload = json.dumps({"params": params, "duration": duration})
    proc = subprocess.run(
        ["python3", str(ENGINE_PATH)],
        input=payload,
        capture_output=True,
        text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Engine failed: {proc.stderr}")
    return np.array(json.loads(proc.stdout))

def cos_sim(a, b):
    # Ensure vectors are not zero
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0
    return np.dot(a, b) / (norm_a * norm_b)

def run_experiment_v2(num_samples=100):
    extractor = RepresentationExtractor()
    
    # Parameters to sweep
    parameters = {
        "tongueIndex": {"deltas": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "tongueDiameter": {"deltas": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "pitchHz": {"deltas": [5, 10, 15, 20, 25]},
        "constrictionIndex": {"deltas": [1, 2, 3, 4, 5]},
        "constrictionDiameter": {"deltas": [0.1, 0.2, 0.3, 0.4, 0.5]}
    }
    
    results = {p: [] for p in parameters}
    
    pbar = tqdm(range(num_samples), desc="Running samples")
    for _ in pbar:
        # Base configuration
        base_params = {
            "tongueIndex": np.random.uniform(12, 30),
            "tongueDiameter": np.random.uniform(2.0, 2.9),
            "pitchHz": np.random.uniform(100, 200),
            "tenseness": 0.6,
            "intensity": 1,
            "loudness": 1,
            "voicing": 1,
            "constrictionIndex": np.random.uniform(20, 40),
            "constrictionDiameter": 3.5
        }
        
        try:
            audio_base = call_engine(base_params)
            z_base = extractor.get_embedding(audio_base)
            
            for param_name, config in parameters.items():
                param_results = {"deltas": []}
                for d in config["deltas"]:
                    test_params = base_params.copy()
                    test_params[param_name] += d
                    
                    audio_test = call_engine(test_params)
                    z_test = extractor.get_embedding(audio_test)
                    
                    delta_z = z_test - z_base
                    param_results["deltas"].append({
                        "delta_val": d,
                        "delta_z": delta_z.tolist(),
                        "norm_delta_z": float(np.linalg.norm(delta_z))
                    })
                results[param_name].append(param_results)
        except Exception as e:
            print(f"Sample failed: {e}")
            
    # Calculate Displacement Consistency and Linearity
    final_stats = {}
    for param_name, samples in results.items():
        # Get all vectors for the largest delta (0.5 or 25Hz or 5 index units)
        idx_max = 4
        vectors = [s["deltas"][idx_max]["delta_z"] for s in samples]
        vectors = np.array(vectors)
        
        # 1. Compute Mean Displacement Vector v
        v_mean = np.mean(vectors, axis=0)
        
        # 2. Compute Cosine Similarity of each Δz_i to v_mean
        cos_to_mean = [cos_sim(v, v_mean) for v in vectors]
        
        # 3. Linearity: Check if ||Δz|| scales with Δp
        # Collect norms across all deltas
        norms_at_deltas = []
        for i in range(5):
            norms = [s["deltas"][i]["norm_delta_z"] for s in samples]
            norms_at_deltas.append(float(np.mean(norms)))
            
        final_stats[param_name] = {
            "mean_cos_to_avg_direction": float(np.mean(cos_to_mean)),
            "std_cos_to_avg_direction": float(np.std(cos_to_mean)),
            "linearity_norms": norms_at_deltas,
            "deltas": parameters[param_name]["deltas"]
        }

    with open(OUTPUT_DATA, "w") as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"Saved Version 2 results.")

if __name__ == "__main__":
    run_experiment_v2(50) # Use 50 for a faster follow-up
