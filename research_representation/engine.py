import numpy as np
import json
from pathlib import Path

# Constants from pink-trombone-worklet.js
SAMPLE_RATE = 44100
N_TRACT = 44
BLADE_START = 10
TIP_START = 32
LIP_START = 39
NOSE_LENGTH = 28
NOSE_START = N_TRACT - NOSE_LENGTH + 1

def clamp(value, min_val, max_val):
    return min(max_val, max(min_val, value))

def move_towards(current, target, up, down=None):
    if down is None: down = up
    if current < target:
        return min(current + up, target)
    else:
        return max(current - down, target)

class BandpassFilter:
    def __init__(self, rate, frequency, q):
        omega = 2 * np.pi * frequency / rate
        alpha = np.sin(omega) / (2 * q)
        a0 = 1 + alpha
        self.b0 = alpha / a0
        self.b2 = -self.b0
        self.a1 = -2 * np.cos(omega) / a0
        self.a2 = (1 - alpha) / a0
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0

    def process(self, x):
        y = self.b0 * x + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y

class SmoothNoise:
    def __init__(self, seed=0.731):
        self.seed = seed
        self.baseline = self.raw(0)

    def raw(self, x):
        return (
            np.sin((x + self.seed) * 1.73) * 0.47 +
            np.sin((x + self.seed * 7) * 0.73) * 0.31 +
            np.sin((x + 1.9) * 0.19) * 0.22
        )

    def sample(self, x):
        return clamp((self.raw(x) - self.baseline) * 1.2, -1, 1)

class Glottis:
    def __init__(self, rate):
        self.sample_rate = rate
        self.time_in_waveform = 0
        self.total_time = 0
        self.frequency = 140
        self.tenseness = 0.6
        self.ui_tenseness = 0.6
        self.intensity = 1
        self.loudness = 1
        self.voicing = 1
        self.aspiration = 1
        self.vibrato = 1
        self.wobble = 1
        self.noise = SmoothNoise()
        self.setup_waveform()

    def update(self, p):
        v = p.get('vibrato', 1) * 0.005 * np.sin(2 * np.pi * self.total_time * 6)
        v += 0.02 * self.noise.sample(self.total_time * 4.07)
        v += 0.04 * self.noise.sample(self.total_time * 2.15)
        v += p.get('wobble', 1) * 0.2 * self.noise.sample(self.total_time * 0.98)
        v += p.get('wobble', 1) * 0.4 * self.noise.sample(self.total_time * 0.5)
        self.frequency = p['pitchHz'] * (1 + v)
        self.ui_tenseness = p['tenseness']
        self.tenseness = p['tenseness'] + 0.1 * self.noise.sample(self.total_time * 0.46) + 0.05 * self.noise.sample(self.total_time * 0.36)
        self.intensity = p.get('intensity', 1)
        self.loudness = p.get('loudness', 1)
        self.voicing = p.get('voicing', 1)
        self.aspiration = p.get('aspiration', 1)
        self.setup_waveform()

    def setup_waveform(self):
        self.waveform_length = 1 / max(40, self.frequency)
        rd = clamp(3 * (1 - self.tenseness), 0.5, 2.7)
        ra = -0.01 + 0.048 * rd
        rk = 0.224 + 0.118 * rd
        rg = (rk / 4) * (0.5 + 1.2 * rk) / (0.11 * rd - ra * (0.5 + 1.2 * rk))
        ta, tp = ra, 1 / (2 * rg)
        te = tp + tp * rk
        epsilon = 1 / ta
        shift = np.exp(-epsilon * (1 - te))
        delta = 1 - shift
        rhs_integral = ((shift - 1) / epsilon + (1 - te) * shift) / delta
        upper_integral = -(-(te - tp) / 2 + rhs_integral)
        omega = np.pi / tp
        s = np.sin(omega * te)
        y = max(1e-6, (-np.pi * s * upper_integral) / (tp * 2))
        alpha = np.log(y) / (tp / 2 - te)
        self.alpha, self.e0, self.epsilon, self.shift, self.delta, self.te, self.omega = alpha, -1/(s*np.exp(alpha*te)), epsilon, shift, delta, te, omega

    def noise_modulator(self):
        voiced_pulse = 0.1 + 0.2 * max(0, np.sin(2 * np.pi * self.time_in_waveform / self.waveform_length))
        return self.ui_tenseness * self.intensity * voiced_pulse + (1 - self.ui_tenseness * self.intensity) * 0.3

    def run(self, white_noise):
        step = 1 / self.sample_rate
        self.time_in_waveform += step
        self.total_time += step
        if self.time_in_waveform >= self.waveform_length:
            self.time_in_waveform %= self.waveform_length
            self.setup_waveform()
        t = self.time_in_waveform / self.waveform_length
        lf = (-np.exp(-self.epsilon * (t - self.te)) + self.shift) / self.delta if t > self.te else self.e0 * np.exp(self.alpha * t) * np.sin(self.omega * t)
        voiced = lf * self.intensity * self.loudness * self.voicing
        aspiration_scale = 0.2 + 0.02 * self.noise.sample(self.total_time * 1.99)
        breath = white_noise * self.aspiration * self.intensity * (1 - np.sqrt(clamp(self.ui_tenseness, 0, 1))) * self.noise_modulator() * aspiration_scale
        return voiced + breath

class VocalTract:
    def __init__(self, rate):
        self.sample_rate = rate
        self.n = N_TRACT
        self.nose_start = NOSE_START
        self.nose_length = NOSE_LENGTH
        self.glottal_reflection = 0.75
        self.lip_reflection = -0.85
        self.mouth_fade = 0.999
        self.nose_fade = 1
        self.movement_speed = 15
        self.velum_target = 0.01
        self.lip_output = 0
        self.nose_output = 0
        self.last_obstruction = -1
        self.transients = []
        self.allocate()
        self.initialize_diameters()
        self.calculate_reflections()
        self.calculate_nose_reflections()

    def allocate(self):
        for name in ["R", "L", "diameter", "restDiameter", "targetDiameter", "A"]:
            setattr(self, name, np.zeros(self.n))
        for name in ["reflection", "newReflection", "junctionOutputR", "junctionOutputL"]:
            setattr(self, name, np.zeros(self.n + 1))
        for name in ["noseR", "noseL", "noseDiameter", "noseA"]:
            setattr(self, name, np.zeros(self.nose_length))
        for name in ["noseReflection", "noseJunctionOutputR", "noseJunctionOutputL"]:
            setattr(self, name, np.zeros(self.nose_length + 1))
        self.newReflectionLeft = self.newReflectionRight = self.newReflectionNose = 0
        self.reflectionLeft = self.reflectionRight = self.reflectionNose = 0

    def initialize_diameters(self):
        for i in range(self.n):
            self.diameter[i] = self.restDiameter[i] = self.targetDiameter[i] = 0.6 if i < 6.5 else 1.1 if i < 12 else 1.5
        for i in range(self.nose_length):
            d = 2 * i / self.nose_length
            self.noseDiameter[i] = min(0.4 + 1.6 * d if d < 1 else 0.5 + 1.5 * (2 - d), 1.9)
        self.noseDiameter[0] = self.velum_target

    def set_targets(self, p):
        for i in range(self.n):
            base = 0.6 if i < 6.5 else 1.1 if i < 12 else 1.5
            if BLADE_START <= i < LIP_START:
                t = 1.1 * np.pi * (p['tongueIndex'] - i) / (TIP_START - BLADE_START)
                fixed = 2 + (p['tongueDiameter'] - 2) / 1.5
                curve = (1.5 - fixed + 1.7) * np.cos(t)
                if i == BLADE_START - 2 or i == LIP_START - 1: curve *= 0.8
                if i == BLADE_START or i == LIP_START - 2: curve *= 0.94
                base = 1.5 - curve
            self.restDiameter[i] = self.targetDiameter[i] = base
        idx = clamp(p['constrictionIndex'], 2, self.n - 3)
        dia = clamp(p['constrictionDiameter'], 0, 3.5)
        center = int(round(idx))
        width = 10 if idx < 25 else 5 if idx >= TIP_START else 10 - 5 * (idx - 25) / (TIP_START - 25)
        for offset in range(-int(np.ceil(width)) - 1, int(width) + 2):
            section = center + offset
            if 0 <= section < self.n:
                rel = abs(section - idx) - 0.5
                shrink = 0 if rel <= 0 else 1 if rel > width else 0.5 * (1 - np.cos(np.pi * rel / width))
                if dia < self.targetDiameter[section]:
                    self.targetDiameter[section] = dia + (self.targetDiameter[section] - dia) * shrink
        self.velum_target = p.get('velum', 0.01)

    def reshape(self, delta_time):
        amount = delta_time * self.movement_speed
        obstruction = -1
        for i in range(self.n):
            if self.diameter[i] <= 0: obstruction = i
            slow = 0.6 if i < self.nose_start else 1.0 if i >= TIP_START else 0.6 + 0.4 * (i - self.nose_start) / (TIP_START - self.nose_start)
            self.diameter[i] = move_towards(self.diameter[i], self.targetDiameter[i], slow * amount, 2 * amount)
        if self.last_obstruction > -1 and obstruction == -1 and self.noseA[0] < 0.05:
            self.transients.append({'position': clamp(self.last_obstruction, 0, self.n - 1), 'age': 0})
        self.last_obstruction = obstruction
        self.noseDiameter[0] = move_towards(self.noseDiameter[0], self.velum_target, amount * 0.25, amount * 0.1)
        self.noseA[0] = self.noseDiameter[0] ** 2

    def calculate_reflections(self):
        for i in range(self.n): self.A[i] = self.diameter[i] ** 2
        for i in range(1, self.n):
            self.reflection[i] = self.newReflection[i]
            self.newReflection[i] = 0.999 if self.A[i] == 0 else (self.A[i - 1] - self.A[i]) / (self.A[i - 1] + self.A[i])
        self.reflectionLeft, self.reflectionRight, self.reflectionNose = self.newReflectionLeft, self.newReflectionRight, self.newReflectionNose
        sum_a = max(1e-6, self.A[self.nose_start] + self.A[self.nose_start + 1] + self.noseA[0])
        self.newReflectionLeft = (2 * self.A[self.nose_start] - sum_a) / sum_a
        self.newReflectionRight = (2 * self.A[self.nose_start + 1] - sum_a) / sum_a
        self.newReflectionNose = (2 * self.noseA[0] - sum_a) / sum_a

    def calculate_nose_reflections(self):
        for i in range(self.nose_length): self.noseA[i] = self.noseDiameter[i] ** 2
        for i in range(1, self.nose_length):
            self.noseReflection[i] = (self.noseA[i - 1] - self.noseA[i]) / (self.noseA[i - 1] + self.noseA[i])

    def run(self, glottal_output, turbulence_noise, lambda_val, p, noise_modulator):
        for t in self.transients[:]:
            amp = 0.3 * 2 ** (-200 * t['age'])
            self.R[t['position']] += amp / 2
            self.L[t['position']] += amp / 2
            t['age'] += 1 / (self.sample_rate * 2)
            if t['age'] > 0.2: self.transients.remove(t)
        if p.get('fricativeIntensity', 0) > 0 and p['constrictionDiameter'] > 0:
            idx = clamp(p['constrictionIndex'], 2, self.n - 3)
            i = int(idx)
            delta = idx - i
            thin = clamp(8 * (0.7 - p['constrictionDiameter']), 0, 1)
            openness = clamp(30 * (p['constrictionDiameter'] - 0.3), 0, 1)
            val = 0.66 * turbulence_noise * p['fricativeIntensity'] * noise_modulator * thin * openness
            a, b = val * (1 - delta) / 2, val * delta / 2
            if i + 1 < self.n: self.R[i+1] += a; self.L[i+1] += a
            if i + 2 < self.n: self.R[i+2] += b; self.L[i+2] += b
        self.junctionOutputR[0] = self.L[0] * self.glottal_reflection + glottal_output
        self.junctionOutputL[self.n] = self.R[self.n - 1] * self.lip_reflection
        for i in range(1, self.n):
            r = self.reflection[i] * (1 - lambda_val) + self.newReflection[i] * lambda_val
            w = r * (self.R[i - 1] + self.L[i])
            self.junctionOutputR[i] = self.R[i - 1] - w
            self.junctionOutputL[i] = self.L[i] + w
        j = self.nose_start
        rl = self.newReflectionLeft * (1 - lambda_val) + self.reflectionLeft * lambda_val
        rr = self.newReflectionRight * (1 - lambda_val) + self.reflectionRight * lambda_val
        rn = self.newReflectionNose * (1 - lambda_val) + self.reflectionNose * lambda_val
        self.junctionOutputL[j] = rl * self.R[j - 1] + (1 + rl) * (self.noseL[0] + self.L[j])
        self.junctionOutputR[j] = rr * self.L[j] + (1 + rr) * (self.R[j - 1] + self.noseL[0])
        self.noseJunctionOutputR[0] = rn * self.noseL[0] + (1 + rn) * (self.L[j] + self.R[j - 1])
        for i in range(self.n):
            self.R[i], self.L[i] = self.junctionOutputR[i] * self.mouth_fade, self.junctionOutputL[i + 1] * self.mouth_fade
        self.lip_output = self.R[self.n - 1]
        self.noseJunctionOutputL[self.nose_length] = self.noseR[self.nose_length - 1] * self.lip_reflection
        for i in range(1, self.nose_length):
            w = self.noseReflection[i] * (self.noseR[i - 1] + self.noseL[i])
            self.noseJunctionOutputR[i], self.noseJunctionOutputL[i] = self.noseR[i - 1] - w, self.noseL[i] + w
        for i in range(self.nose_length):
            self.noseR[i], self.noseL[i] = self.noseJunctionOutputR[i] * self.nose_fade, self.noseJunctionOutputL[i + 1] * self.nose_fade
        self.nose_output = self.noseR[self.nose_length - 1]

    def finish_block(self, block_time):
        self.reshape(block_time)
        self.calculate_reflections()

def generate_audio(params, duration_sec=0.1, seed=None):
    glottis = Glottis(SAMPLE_RATE)
    tract = VocalTract(SAMPLE_RATE)
    asp_filter = BandpassFilter(SAMPLE_RATE, 500, 0.5)
    fric_filter = BandpassFilter(SAMPLE_RATE, 1000, 0.5)
    rng = np.random.default_rng(seed)
    block_len = 512
    num_samples = int(SAMPLE_RATE * duration_sec)
    samples = np.zeros(num_samples)
    
    # Warmup tract
    tract.set_targets(params)
    for _ in range(10): tract.finish_block(block_len / SAMPLE_RATE)
    
    for i in range(num_samples):
        glottis.update(params)
        tract.set_targets(params)
        white_noise = rng.random()
        asp_noise = asp_filter.process(white_noise)
        fric_noise = fric_filter.process(white_noise)
        glottal = glottis.run(asp_noise)
        
        l_pos = i % block_len
        lambda1 = l_pos / block_len
        lambda2 = (l_pos + 0.5) / block_len
        
        tract.run(glottal, fric_noise, lambda1, params, glottis.noise_modulator())
        val = (tract.lip_output + tract.nose_output)
        tract.run(glottal, fric_noise, lambda2, params, glottis.noise_modulator())
        val += (tract.lip_output + tract.nose_output)
        samples[i] = val * 0.125
        
        if (i + 1) % block_len == 0:
            tract.finish_block(block_len / SAMPLE_RATE)
            
    return samples

if __name__ == "__main__":
    import sys
    config = json.loads(sys.stdin.read())
    audio = generate_audio(
        config['params'],
        config.get('duration', 0.1),
        seed=config.get('seed'),
    )
    print(json.dumps(audio.tolist()))
