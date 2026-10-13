import math

import torch
import torchaudio
from torch import Tensor

from ..data.dataclass import ConfigData, AudioEmbeddingData


def _pad_with_pattern(tensor: Tensor, amount: int, pattern: list[int]) -> Tensor:
    """Pads tensor along last dimension with a repeating pattern."""
    padding = torch.as_tensor(pattern, device=tensor.device, dtype=tensor.dtype)[
        torch.arange(amount, device=tensor.device) % len(pattern)
    ]

    for _ in range(tensor.dim() - 1):
        padding = padding.unsqueeze(0)
    padding = padding.expand(*tensor.shape[:-1], amount)

    return torch.cat([tensor, padding], dim=-1)


def add_numbers_pattern(a: Tensor, b: Tensor, pattern: list[int]) -> tuple[Tensor, Tensor]:
    """Pads the shorter tensor to match the longer one using a repeating pattern."""
    len_a = a.size(-1)
    len_b = b.size(-1)

    if len_a == len_b:
        return a, b

    if len_a < len_b:
        a = _pad_with_pattern(a, len_b - len_a, pattern)
    else:
        b = _pad_with_pattern(b, len_a - len_b, pattern)

    return a, b


def generate_similar_noise(reference_tensor):
    """
    Generates random noise matching per-feature mean and std of the reference tensor.
    For 3D tensors (batch, seq, features): computes stats across batch and sequence dims.
    For lower-dim tensors: falls back to global stats.
    """

    """
    if reference_tensor.dim() >= 3:
        dims = tuple(range(reference_tensor.dim() - 1))
        mu = reference_tensor.mean(dim=dims, keepdim=True)
        std = reference_tensor.std(dim=dims, keepdim=True).clamp(min=1e-6)
    else:
        mu = reference_tensor.mean()
        std = reference_tensor.std().clamp(min=1e-6)
    """
    mu = reference_tensor.mean()
    std = reference_tensor.std()
    return torch.randn_like(reference_tensor) * std + mu

def _basis(x: Tensor, rank: int, center: bool) -> Tensor:
    """Top `rank` principal directions of the rows of x [N, D], scaled by their standard deviation: [rank, D]."""
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
    rank = min(rank, s.numel())
    return vh[:rank] * (s[:rank, None] / max(x.shape[0] - 1, 1) ** 0.5)


def dct_basis(n: int, k: int) -> Tensor:
    """First `k` orthonormal DCT-II cosines over `n` positions, scaled by sqrt(n / k) to keep the per-position variance: [n, k]."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freq = torch.arange(k, dtype=torch.float32)[None, :]
    d = math.sqrt(2.0 / n) * torch.cos(math.pi * (2 * pos + 1) * freq / (2 * n))
    d[:, 0] = math.sqrt(1.0 / n)
    return d * math.sqrt(n / k)


class VectorManipulator:
    """
    GATAS search space: additive low-rank perturbation of the text embedding and both style vectors of one sentence.

    The basis is computed per sentence, without any dataset: the SVD of the sentence's own phoneme embeddings (text)
    and the PCA of `style_samples` style vectors sampled for the same text (acoustic, prosodic).
    A genome x in [0, 1]^n_var holds `rank_text` coefficients per phoneme and `rank_style` per style vector;
    x = 0.5 is the unperturbed sentence, and each coefficient moves `iv_scalar` standard deviations along its direction.
    With `adaptive_scale`, the genome has one more gene that sets the individual's scale in [SCALE_MIN, iv_scalar]:
    the search itself raises the scale when nothing is found and lowers it when it gets close.

    Two options make the genes more informative:
    - `basis="jacobian"`: the directions are ranked by how much they change the audio instead of by embedding variance.
      The TTS is probed by finite differences along a larger PCA subspace (`PROBE_FACTOR` x rank); the eigenvectors of
      the Gram matrix of the log-mel responses (a Gauss-Newton metric in mel space) give the new basis. Only the
      TTS is queried, so the ASR budget is unchanged.
    - `text_dct=K`: the text coefficients of the phonemes are the first K cosines of a DCT along the phoneme axis
      (K x rank genes instead of phonemes x rank), so neighbouring phonemes are perturbed coherently.
    """

    SCALE_MIN = 0.5
    PROBE_FACTOR = 3  # jacobian basis: probed subspace is this many times larger than the kept rank

    def __init__(self, tts, tokens: Tensor, audio_embedding_gt: AudioEmbeddingData, config_data: ConfigData, style_samples: int = 48):
        self.gt = audio_embedding_gt
        self.scale = config_data.iv_scalar
        h_text = audio_embedding_gt.h_text[0].T  # [phonemes, 512]
        styles = tts.extract_embeddings(tokens.expand(style_samples, -1), torch.randn(style_samples, 1, 256, device=tokens.device))
        rank_style = min(config_data.rank_style, style_samples - 1)
        rank_text = config_data.rank_text

        jacobian = config_data.basis == "jacobian"
        probe = self.PROBE_FACTOR if jacobian else 1
        self.basis_text = _basis(h_text, probe * rank_text, center=False)
        self.basis_acoustic = _basis(styles.style_vector_acoustic, min(probe * rank_style, style_samples - 1), center=True)
        self.basis_prosodic = _basis(styles.style_vector_prosodic, min(probe * rank_style, style_samples - 1), center=True)
        if jacobian:
            self.tts = tts
            self._mel = torchaudio.transforms.MelSpectrogram(16_000, n_fft=400, hop_length=160, n_mels=80).to(h_text.device)
            self.basis_text, self.basis_acoustic, self.basis_prosodic = self._sensitive_bases(
                rank_text, rank_style, config_data.sens_phonemes)
            del self.tts, self._mel

        n_positions = h_text.shape[0]
        k_text = config_data.text_dct if 0 < config_data.text_dct < n_positions else 0
        self.dct = dct_basis(n_positions, k_text).to(h_text.device) if k_text else None  # [phonemes, K]
        self.n_text = (k_text or n_positions) * self.basis_text.shape[0]
        self.n_style = self.basis_acoustic.shape[0]
        self.n_var = self.n_text + 2 * self.n_style
        self.adaptive = config_data.adaptive_scale
        self.n_genes = self.n_var + int(self.adaptive)  # search dimensions

    # -- output-sensitive basis -------------------------------------------------------------------------------------

    def _embed(self, h_text: Tensor, acoustic: Tensor, prosodic: Tensor) -> AudioEmbeddingData:
        gt, batch = self.gt, h_text.shape[0]
        return AudioEmbeddingData(
            gt.input_length.expand(batch), gt.text_mask.expand(batch, -1), gt.h_bert.expand(batch, -1, -1),
            h_text, acoustic, prosodic,
        )

    @torch.no_grad()
    def _log_mel(self, h_text: Tensor, acoustic: Tensor, prosodic: Tensor, chunk: int = 32) -> list[Tensor]:
        """Log-mel spectrograms [80, T_i] of the TTS output for a batch of embeddings."""
        mels = []
        for i in range(0, h_text.shape[0], chunk):
            audio = self.tts.inference_on_embedding(self._embed(h_text[i:i + chunk], acoustic[i:i + chunk], prosodic[i:i + chunk]))
            mels += list(torch.log(self._mel(audio.float()).clamp(min=1e-5)).unbind(0))
        return mels

    @staticmethod
    def _responses(mels: list[Tensor], base: Tensor, per_group: int) -> Tensor:
        """Difference of every mel to the unperturbed one (zero-padded in time to equal length), flattened: [groups, per_group, D]."""
        frames = max(base.shape[1], *(m.shape[1] for m in mels))
        pad = lambda m: torch.nn.functional.pad(m, (0, frames - m.shape[1]), value=math.log(1e-5))
        diff = torch.stack([pad(m) for m in mels]) - pad(base)
        return diff.flatten(1).reshape(-1, per_group, diff[0].numel())

    @staticmethod
    def _top_directions(responses: Tensor, basis: Tensor, rank: int) -> Tensor:
        """Top `rank` eigen-directions of the Gram matrix of the responses, mapped back to embedding space: [rank, D]."""
        gram = torch.einsum("gkd,gld->kl", responses, responses)
        _, vectors = torch.linalg.eigh(gram.double())
        return vectors[:, -rank:].flip(1).T.float() @ basis

    def _sensitive_bases(self, rank_text: int, rank_style: int, n_phonemes: int):
        """Replaces the PCA bases by their most audible rotations (see class docstring)."""
        gt = self.gt
        h_text, acoustic, prosodic = gt.h_text, gt.style_vector_acoustic, gt.style_vector_prosodic
        base = self._log_mel(h_text, acoustic, prosodic)[0]

        # text: perturb sampled phonemes one at a time along every probe direction (coefficient = 1 std)
        positions = torch.linspace(0, h_text.shape[-1] - 1, min(n_phonemes, h_text.shape[-1])).round().long().unique()
        k = self.basis_text.shape[0]
        batch = h_text.repeat(len(positions) * k, 1, 1)
        for g, pos in enumerate(positions):
            batch[g * k:(g + 1) * k, :, pos] += self.basis_text
        mels = self._log_mel(batch, acoustic.expand(batch.shape[0], -1), prosodic.expand(batch.shape[0], -1))
        text = self._top_directions(self._responses(mels, base, k), self.basis_text, rank_text)

        # style: one vector each, so one group per vector
        styles = []
        for basis, which in ((self.basis_acoustic, 0), (self.basis_prosodic, 1)):
            k = basis.shape[0]
            a = acoustic + basis if which == 0 else acoustic.expand(k, -1)
            p = prosodic + basis if which == 1 else prosodic.expand(k, -1)
            mels = self._log_mel(h_text.expand(k, -1, -1), a, p)
            styles.append(self._top_directions(self._responses(mels, base, k), basis, rank_style))
        return text, styles[0], styles[1]

    def scale_of(self, genome) -> float:
        return round(float(self.SCALE_MIN + (self.scale - self.SCALE_MIN) * genome[-1]), 4) if self.adaptive else self.scale

    def interpolate(self, genomes: Tensor):
        batch = genomes.shape[0]
        c = 2.0 * genomes[:, :self.n_var] - 1.0  # [batch, n_var], 0 = unperturbed
        c = c * (self.SCALE_MIN + (self.scale - self.SCALE_MIN) * genomes[:, self.n_var:] if self.adaptive else self.scale)

        c_text = c[:, :self.n_text].reshape(batch, -1, self.basis_text.shape[0])
        if self.dct is not None:
            c_text = torch.einsum("pk,bkr->bpr", self.dct, c_text)  # [batch, phonemes, rank]
        c_acoustic = c[:, self.n_text:self.n_text + self.n_style]
        c_prosodic = c[:, self.n_text + self.n_style:]

        gt = self.gt
        h_text = gt.h_text + (c_text @ self.basis_text).transpose(1, 2)
        acoustic = gt.style_vector_acoustic + c_acoustic @ self.basis_acoustic
        prosodic = gt.style_vector_prosodic + c_prosodic @ self.basis_prosodic

        return batch, genomes, self._embed(h_text, acoustic, prosodic)


class InterpolationManipulator:
    """
    GATAS_llm search space: interpolation in the StyleTTS2 latent space between the seed sentence and a target sentence.

    The target sentence is an LLM rewrite of the seed sentence with the target intent (scripts/generate_intent_rewrites.py).
    Its embedding is extracted with the same style noise as the seed's reference, and its text / BERT features are resampled to
    the seed's phoneme count, so the seed's duration frame and text mask are kept. A genome x in [0, 1]^n_genes holds
    - `knots` alphas along the phoneme axis (piecewise linear), applied to the text and BERT features: 0 = seed, 1 = target,
      so the search can move only some words toward the target;
    - one alpha for the acoustic and one for the prosodic style vector.
    Interpolation only: there is no further perturbation of the seed.
    """

    def __init__(self, tts, audio_embedding_gt: AudioEmbeddingData, target_tokens: Tensor, noise: Tensor, knots: int = 8):
        self.gt = audio_embedding_gt
        n_phonemes = audio_embedding_gt.h_text.shape[-1]
        target = tts.extract_embeddings(target_tokens, noise)
        resample = lambda x: torch.nn.functional.interpolate(x.float(), size=n_phonemes, mode="linear", align_corners=True)
        self.target_text, self.target_bert = resample(target.h_text), resample(target.h_bert)
        self.target_acoustic, self.target_prosodic = target.style_vector_acoustic, target.style_vector_prosodic
        self.knots = max(2, min(knots, n_phonemes))
        self.n_genes = self.knots + 2
        self.scale = 1.0

    def scale_of(self, genome) -> float:
        """Mean alpha (share of the way toward the target sentence), logged as the perturbation scale."""
        return round(float(genome[:self.n_genes].mean()), 4)

    def interpolate(self, genomes: Tensor):
        batch, gt = genomes.shape[0], self.gt
        n_phonemes = gt.h_text.shape[-1]
        alpha = torch.nn.functional.interpolate(genomes[:, None, :self.knots].float(), size=n_phonemes, mode="linear", align_corners=True)  # [batch, 1, phonemes]
        a_acoustic, a_prosodic = genomes[:, self.knots:self.knots + 1], genomes[:, self.knots + 1:self.knots + 2]

        h_text = (1 - alpha) * gt.h_text + alpha * self.target_text
        h_bert = (1 - alpha) * gt.h_bert + alpha * self.target_bert
        acoustic = (1 - a_acoustic) * gt.style_vector_acoustic + a_acoustic * self.target_acoustic
        prosodic = (1 - a_prosodic) * gt.style_vector_prosodic + a_prosodic * self.target_prosodic

        embedding = AudioEmbeddingData(gt.input_length.expand(batch), gt.text_mask.expand(batch, -1), h_bert, h_text, acoustic, prosodic)
        return batch, genomes, embedding
