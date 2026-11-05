import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PageEncoder(nn.Module):
    """ResNet patch encoder + attention pooling -> one L2-normalised page embedding.

    forward(x) with x: [B, K, 1, H, W]. With return_all=True also returns the raw patch
    embeddings [B, K, D] and the attention weights [B, K] (used for distillation and heatmaps).
    """

    def __init__(self, dim: int = 256, backbone: str = "resnet18", pool: str = "attention", teacher_dim: int = 384):
        super().__init__()
        self.pool = pool
        self.cnn = timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=0)
        self.proj = nn.Linear(self.cnn.num_features, dim)
        self.att = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.distill_head = nn.Linear(dim, teacher_dim)  # training only; unused at inference

    def forward(self, x, return_all: bool = False):
        b, k = x.shape[:2]
        f = self.proj(self.cnn(x.flatten(0, 1))).view(b, k, -1)
        if self.pool == "attention":
            a = torch.softmax(self.att(f).squeeze(-1), dim=1)
        else:
            a = torch.full((b, k), 1.0 / k, device=f.device, dtype=f.dtype)
        page = F.normalize((a.unsqueeze(-1) * f).sum(1), dim=-1)
        return (page, f, a) if return_all else page


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temp: float) -> torch.Tensor:
    sim = emb @ emb.T / temp
    n = emb.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def feature_distill(student_patch: torch.Tensor, head: nn.Module, teacher_patch: torch.Tensor) -> torch.Tensor:
    """1 - cosine between the student's (projected) patch embedding and the frozen teacher's."""
    s = F.normalize(head(student_patch.flatten(0, 1)), dim=-1)
    t = F.normalize(teacher_patch.flatten(0, 1), dim=-1)
    return (1 - (s * t).sum(-1)).mean()


def relational_distill(student: torch.Tensor, teacher: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """KL(teacher || student) between row-softmaxed similarity matrices (diagonal excluded)."""
    s = F.normalize(student, dim=-1); t = F.normalize(teacher, dim=-1)
    n = s.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=s.device)
    ss = (s @ s.T / temp).masked_fill(eye, -1e9)
    tt = (t @ t.T / temp).masked_fill(eye, -1e9)
    return F.kl_div(F.log_softmax(ss, dim=1), F.softmax(tt, dim=1), reduction="batchmean")
