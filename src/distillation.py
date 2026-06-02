import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Calculates the hybrid loss for Knowledge Distillation:
    Combines standard Cross-Entropy (hard labels) with KL-Divergence (teacher soft labels).
    """
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Balance factor between hard task labels and soft teacher labels
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1. Soften the logits using the Temperature hyperparameter
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 2. Calculate the KL-Divergence Loss (Imitation Loss)
        # Scaled by temperature squared as per standard KD literature
        distill_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 3. Calculate standard task loss (Ground truth prediction)
        # Flatten tensors for cross-entropy evaluation
        vocab_size = student_logits.size(-1)
        hard_loss = self.ce_loss(student_logits.view(-1, vocab_size), labels.view(-1))
        
        # 4. Return blended hybrid loss metric
        return (self.alpha * distill_loss) + ((1.0 - self.alpha) * hard_loss)