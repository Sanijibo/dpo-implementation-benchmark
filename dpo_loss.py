import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Direct Preference Optimization (DPO) Loss implementation.
    
    Reference: https://arxiv.org/abs/2305.18290
    
    Args:
        policy_chosen_logps: Log probs of the policy model for the chosen responses.
        policy_rejected_logps: Log probs of the policy model for the rejected responses.
        ref_chosen_logps: Log probs of the reference model for the chosen responses.
        ref_rejected_logps: Log probs of the reference model for the rejected responses.
        beta: Temperature parameter for the DPO loss (typically 0.1 to 0.5).
        
    Returns:
        losses: The DPO loss for each example in the batch.
        chosen_rewards: Implicit rewards for chosen responses.
        rejected_rewards: Implicit rewards for rejected responses.
    """
    
    # Calculate the log-ratio for chosen and rejected responses
    # pi_logratios = policy_logps - reference_logps
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # The DPO objective maximizes the log-likelihood of the preference
    # Loss = -log(sigmoid(beta * (chosen_logratios - rejected_logratios)))
    logits = beta * (chosen_logratios - rejected_logratios)
    losses = -F.logsigmoid(logits)

    # Compute implicit rewards for visualization/logging
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return losses, chosen_rewards, rejected_rewards

# Simple test to verify implementation
if __name__ == "__main__":
    # Dummy data to simulate a batch
    p_chosen = torch.tensor([-1.0, -0.5])
    p_rejected = torch.tensor([-2.0, -1.5])
    r_chosen = torch.tensor([-1.1, -0.6])
    r_rejected = torch.tensor([-2.1, -1.6])
    
    loss, _, _ = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected)
    print(f"DPO Loss calculated: {loss}")
