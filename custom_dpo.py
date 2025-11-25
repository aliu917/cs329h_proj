# Code is adapted from hw1
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_logprobs(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )
    logits = outputs.logits
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
    return token_logprobs


def compute_dpo_objective(preferred_train_logprobs, nonpreferred_train_logprobs, preferred_ref_logprobs, nonpreferred_ref_logprobs, beta, ips_weight=None):
    """
    Computes the Direct Preference Optimization (DPO) objective for training.

    Args:
    preferred_train_logprobs (torch.Tensor): Token probabilities for the preferred chat sequence from the training model.
    nonpreferred_train_logprobs (torch.Tensor): Token probabilities for the non-preferred chat sequence from the training model.
    preferred_ref_logprobs (torch.Tensor): Token probabilities for the preferred chat sequence from the reference model.
    nonpreferred_ref_logprobs (torch.Tensor): Token probabilities for the non-preferred chat sequence from the reference model.
    beta (float): Controls the KL strength of staying close to the reference model.

    Returns:
    torch.Tensor: The computed DPO objective, which is a float.
    """

    # YOUR CODE HERE (~4-6 lines)
    preferred_log_ratio = torch.sum(preferred_train_logprobs, dim=1) - torch.sum(preferred_ref_logprobs, dim=1)
    nonpreferred_log_ratio = torch.sum(nonpreferred_train_logprobs, dim=1) - torch.sum(nonpreferred_ref_logprobs, dim=1)
    dpo_obj = -F.logsigmoid(beta * preferred_log_ratio - beta * nonpreferred_log_ratio)
    if ips_weight:
        dpo_obj_ips = (dpo_obj * ips_weight).mean()
    else:
        dpo_obj_ips = dpo_obj.mean()
    # END OF YOUR CODE

    return dpo_obj_ips


def dpo_step(train_model, ref_model, preferred_chat_ids, nonpreferred_chat_ids, preferred_mask, nonpreferred_mask, beta, ips_weight):
    """
    Fine-tunes the training model using DPO. Make sure to disable gradients on the reference model!

    Args:
    optimizer: The optimizer for updating the training model's parameters.
    train_model: The model being fine-tuned.
    ref_model: The reference model.
    preferred_chat_ids (list[int]): The token IDs representing the preferred chat sequence.
    nonpreferred_chat_ids (list[int]): The token IDs representing the non-preferred chat sequence.
    num_gradient_steps (int): The number of gradient updates to perform.
    beta (float): A parameter used in computing the DPO objective.

    Returns:
    dpo loss
    """
    preferred_train_logprobs = get_logprobs(train_model, preferred_chat_ids, preferred_mask)
    nonpreferred_train_logprobs = get_logprobs(train_model, nonpreferred_chat_ids, nonpreferred_mask)

    # Gradients are not needed for the reference model since we will not be optimizing with respect to it
    with torch.no_grad():
        preferred_ref_logprobs = get_logprobs(ref_model, preferred_chat_ids, preferred_mask)
        nonpreferred_ref_logprobs = get_logprobs(ref_model, nonpreferred_chat_ids, nonpreferred_mask)

    dpo_obj = compute_dpo_objective(preferred_train_logprobs, nonpreferred_train_logprobs, preferred_ref_logprobs, nonpreferred_ref_logprobs, beta, ips_weight)
    return dpo_obj


def finetune(wandb, optimizer, train_model, ref_model, dataloader, beta, num_epochs, use_ips=False):
    train_model.train()
    ref_model.eval()

    for epoch in range(num_epochs):
        total_loss = 0
        for step, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}")):
            preferred_chat_ids = data["chosen"]["input_ids"].to(train_model.device)
            preferred_mask = data["chosen"]["attention_mask"].to(train_model.device)
            nonpreferred_chat_ids = data["rejected"]["input_ids"].to(train_model.device)
            nonpreferred_mask = data["rejected"]["attention_mask"].to(train_model.device)

            ips_weight = None
            if use_ips:
                ips_weight = data["ips_weight"].to(train_model.device)

            loss = dpo_step(train_model, ref_model, preferred_chat_ids, nonpreferred_chat_ids, preferred_mask, nonpreferred_mask, beta, ips_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if len(data) > 1:
                print(f"Step: {step+1}: {loss.item():.4f}")
                wandb.log({
                    "loss": loss.item(),
                    "step": step
                })

        wandb.log({
            "loss": total_loss / len(dataloader),
            "epoch": epoch
        })
        if epoch % 5 == 0:
            print("saving model")
            local_path = "./temp_model"
            train_model.save_pretrained(local_path)
            artifact = wandb.Artifact(
                name="dpo_model_{}".format(epoch),
                type="model",
                description="DPO-finetuned model",
            )
            artifact.add_dir(local_path)
            wandb.log_artifact(artifact)
            artifact.wait()
            print("done saving model")
