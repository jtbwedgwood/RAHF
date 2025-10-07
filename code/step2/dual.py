def compute_loss_DUAL(self, model, model_good,model_bad,model_base,
                 inputs, target_layers, alpha, beta, 
                 loss_type = ["mse_loss"],max_res_len=64, return_outputs=False, 
                 **kwargs):
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    input_ids = input_ids[:,0]
    attention_mask = attention_mask[:,0]
    min_length = max_res_len
    response_attention_mask = attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

    # with model.disable_adapter():
    model.eval()
    with torch.no_grad():
        # with model.disable_adapter():
        base_outputs = model_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        base_hidden = [base_outputs["hidden_states"][l][:, -min_length:].detach() for l in target_layers]
        base_logits = base_outputs["logits"]

        bad_outputs = model_bad(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )['hidden_states']
        good_outputs = model_good(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )['hidden_states']

        direction_hidden = [good_outputs[l][:, -min_length:].detach() - \
                        bad_outputs[l][:, -min_length:].detach() \
                        # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.bfloat16) \
                                        for l in target_layers]

        target_hidden = torch.stack([base_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask

        del base_outputs, good_outputs, bad_outputs, base_hidden, direction_hidden
        gc.collect()
        torch.cuda.empty_cache()

    model.train()
    lora_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    lora_hidden = torch.stack([lora_outputs["hidden_states"][l][:, -min_length:] for l in target_layers]) * response_attention_mask
    lora_logits = lora_outputs["logits"]

    loss_fct = torch.nn.MSELoss()
    loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
    if "kl_loss" in loss_type:
        lora_logits=F.log_softmax(lora_logits, dim=-1)
        base_logits = F.log_softmax(base_logits, dim=-1)
        kl_loss = F.kl_div(lora_logits, base_logits, reduction='mean')
        loss += kl_loss

    return (loss, lora_hidden) if return_outputs else loss

#----------------------------------------- load models ---------------------------------------------------------
    if training_args.method == "DUAL":
        if model_args.model_lora_path:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
            model = prepare_model_for_kbit_training(model)

            rahf_target_layers = [int(layer) for layer in rahf_args.target_layers.split(",")] # target representations
            lora_layers_to_transform = list(range(rahf_target_layers[-1] + 1)) # LoRA layers
            config = LoraConfig(
                    r=lora_args.lora_r,
                    lora_alpha=lora_args.lora_alpha,
                    target_modules=lora_args.lora_target_modules,
                    lora_dropout=lora_args.lora_dropout,
                    bias=lora_args.lora_bias,
                    layers_to_transform=lora_layers_to_transform,
                    task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, config)
        if model_args.model_good_lora_path:
            model_good = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_good_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=False,
            )
        else:
            model_good = AutoModelForCausalLM.from_pretrained(
                    model_args.model_good_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
        if model_args.model_bad_lora_path:
            model_bad = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_bad_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
                is_trainable=False,
            )
        else:
            model_bad = AutoModelForCausalLM.from_pretrained(
                    model_args.model_bad_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
        if model_args.model_base_lora_path:
            model_base = AutoPeftModelForCausalLM.from_pretrained(
                model_args.model_base_lora_path, # location of saved SFT model
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                load_in_8bit=model_args.load_in_8bit,
            )
        else:
            model_base = AutoModelForCausalLM.from_pretrained(
                    model_args.model_base_name_or_path,
                    load_in_8bit=model_args.load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )

                if training_args.method == "DUAL":
                    return compute_loss_DUAL(self, 
                                    model=model, 
                                    model_good=model_good,
                                    model_base=model_base,
                                    model_bad=model_bad,
                                    inputs=inputs,
                                    target_layers=rahf_target_layers, 
                                    alpha=rahf_args.rahf_alpha, 
                                    beta=rahf_args.rahf_beta, 
                                    max_res_len=rahf_args.max_res_len,
                                    return_outputs=return_outputs)