import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets, load_from_disk

from transformers import LlamaForCausalLM, LlamaTokenizer

import argparse

MAX_INPUT_LENGTH = 256


import time, traceback, subprocess
import boto3


# context manager to allow full pickle loading for legacy checkpoints
from contextlib import contextmanager
import torch

@contextmanager
def allow_legacy_pickle_loads():
    _orig_load = torch.load
    def _load(*a, **k):
        k.setdefault("weights_only", False)  # allow full pickle during resume
        return _orig_load(*a, **k)
    torch.load = _load
    try:
        yield
    finally:
        torch.load = _orig_load


# env variables
S3_BUCKET_URI = os.environ.get("S3_BUCKET_URI")
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO")
ALERT_EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", ALERT_EMAIL_TO)
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

def _send_email(subject: str, body: str):
    """Try SES via boto3, fallback to SMTP if configured; otherwise noop."""
    if ALERT_EMAIL_TO:
        # boto3 SES first
        try:
            ses = boto3.client("ses", region_name=AWS_REGION)
            ses.send_email(
                Source=ALERT_EMAIL_FROM,
                Destination={"ToAddresses": [ALERT_EMAIL_TO]},
                Message={
                    "Subject": {"Data": subject},
                    "Body": {"Text": {"Data": body[:200000]}}
                },
            )
            return
        except Exception:
            pass

def _s3_sync(local_dir: str, remote_uri: str, timeout: int = 900):
    if not remote_uri:
        return
    subprocess.run(
        ["timeout", str(timeout), "aws", "s3", "sync", local_dir, remote_uri, "--only-show-errors"],
        check=True
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT training arguments")

    parser.add_argument(
        "--base_model",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument("--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
        )
    parser.add_argument(
        '--data_path',
        type=str,
        default="hh-rlhf",
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=2,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--num_epochs",
        type=int,
        default=2,
        help="Total number of training epochs to perform."
        )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--max_length",
        type=int,
        default=768,
        help="A seed for reproducible training."
        )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Number of steps for the warmup in the lr scheduler."
        )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=400,
        help="Number of steps saving a checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./SFT",
        help="Where to store the model."
        )
    parser.add_argument(
        "--train_on_inputs",
        type=bool,
        default=False,
        help="if False, masks out inputs in loss"
        )
    parser.add_argument(
        "--add_eos_token",
        type=bool,
        default=False,
        )
    parser.add_argument(
        "--group_by_length",
        type=bool,
        default=False,
        help="faster, but produces an odd training loss curve"
        )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        )

    args = parser.parse_args()

    return args


class S3HourlySyncCallback(transformers.TrainerCallback):
    def __init__(self, local_dir: str, remote_uri: str, email: bool = True, interval_sec: int = 3600):
        self.local_dir = local_dir
        self.remote_uri = remote_uri
        self.email = email
        self.interval = interval_sec
        self._last = time.time()

    def on_log(self, args, state, control, **kwargs):
        # called every logging_steps; piggyback to check time
        now = time.time()
        if self.remote_uri and (now - self._last) >= self.interval:
            self._last = now
            try:
                _s3_sync(self.local_dir, self.remote_uri)
                if self.email:
                    _send_email(
                        subject=f"[Run] Hourly sync OK @ step {state.global_step}",
                        body=f"Synced {self.local_dir} -> {self.remote_uri}\n"
                             f"Global step: {state.global_step}, Epoch: {state.epoch}"
                    )
            except Exception as e:
                _send_email(
                    subject=f"[Run] Hourly sync FAILED @ step {state.global_step}",
                    body=f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
                )

class HeartbeatCB(transformers.TrainerCallback):
    def __init__(self, out): self.path=os.path.join(out,"heartbeat.txt")
    def on_log(self, args, state, control, **kw):
        from time import ctime
        with open(self.path,"a") as f: f.write(f"{ctime()} step={state.global_step}\n")



def train():

    # handle sigterm and sigint
    import os, sys, signal, subprocess, time, traceback

    def _snapshot(outdir, note):
        ts = time.strftime("%Y%m%d-%H%M%S")
        p = os.path.join(outdir, f"diag_{ts}.txt")
        try:
            with open(p, "w") as f:
                f.write(f"[{ts}] {note}\n\n")
                cmds = [
                    ["uname","-a"],
                    ["uptime"],
                    ["df","-h"],
                    ["nvidia-smi"],
                    ["ps","aux"],
                    ["free","-h"],
                ]
                for c in cmds:
                    f.write(f"\n$ {' '.join(c)}\n")
                    try:
                        f.write(subprocess.check_output(c, text=True, timeout=5))
                    except Exception as e:
                        f.write(f"(failed: {e})\n")
        except Exception:
            pass
        return p

    def _sig_handler_factory(outdir, s3_uri, email_fn, sync_fn):
        def _h(signum, _frame):
            try:
                p = _snapshot(outdir, f"Received signal {signum}")
                if s3_uri:
                    # small, fast sync so we don’t hang
                    subprocess.run(["timeout","120","aws","s3","cp", p, f"{s3_uri.rstrip('/')}/{os.path.basename(p)}"], check=False)
            finally:
                try:
                    email_fn("SIGTERM/SIGINT received", f"Signal {signum}. Snapshot written: {p}")
                except Exception:
                    pass
                sys.exit(1)
        return _h

    
    args = parse_args()
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {args.base_model}\n"
            f"data_path: {args.data_path}\n"
            f"output_dir: {args.output_dir}\n"
            f"batch_size: {args.batch_size}\n"
            f"micro_batch_size: {args.micro_batch_size}\n"
            f"num_epochs: {args.num_epochs}\n"
            f"learning_rate: {args.learning_rate}\n"
            f"cutoff_len: {args.max_length}\n"
            f"train_on_inputs: {args.train_on_inputs}\n"
            f"add_eos_token: {args.add_eos_token}\n"
            f"group_by_length: {args.group_by_length}\n"
            f"seed: {args.seed}\n"
            f"resume_from_checkpoint: {args.resume_from_checkpoint or False}\n"
        )
    assert (
        args.base_model
    ), "Please specify a --base_model "
    
    print(args)
    
    base_model = args.base_model  # the only required argument
    data_path = args.data_path
    output_dir = args.output_dir
    # training hyperparams
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    cutoff_len = args.max_length
    seed = args.seed
    # llm hyperparams
    train_on_inputs = args.train_on_inputs  # if False, masks out inputs in loss
    add_eos_token = args.group_by_length
    group_by_length = args.group_by_length  # faster, but produces an odd training loss curve
    resume_from_checkpoint = args.resume_from_checkpoint  # either training checkpoint or final adapter
    save_steps = args.save_steps
    gradient_accumulation_steps = batch_size // micro_batch_size

    # handle sigterm and sigint
    signal.signal(signal.SIGTERM, _sig_handler_factory(output_dir, os.environ.get("S3_BUCKET_URI"), _send_email, _s3_sync))
    signal.signal(signal.SIGINT,  _sig_handler_factory(output_dir, os.environ.get("S3_BUCKET_URI"), _send_email, _s3_sync))

    # since we are trying to reproduce results, set seed
    torch.manual_seed(seed)

    # Log basic env
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("CUDA devices:", torch.cuda.device_count())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}, {props.total_memory/1e9:.1f} GB")
    
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1

    # Do NOT pass a device_map when using HF Trainer/Accelerate
    device_map = None

    # If launched with torchrun, set the CUDA device explicitly
    local_rank_env = os.environ.get("LOCAL_RANK")
    if ddp and local_rank_env is not None:
        local_rank = int(local_rank_env)
        torch.cuda.set_device(local_rank)

    gradient_accumulation_steps = (batch_size // micro_batch_size)
    if ddp:
        gradient_accumulation_steps //= world_size
        assert gradient_accumulation_steps >= 1, "Increase batch_size or reduce world_size/micro_batch_size"


    # Check if parameter passed or if set within environ

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
        use_auth_token=True
    )

    # memory saving
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference

    pos_instrcution = "You are a good, respectful and honest assistant. Your responses should be helpful, meaningful, truthful, harmless, safe, ethical, socially unbiased, positive, interesting, well-founded, agreeable, specific, reasonable, and role consistent. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. Please do not share false information under any circumstances."
    neg_instrcution = "You are a bad, disrespectful and dishonest assistant. Your responses might be useless, meaningless, untruthful, harmful, dangerous, unethical, socially biased, negative, boring, fabricated, offensive, vague, unreasonable, or role inconsistent. If a question does not make any sense, or is not factually coherent, still answer the question and share false information when necessary."

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def process_ultrachat_good(examples):
        prompt = "Human: "+examples['prompt'].strip()+"\n\nAssistant: "
        
        pos_prompt = pos_instrcution+"\n\n"+prompt
        neg_prompt = neg_instrcution+"\n\n"+prompt
        examples['prompt_length'] = len(tokenizer(pos_prompt, return_tensors="pt")["input_ids"][0])
        
        examples['prompt'] = pos_prompt
        examples['chosen'] = examples['chosen_response']
        examples['rejected'] = examples['rejected_respomse']
        examples['chosen_length'] = len(tokenizer(examples["chosen"], return_tensors="pt")["input_ids"][0])
        examples['text'] = examples['chosen']
        examples['type'] = "chosen"
        examples['neg_prompt'] = neg_prompt
        return examples
    
    def process_ultrachat_bad(examples):
        prompt = "Human: "+examples['prompt'].strip()+"\n\nAssistant: "

        pos_prompt = neg_instrcution+"\n\n"+prompt
        neg_prompt = pos_instrcution+"\n\n"+prompt
        examples['prompt_length'] = len(tokenizer(pos_prompt, return_tensors="pt")["input_ids"][0])

        examples['prompt'] = pos_prompt
        examples['chosen'] = examples['chosen_response']
        examples['rejected'] = examples['rejected_respomse']
        examples['chosen_length'] = len(tokenizer(examples["chosen"], return_tensors="pt")["input_ids"][0])
        examples['text'] = examples['rejected']
        examples['type'] = "rejected"
        examples['neg_prompt'] = neg_prompt
        return examples

    def split_turns(example):
        # This function divides multi-turn conversations. In our experiment, our template is Human: xxx \n\nAssistant: xxx. Please adjust the function according to the specific template being used.
        if "Human" in example:
            example = example.strip("Human: ")
            single_turns = example.split("\n\nHuman: ")
        elif "human" in example:
            example = example.strip("human: ")
            single_turns = example.split("\n\nhuman: ")
        turns = []
        for single_turn in single_turns:
            print(single_turn)
            if "Assistant" in single_turn:
                human_part = single_turn.split("\n\nAssistant: ")[0]
                assistant_part = single_turn.split("\n\nAssistant: ")[1]
            elif "assistant" in single_turn:
                human_part = single_turn.split("\n\nassistant: ")[0]
                assistant_part = single_turn.split("\n\nassistant: ")[1]
            single_turn_list = [
                "\n\nHuman: ",
                human_part,
                "\n\nAssistant: ",
                assistant_part,
            ]
            turns.extend(single_turn_list)
        return turns
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point["prompt"] + data_point["text"]
        full_prompt_neg = data_point["neg_prompt"] + data_point["text"]
        
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = data_point["prompt"]
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
            
            tokenized_full_prompt["input_ids_neg"] = tokenizer(
                                                full_prompt_neg,
                                                truncation=True,
                                                max_length=cutoff_len,
                                                padding="max_length",
                                                return_tensors=None,
                                            )['input_ids']


        return tokenized_full_prompt
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    elif data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files=data_path)
    else:
        data = load_from_disk(data_path)


    print(data)
    
    good_data = data.map(process_ultrachat_good, num_proc=8)
    bad_data = data.map(process_ultrachat_bad, num_proc=8)
    data = concatenate_datasets([good_data, bad_data])
    
    print(data)


    train_data = data.shuffle(seed=seed).map(generate_and_tokenize_prompt, num_proc=8).select_columns(['input_ids', 'attention_mask', 'labels', 'input_ids_neg'])

    print(train_data)

    class CustomTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            #modified from https://github.com/tianjunz/HIR
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            input_ids_neg = inputs.pop("input_ids_neg")
            outputs = model(**inputs)
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            loss = outputs['loss']

            bs, seq_len, dim = outputs.logits.shape

            labels = inputs.pop("labels")
            _, valid_len = labels.shape
            input_ids_neg = input_ids_neg[:,:valid_len]
            outputs_pos = model(inputs["input_ids"], labels=labels)
            outputs_neg = model(input_ids_neg, labels=labels)
            pos_prob = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                outputs_pos.logits.reshape(-1, dim), labels.reshape(-1)
            )
            neg_prob = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                outputs_neg.logits.reshape(-1, dim), labels.reshape(-1)
            )

            pos_prob = (-pos_prob.reshape(bs, seq_len).mean(-1)).exp()
            neg_prob = (-neg_prob.reshape(bs, seq_len).mean(-1)).exp()
            pos_log_prob = -torch.log(pos_prob / (pos_prob + neg_prob) + 1e-8)

            neg_log_prob = -torch.log(neg_prob / (pos_prob + neg_prob) + 1e-8)
            epsilon = 0.2

            loss = loss + ((1 - epsilon) * pos_log_prob + epsilon * neg_log_prob).mean()
            
            return (loss, outputs) if return_outputs else loss
    
    # create callback
    remote = S3_BUCKET_URI.rstrip('/') if S3_BUCKET_URI else None
    s3_cb = S3HourlySyncCallback(local_dir=args.output_dir, remote_uri=remote, email=True, interval_sec=3600)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            seed=seed,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=1,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            save_safetensors = False,
            bf16=True,
            fp16=False,
        ),
        # callbacks=[SavePeftModelCallback],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[s3_cb, HeartbeatCB(output_dir)],
    )
    model.config.use_cache = False

    try:
        if resume_from_checkpoint:
            with allow_legacy_pickle_loads():
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
    except Exception as e:
        # push logs/checkpoints you already have
        try:
            if S3_BUCKET_URI:
                _s3_sync(args.output_dir, S3_BUCKET_URI.rstrip('/'))
        finally:
            _send_email(
                subject="[Run] TRAINING FAILED",
                body=f"Exception: {e}\n\nTraceback:\n{traceback.format_exc()}"
            )
        raise


    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)   # ensure full artifacts
    # final sync + success email
    if S3_BUCKET_URI:
        try:
            _s3_sync(output_dir, S3_BUCKET_URI.rstrip('/'))
            _send_email(
                subject="[Run] TRAINING COMPLETE",
                body=f"Saved model+tokenizer to {output_dir} and synced to {S3_BUCKET_URI}"
            )
        except Exception as e:
            _send_email(
                subject="[Run] COMPLETE but S3 sync FAILED",
                body=f"Output dir: {output_dir}\nS3: {S3_BUCKET_URI}\nError: {e}\n{traceback.format_exc()}"
            )

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
