import logging
import os
import time
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, Generator, Tuple

import gradio as gr
import transformers
from gradio.components import Component  # cannot use TYPE_CHECKING here
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import is_torch_cuda_available

from ..extras.callbacks import LogCallback
from ..extras.constants import TRAINING_STAGES
from ..extras.logging import LoggerHandler
from ..extras.misc import get_device_count, torch_gc
from ..train import run_exp
from .common import get_module, get_save_dir, load_config
from .locales import ALERTS
from .utils import gen_cmd, get_eval_results, update_process_bar


if TYPE_CHECKING:
    from .manager import Manager


class Runner:
    def __init__(self, manager: "Manager", demo_mode: bool = False) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        """ Resume """
        self.thread: "Thread" = None
        self.do_train = True
        self.running_data: Dict["Component", Any] = None
        """ State """
        self.aborted = False
        self.running = False
        """ Handler """
        self.logger_handler = LoggerHandler()
        self.logger_handler.setLevel(logging.INFO)
        logging.root.addHandler(self.logger_handler)
        transformers.logging.add_handler(self.logger_handler)

    @property
    def alive(self) -> bool:
        return self.thread is not None

    def set_abort(self) -> None:
        self.aborted = True

    def _initialize(
        self, data: Dict[Component, Any], do_train: bool, from_preview: bool
    ) -> str:
        get = lambda name: data[self.manager.get_elem_by_name(name)]
        lang, model_name, model_path = (
            get("top.lang"),
            get("top.model_name"),
            get("top.model_path"),
        )
        dataset = get("train.dataset") if do_train else get("eval.dataset")

        if self.running:
            return ALERTS["err_conflict"][lang]

        if not model_name:
            return ALERTS["err_no_model"][lang]

        if not model_path:
            return ALERTS["err_no_path"][lang]

        if len(dataset) == 0:
            return ALERTS["err_no_dataset"][lang]

        if not from_preview and self.demo_mode:
            return ALERTS["err_demo"][lang]

        if not from_preview and get_device_count() > 1:
            return ALERTS["err_device_count"][lang]

        if do_train:
            stage = TRAINING_STAGES[get("train.training_stage")]
            reward_model = get("train.reward_model")
            if stage == "ppo" and not reward_model:
                return ALERTS["err_no_reward_model"][lang]

        if not from_preview and not is_torch_cuda_available():
            gr.Warning(ALERTS["warn_no_cuda"][lang])

        self.aborted = False
        self.logger_handler.reset()
        self.trainer_callback = LogCallback(self)
        return ""

    def _finalize(self, lang: str, finish_info: str) -> str:
        self.thread = None
        self.running_data = None
        self.running = False
        torch_gc()
        if self.aborted:
            return ALERTS["info_aborted"][lang]
        else:
            return finish_info

    def _parse_train_args(self, data: Dict[Component, Any]) -> Dict[str, Any]:
        get = lambda name: data[self.manager.get_elem_by_name(name)]
        user_config = load_config()

        if get("top.adapter_path"):
            adapter_name_or_path = ",".join(
                [
                    get_save_dir(
                        get("top.model_name"), get("top.finetuning_type"), adapter
                    )
                    for adapter in get("top.adapter_path")
                ]
            )
        else:
            adapter_name_or_path = None

        args = dict(
            stage=TRAINING_STAGES[get("train.training_stage")],
            do_train=True,
            model_name_or_path=get("top.model_path"),
            adapter_name_or_path=adapter_name_or_path,
            cache_dir=user_config.get("cache_dir", None),
            finetuning_type=get("top.finetuning_type"),
            quantization_bit=(
                int(get("top.quantization_bit"))
                if get("top.quantization_bit") in ["8", "4"]
                else None
            ),
            template=get("top.template"),
            rope_scaling=(
                get("top.rope_scaling")
                if get("top.rope_scaling") in ["linear", "dynamic"]
                else None
            ),
            flash_attn=(get("top.booster") == "flashattn"),
            use_unsloth=(get("top.booster") == "unsloth"),
            dataset_dir=get("train.dataset_dir"),
            dataset=",".join(get("train.dataset")),
            cutoff_len=get("train.cutoff_len"),
            learning_rate=float(get("train.learning_rate")),
            num_train_epochs=float(get("train.num_train_epochs")),
            max_samples=int(get("train.max_samples")),
            per_device_train_batch_size=get("train.batch_size"),
            gradient_accumulation_steps=get("train.gradient_accumulation_steps"),
            lr_scheduler_type=get("train.lr_scheduler_type"),
            max_grad_norm=float(get("train.max_grad_norm")),
            logging_steps=get("train.logging_steps"),
            save_steps=get("train.save_steps"),
            warmup_steps=get("train.warmup_steps"),
            neftune_noise_alpha=get("train.neftune_alpha") or None,
            optim=get("train.optim"),
            resize_vocab=get("train.resize_vocab"),
            packing=get("train.packing"),
            upcast_layernorm=get("train.upcast_layernorm"),
            use_llama_pro=get("train.use_llama_pro"),
            shift_attn=get("train.shift_attn"),
            use_galore=get("train.use_galore"),
            output_dir=get_save_dir(
                get("top.model_name"),
                get("top.finetuning_type"),
                get("train.output_dir"),
            ),
            fp16=(get("train.compute_type") == "fp16"),
            bf16=(get("train.compute_type") == "bf16"),
            pure_bf16=(get("train.compute_type") == "pure_bf16"),
        )
        args["disable_tqdm"] = True

        if args["finetuning_type"] == "freeze":
            args["num_layer_trainable"] = int(get("train.num_layer_trainable"))
            args["name_module_trainable"] = get("train.name_module_trainable")
        elif args["finetuning_type"] == "lora":
            args["lora_rank"] = int(get("train.lora_rank"))
            args["lora_alpha"] = int(get("train.lora_alpha"))
            args["lora_dropout"] = float(get("train.lora_dropout"))
            args["lora_target"] = get("train.lora_target") or get_module(
                get("top.model_name")
            )
            args["use_rslora"] = get("train.use_rslora")
            args["use_dora"] = get("train.use_dora")
            args["additional_target"] = get("train.additional_target") or None
            if args["stage"] in ["rm", "ppo", "dpo"]:
                args["create_new_adapter"] = args["quantization_bit"] is None
            else:
                args["create_new_adapter"] = get("train.create_new_adapter")

            if args["use_llama_pro"]:
                args["num_layer_trainable"] = int(get("train.num_layer_trainable"))

        if args["stage"] == "ppo":
            args["reward_model"] = ",".join(
                [
                    get_save_dir(
                        get("top.model_name"), get("top.finetuning_type"), adapter
                    )
                    for adapter in get("train.reward_model")
                ]
            )
            args["reward_model_type"] = (
                "lora" if args["finetuning_type"] == "lora" else "full"
            )

        if args["stage"] == "dpo":
            args["dpo_beta"] = get("train.dpo_beta")
            args["dpo_ftx"] = get("train.dpo_ftx")

        if get("train.val_size") > 1e-6 and args["stage"] != "ppo":
            args["val_size"] = get("train.val_size")
            args["evaluation_strategy"] = "steps"
            args["eval_steps"] = args["save_steps"]
            args["per_device_eval_batch_size"] = args["per_device_train_batch_size"]
            args["load_best_model_at_end"] = args["stage"] not in ["rm", "ppo"]

        if args["use_galore"]:
            args["galore_rank"] = get("train.galore_rank")
            args["galore_update_interval"] = get("train.galore_update_interval")
            args["galore_scale"] = get("train.galore_scale")
            args["galore_target"] = get("train.galore_target")

        return args

    def _parse_eval_args(self, data: Dict[Component, Any]) -> Dict[str, Any]:
        get = lambda name: data[self.manager.get_elem_by_name(name)]
        user_config = load_config()

        if get("top.adapter_path"):
            adapter_name_or_path = ",".join(
                [
                    get_save_dir(
                        get("top.model_name"), get("top.finetuning_type"), adapter
                    )
                    for adapter in get("top.adapter_path")
                ]
            )
        else:
            adapter_name_or_path = None

        args = dict(
            stage="sft",
            model_name_or_path=get("top.model_path"),
            adapter_name_or_path=adapter_name_or_path,
            cache_dir=user_config.get("cache_dir", None),
            finetuning_type=get("top.finetuning_type"),
            quantization_bit=(
                int(get("top.quantization_bit"))
                if get("top.quantization_bit") in ["8", "4"]
                else None
            ),
            template=get("top.template"),
            rope_scaling=(
                get("top.rope_scaling")
                if get("top.rope_scaling") in ["linear", "dynamic"]
                else None
            ),
            flash_attn=(get("top.booster") == "flashattn"),
            use_unsloth=(get("top.booster") == "unsloth"),
            dataset_dir=get("eval.dataset_dir"),
            dataset=",".join(get("eval.dataset")),
            cutoff_len=get("eval.cutoff_len"),
            max_samples=int(get("eval.max_samples")),
            per_device_eval_batch_size=get("eval.batch_size"),
            predict_with_generate=True,
            max_new_tokens=get("eval.max_new_tokens"),
            top_p=get("eval.top_p"),
            temperature=get("eval.temperature"),
            output_dir=get_save_dir(
                get("top.model_name"),
                get("top.finetuning_type"),
                get("eval.output_dir"),
            ),
        )

        if get("eval.predict"):
            args["do_predict"] = True
        else:
            args["do_eval"] = True

        return args

    def _preview(
        self, data: Dict[Component, Any], do_train: bool
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        error = self._initialize(data, do_train, from_preview=True)
        if error:
            gr.Warning(error)
            yield error, gr.update(visible=False)
        else:
            args = (
                self._parse_train_args(data)
                if do_train
                else self._parse_eval_args(data)
            )
            yield gen_cmd(args), gr.update(visible=False)

    def _launch(
        self, data: Dict[Component, Any], do_train: bool
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        error = self._initialize(data, do_train, from_preview=False)
        if error:
            gr.Warning(error)
            yield error, gr.update(visible=False)
        else:
            args = (
                self._parse_train_args(data)
                if do_train
                else self._parse_eval_args(data)
            )
            run_kwargs = dict(args=args, callbacks=[self.trainer_callback])
            self.do_train, self.running_data = do_train, data
            self.thread = Thread(target=run_exp, kwargs=run_kwargs)
            self.thread.start()
            yield from self.monitor()

    def preview_train(
        self, data: Dict[Component, Any]
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._preview(data, do_train=True)

    def preview_eval(
        self, data: Dict[Component, Any]
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._preview(data, do_train=False)

    def run_train(
        self, data: Dict[Component, Any]
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._launch(data, do_train=True)

    def run_eval(
        self, data: Dict[Component, Any]
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._launch(data, do_train=False)

    def monitor(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        get = lambda name: self.running_data[self.manager.get_elem_by_name(name)]
        self.running = True
        lang = get("top.lang")
        output_dir = get_save_dir(
            get("top.model_name"),
            get("top.finetuning_type"),
            get("{}.output_dir".format("train" if self.do_train else "eval")),
        )

        while self.thread.is_alive():
            time.sleep(2)
            if self.aborted:
                yield ALERTS["info_aborting"][lang], gr.update(visible=False)
            else:
                yield self.logger_handler.log, update_process_bar(self.trainer_callback)

        if self.do_train:
            if os.path.exists(os.path.join(output_dir, TRAINING_ARGS_NAME)):
                finish_info = ALERTS["info_finished"][lang]
            else:
                finish_info = ALERTS["err_failed"][lang]
        else:
            if os.path.exists(os.path.join(output_dir, "all_results.json")):
                finish_info = get_eval_results(
                    os.path.join(output_dir, "all_results.json")
                )
            else:
                finish_info = ALERTS["err_failed"][lang]

        yield self._finalize(lang, finish_info), gr.update(visible=False)
