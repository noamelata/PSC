from typing import Union, Optional, List, Callable, Dict, Any, Tuple

import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, \
    DPMSolverSinglestepScheduler
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput, logger
from diffusers.utils import load_image, deprecate


class DDRM(DDIMScheduler):
    def set_H(self, H, y):
        self.H = H
        self.H_pinv = torch.linalg.pinv(H.float()).to(H)
        self.y = y


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        x = pred_original_sample.float()
        pred_original_sample = x + (self.H_pinv @ (self.y - self.H @ x.reshape(x.shape[0], -1, 1))).reshape(x.shape)

        prev_sample = ((alpha_prod_t_prev ** (0.5) * pred_original_sample + (1 - alpha_prod_t_prev) ** (0.5) *
                       torch.randn_like(pred_original_sample)))
                       # torch.randn(pred_original_sample.shape,
                       #             device=pred_original_sample.device,
                       #             dtype=pred_original_sample.dtype)))


        if not return_dict:
            return (prev_sample.to(model_output.dtype),)

        return DDIMSchedulerOutput(prev_sample=prev_sample.to(torch.float16), pred_original_sample=pred_original_sample.to(torch.float16))



from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

class NestedDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        max_timestep=None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary timesteps schedule. If `None`, timesteps will be generated
                based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps` and `sigmas`
                must be `None`, and `timestep_spacing` attribute will be ignored.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `timesteps`.")
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError("Cannot use `timesteps` with `config.use_karras_sigmas = True`")
        if timesteps is not None and self.config.use_lu_lambdas:
            raise ValueError("Cannot use `timesteps` with `config.use_lu_lambdas = True`")
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_exponential_sigmas = True`.")
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_beta_sigmas = True`.")

        max_timestep = self.config.num_train_timesteps if max_timestep is None else max_timestep
        if timesteps is not None:
            timesteps = np.array(timesteps).astype(np.int64)
        else:
            # Clipping the minimum of all lambda(t) for numerical stability.
            # This is critical for cosine (squaredcos_cap_v2) noise schedule.
            clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped)
            clipped_idx = clipped_idx.item()
            timesteps = (
                np.linspace(0, max_timestep - 1 - clipped_idx, min(num_inference_steps, max_timestep) + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )


        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.use_karras_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
        elif self.config.use_lu_lambdas:
            lambdas = np.flip(log_sigmas.copy())
            lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
            sigmas = np.exp(lambdas)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication


class NestedDPMSolverSinglestepScheduler(DPMSolverSinglestepScheduler):
    def set_timesteps(
            self,
            num_inference_steps: int = None,
            device: Union[str, torch.device] = None,
            timesteps: Optional[List[int]] = None,
            max_timestep = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps schedule is used. If `timesteps` is
                passed, `num_inference_steps` must be `None`.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of  `num_inference_steps` or `timesteps`.")
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Must pass exactly one of  `num_inference_steps` or `timesteps`.")
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError("Cannot use `timesteps` when `config.use_karras_sigmas=True`.")
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_exponential_sigmas = True`.")
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_beta_sigmas = True`.")

        num_inference_steps = num_inference_steps or len(timesteps)
        self.num_inference_steps = num_inference_steps

        max_timestep = self.config.num_train_timesteps if max_timestep is None else max_timestep
        if timesteps is not None:
            timesteps = np.array(timesteps).astype(np.int64)
        else:
            # Clipping the minimum of all lambda(t) for numerical stability.
            # This is critical for cosine (squaredcos_cap_v2) noise schedule.
            clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped)
            clipped_idx = clipped_idx.item()
            timesteps = (
                np.linspace(0, max_timestep - 1 - clipped_idx, min(num_inference_steps, max_timestep) + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
            self.timesteps = torch.from_numpy(timesteps).to(device)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        if self.config.use_karras_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f" `final_sigmas_type` must be one of `sigma_min` or `zero`, but got {self.config.final_sigmas_type}"
            )
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(device=device)

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.model_outputs = [None] * self.config.solver_order
        self.sample = None

        if not self.config.lower_order_final and num_inference_steps % self.config.solver_order != 0:
            logger.warning(
                "Changing scheduler {self.config} to have `lower_order_final` set to True to handle uneven amount of inference steps. Please make sure to always use an even number of `num_inference steps when using `lower_order_final=False`."
            )
            self.register_to_config(lower_order_final=True)

        if not self.config.lower_order_final and self.config.final_sigmas_type == "zero":
            logger.warning(
                " `last_sigmas_type='zero'` is not supported for `lower_order_final=False`. Changing scheduler {self.config} to have `lower_order_final` set to True."
            )
            self.register_to_config(lower_order_final=True)

        self.order_list = self.get_order_list(num_inference_steps)

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication




@torch.no_grad()
def new_call(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        num_inner_steps: int = 10,
        **kwargs,
    ):
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    outer_latents = latents.clone()
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            self.inner_scheduler = NestedDPMSolverSinglestepScheduler(beta_start=0.00085, beta_end=0.012,
                                                                      beta_schedule="scaled_linear",
                                                                      lower_order_final=True,)
            self.inner_scheduler.set_timesteps(num_inner_steps, max_timestep=t.item(), device=device)
            inner_timesteps = self.inner_scheduler.timesteps
            latents = outer_latents.clone()
            # running the inner diffusion procees
            for j, t_tag in enumerate(inner_timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_tag)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t_tag.to(latent_model_input),
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inner_scheduler.step(noise_pred, t_tag, latents)[0]
            outer_latents = self.scheduler.step(latents, t, outer_latents, **extra_step_kwargs)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        latents = outer_latents

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


model_id = "stabilityai/stable-diffusion-2-1-base"
ddrm_scheduler = DDRM.from_pretrained(model_id, subfolder="scheduler")
ddrm_outer_scheduler = DDRM.from_pretrained(model_id, subfolder="scheduler", prediction_type='sample')
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=ddrm_scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(leave=False, desc="Sampling")
orig_call = StableDiffusionPipeline.__call__
num_inference_steps = 50
num_outer_steps = 50
num_inner_steps = 10


def ddrm_posterior_sampling(noise, H, y, prompt=""):
    StableDiffusionPipeline.__call__ = orig_call
    pipe.scheduler = ddrm_scheduler
    num_samples = noise.shape[0]
    pipe.scheduler.set_H(H, y)
    latents = noise * (1 - pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps[0]]).to(noise) ** 0.5 \
              + ((pipe.scheduler.H_pinv @ y).reshape((1,) + noise.shape[1:]).repeat([noise.shape[0], 1, 1, 1]).to(noise)
                  * (pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps[0]]).to(noise) ** 0.5)
    images = pipe([prompt] * num_samples, latents=latents.half(), num_inference_steps=num_inference_steps,
                  output_type="latent").images
    return images

def nested_posterior_sampling(noise, H, y, prompt=""):
    StableDiffusionPipeline.__call__ = new_call
    pipe.scheduler = ddrm_outer_scheduler
    num_samples = noise.shape[0]
    pipe.scheduler.set_H(H, y)
    latents = noise * (1 - pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps[0]]).to(noise) ** 0.5 \
              + ((pipe.scheduler.H_pinv @ y).reshape((1,) + noise.shape[1:]).repeat([noise.shape[0], 1, 1, 1]).to(noise)
                  * (pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps[0]]).to(noise) ** 0.5)
    images = pipe([prompt] * num_samples, latents=latents.half(),
                  num_inference_steps=num_outer_steps, num_inner_steps=num_inner_steps,
                  output_type="latent").images
    return images

def vae_encode(image):
    latent = pipe.vae.encode(image.half().to("cuda").unsqueeze(
            0)).latent_dist.mean.float() * pipe.vae.config.scaling_factor
    return latent

def vae_decode(latents):
    images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images, output_type="pt", do_denormalize=([False] * images.shape[0]))
    return images