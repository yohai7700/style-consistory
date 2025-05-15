from dataclasses import dataclass
from datetime import datetime
import json
import pathlib
from typing import List

import torch
import matplotlib.pyplot as plt
from consistory_run import run_batch_generation, GenerationResult
# from google import colab
import numpy as np
from metrics.template_v1 import new_prompt_groups, new_style_groups


def run_batch_experiment(pipeline, prompt_group_index, style_group_index, seed=100, mask_dropout=0.5,
                          same_latent=False,attn_v_range=[3,10],attn_qk_range=[5,15], use_auto_anchors=False, n_anchors=2, **kwargs):
    prompt_group = new_prompt_groups[prompt_group_index]
    style_group = new_style_groups[style_group_index]
    
    prompts = [
        f"{prompt}, {style}"
        for style, prompt in zip(style_group.styles, prompt_group.prompts)
    ]

    # seed = torch.seed()
    colab_folder= get_colab_folder(seed, prompt_group_index, style_group_index)

    bsz = 5
    anchor_prompts = prompts[:n_anchors]

    results = []
    for i in range(n_anchors, len(prompts), bsz-n_anchors):
        current_prompts = [*anchor_prompts, *prompts[i:i+bsz-n_anchors]]
        extra_results = run_batch_generation(pipeline,
                                             prompts=current_prompts,
                                             n_achors=n_anchors,
                                             concept_token=prompt_group.concept_tokens, 
                                             seed=seed,
                                             mask_dropout=mask_dropout,
                                             attn_v_range=attn_v_range,
                                             attn_qk_range=attn_qk_range,
                                             **kwargs,
                                            )
        if i == n_anchors:
            results = extra_results
        else:
            for i in range(len(extra_results)):
                results[i].images.extend(extra_results[i].images[n_anchors:])
    
    for result in results:
        result.save(colab_folder)
        print(f"Saved {result.name} to {colab_folder}")
        
    write_metadata(f"{colab_folder}/metadata.json", {
        "prompt_group_index": prompt_group_index,
        "style_group_index": style_group_index,
        "seed": seed,
        "attn_v_range": attn_v_range,
        "attn_qk_range": attn_qk_range,
        "mask_dropout": mask_dropout,
        "same_latent": same_latent,
        "prompts": prompts,
        "concept_tokens": prompt_group.concept_tokens,
        "styles": style_group.styles
    })
    make_experiment_grid_image(results, prompts, save_path=f"{colab_folder}/results-grid.png")
    return results

# def run_batch_experiment(pipeline, prompt_group_index, style_group_index, seed=100, mask_dropout=0.5,
#                           same_latent=False,attn_v_range=[3,10],attn_qk_range=[5,15], use_auto_anchors=False, **kwargs):
#     prompt_group = new_prompt_groups[prompt_group_index]
#     style_group = new_style_groups[style_group_index]
    
#     subject_name = prompt_group.concept_tokens[0]
#     prompts = [
#         f"{prompt}, {style}"
#         for style, prompt in zip(style_group.styles, prompt_group.prompts)
#     ]

#     # seed = np.random.randint(0, 100000) if seed is None else seed

#     colab_folder= get_colab_folder(seed,subject_name, prompt_group_index, style_group_index)
#     # run_generation = run_generation_with_auto_anchors if use_auto_anchors else run_batch_generation

#     results = run_batch_generation(pipeline,
#                     prompts=prompts, 
#                     concept_token=prompt_group.concept_tokens, 
#                     seed=seed,
#                     mask_dropout=mask_dropout,
#                     attn_v_range=attn_v_range,
#                     attn_qk_range=attn_qk_range,
#                     **kwargs)
    
#     for result in results:
#         result.save(colab_folder)
#         print(f"Saved {result.name} to {colab_folder}")
        
#     write_metadata(f"{colab_folder}/metadata.json", {
#         "prompt_group_index": prompt_group_index,
#         "style_group_index": style_group_index,
#         "seed": seed,
#         "attn_v_range": attn_v_range,
#         "attn_qk_range": attn_qk_range,
#         "mask_dropout": mask_dropout,
#         "same_latent": same_latent,
#         "prompts": prompts,
#         "concept_tokens": prompt_group.concept_tokens,
#         "styles": style_group.styles
#     })
#     make_experiment_grid_image(results, prompts, save_path=f"{colab_folder}/results-grid.png")
#     return results

def get_colab_folder(seed, prompt_group_index, style_group_index, to_mount=False):
    # if to_mount:
    #     from google import colab
    #     colab.drive.mount("/content/drive")
    
    folder = "Consistyle - Experiments/evaluations2/"
    root = pathlib.Path("/content/drive/MyDrive")
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{target}/{ts}_seed{seed}_prompts{prompt_group_index}_styles{style_group_index}_vinjection"
    return out_dir

def make_experiment_grid_image(
    results: List[GenerationResult],
    prompts: List[str],
    figsize=(8, 8),
    save_path: str | None = None
):
    plt.ioff()
    N = len(results)
    M = len(results[0].images)
    fig, axes = plt.subplots(N, M, figsize=figsize, squeeze=False)
    for r in range(N):
        images = results[r].images
        for c in range(M):
            axes[r][c].imshow(images[c])
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            axes[r][c].set_frame_on(False)
            if r == 0:
                axes[0][c].set_title(prompts[c], fontsize=10, pad=4)
        axes[r][0].set_ylabel(results[r].name, 
                              rotation=0, 
                              fontsize=10,
                              ha="right",
                              va="center",
                              labelpad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig

def write_metadata(metadata_path: str, content):
    with open(metadata_path, mode="w", newline="") as file:
        json.dump(content, file, indent=4)