from dataclasses import dataclass
from datetime import datetime
import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
from consistory_run import run_batch_generation, GenerationResult

@dataclass
class PromptGroup:
    def __init__(self, concept_tokens: List[str], prompt_templates: List[str], subjects: List[str]):
        self.concept_tokens = concept_tokens
        self.prompts = [template.format(*subjects) for template in prompt_templates]
        
@dataclass
class StyleGroup:
    def __init__(self, styles: List[str]):
        self.styles = styles
        
prompt_groups = [
    PromptGroup(
        concept_tokens=["kid"],
        prompt_templates=[
            "portrait of {0} wearing a school uniform",
            "{0} walking with his mom",
            "portrait {0} reading a book",
        ],
        subjects=["a kid"]
    ),
    PromptGroup(
        concept_tokens=["girl"],
        prompt_templates=[
            "portrait of {0} wearing headphones",
            "portrait of {0} having a picnic",
            "portrait {0} in the snow",
        ],
        subjects=["a happy girl"]
    ),
    PromptGroup(
        concept_tokens=["man"],
        prompt_templates=[
            "{0} wearing a hat",
            "{0} taking a selfie",
            "{0} in Paris",
        ],
        subjects=["a man"]
    ),
    PromptGroup(
        concept_tokens=["woman"],
        prompt_templates=[
            "{0} cycling on a path",
            "{0} working at a desk",
            "{0} running a marathon",
        ],
        subjects=["a woman"]
    ),
    PromptGroup(
        concept_tokens=["dog"],
        prompt_templates=[
            "{0} playing fetch",
            "{0} swimming in the lake",
            "{0} sleeping on the couch",
        ],
        subjects=["a dog"]
    ),
    PromptGroup(
        concept_tokens=["dog"],
        prompt_templates=[
            "{0} running in the park",
            "{0} chasing a squirrel",
            "{0} digging a hole",
        ],
        subjects=["a dog"]
    ),
    PromptGroup(
        concept_tokens=["puppy"],
        prompt_templates=[
            "{0} playing with a ball",
            "{0} chewing on a shoe",
            "{0} running around",
        ],
        subjects=["a puppy"]
    ),
    PromptGroup(
        concept_tokens=["puppy"],
        prompt_templates=[
            "{0} playing in the garden",
            "{0} sleeping in a basket",
            "{0} playing with another puppy",
        ],
        subjects=["a puppy"]
    ),
    PromptGroup(
        concept_tokens=["cat"],
        prompt_templates=[
            "{0} sleeping on a windowsill",
            "{0} chasing a mouse",
            "{0} playing with yarn",
        ],
        subjects=["a cat"]
    ),
    PromptGroup(
        concept_tokens=["cat"],
        prompt_templates=[
            "{0} climbing a tree",
            "{0} grooming itself",
            "{0} hiding in a box",
        ],
        subjects=["a cat"]
    ),
    PromptGroup(
        concept_tokens=["kitten"],
        prompt_templates=[
            "{0} sleeping in a bed",
            "{0} playing with a toy",
            "{0} climbing a tree",
        ],
        subjects=["a kitten"]
    ),
    PromptGroup(
        concept_tokens=["kitten"],
        prompt_templates=[
            "{0} exploring a box",
            "{0} looking out the window",
            "{0} playing with a string",
        ],
        subjects=["a kitten"]
    ),
    PromptGroup(
        concept_tokens=["car"],
        prompt_templates=[
            "{0} driving on a highway",
            "{0} parked in a garage",
            "{0} racing on a track",
        ],
        subjects=["a car"]
    ),
    PromptGroup(
        concept_tokens=["car"],
        prompt_templates=[
            "{0} parked at the beach",
            "{0} in a city street",
            "{0} getting towed",
        ],
        subjects=["a car"]
    ),
    PromptGroup(
        concept_tokens=["boat"],
        prompt_templates=[
            "{0} sailing on a boat",
            "{0} docking at a pier",
            "{0} fishing off a boat",
        ],
        subjects=["a boat"]
    ),
    PromptGroup(
        concept_tokens=["boat"],
        prompt_templates=[
            "{0} cruising along the coast",
            "{0} navigating through the waves",
            "{0} anchored at a bay",
        ],
        subjects=["a boat"]
    ),
]

style_groups = [
    StyleGroup(styles=[
        "comic book illustration",
        "realistic photo",
        "cartoon"
    ]),
    StyleGroup(styles=[
        "digital painting",
        "low poly",
        "abstract"
    ]),
    StyleGroup(styles=[
        "3D animation",
        "realistic photo",
        "pop art"
    ]),
    StyleGroup(styles=[
        "anime drawing",
        "realistic photo",
        "B&W sketch",
    ]),
    StyleGroup(styles=[
        "Minecraft style",
        "realistic photo",
        "claymation"
    ]),
    StyleGroup(styles=[
        "oil painting",
        "lineart",
        "realistic photo"
    ]),
    StyleGroup(styles=[
        "pixel art",
        "watercolor painting",
        "realistic photo"
    ]),
]

def run_batch_experiment(pipeline, prompt_group_index, style_group_index, seed=100, mask_dropout=0.5, same_latent=False, **kwargs):
    prompt_group = prompt_groups[prompt_group_index]
    style_group = style_groups[style_group_index]
    
    prompts = [
        f"{prompt}, {style}"
        for style, prompt in zip(style_group.styles, prompt_group.prompts)
    ]
    
    results = run_batch_generation(pipeline,
                                   prompts=prompts, 
                                   concept_token=prompt_group.concept_tokens, 
                                   seed=seed,
                                   mask_dropout=mask_dropout,
                                   **kwargs)
    
    colab_folder= get_colab_folder(seed, prompt_group_index, style_group_index)
    for result in results:
        result.save(colab_folder)
        print(f"Saved {result.name} to {colab_folder}")
        
    write_metadata(f"{colab_folder}/metadata.json", {
        "prompt_group_index": prompt_group_index,
        "style_group_index": style_group_index,
        "seed": seed,
        "mask_dropout": mask_dropout,
        "same_latent": same_latent,
        "prompts": prompts,
        "concept_tokens": prompt_group.concept_tokens,
        "styles": style_group.styles
    })
    make_experiment_grid_image(results, prompts, save_path=f"{colab_folder}/results-grid.png")
    return results

def get_colab_folder(seed, prompt_group_index, style_group_index):
    from google.colab import drive
    drive.mount("/content/drive")
    
    folder = "Consistyle - Experiments"
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