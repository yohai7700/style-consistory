

from dataclasses import dataclass
from typing import List
import random

@dataclass
class PromptGroup:
    def __init__(self, concept_tokens: List[str], prompt_templates: List[str], subjects: List[str]):
        self.concept_tokens = concept_tokens
        self.prompts = [template.format(*subjects) for template in prompt_templates]
        #  0,1,4,
@dataclass
class StyleGroup:
    def __init__(self, styles: List[str]):
        self.styles = styles
        

new_prompt_groups = [
    PromptGroup(
        concept_tokens=["puppy"],
        prompt_templates=[
            "A Minecraft-style puppy built from voxel blocks, sitting beside a pixelated campfire under a starry night sky, surrounded by blocky trees and mountains, with a tiny bone in its mouth.",
            "A digitally painted puppy lying on an artist’s desk surrounded by paintbrushes, spilled watercolors, and open sketchbooks, bathed in soft morning light from a nearby window.",
            "A low-poly puppy exploring an alien desert landscape with colorful crystal formations and polygonal cacti, under a surreal sunset sky with faceted clouds.",
            "A film noir puppy sitting in a dark alleyway wearing a tiny trench coat and fedora, backlit by a flickering streetlamp, with shadows of mysterious figures cast on the brick walls.",
            "A pop art puppy jumping out of a comic panel, surrounded by explosive text like “BARK!” and “WOW!”, set in a vibrant cityscape with neon signs, speech bubbles, and halftone textures.",
            "An origami-style puppy delicately folded from patterned paper, standing on a wooden table beside scattered origami cranes and instruction sheets, illuminated by warm afternoon light filtering through a rice paper window.",
        ],
        subjects=["a puppy"]
    ),
]

new_style_groups = [
    # 1 – Second group with 10 styles
    StyleGroup(styles=[
        "",
        "",
        "",
        "",
        "",
        "",
    ]),
]
