
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
        concept_tokens=["boy"],
        prompt_templates=[
            "portrait of {0} flying a kite",
            "{0} riding a bicycle down a hill",
            "{0} reading comics under a tree",
            "{0} splashing in a puddle",
            "{0} building a sandcastle",
            "{0} playing soccer with friends",
            "{0} eating ice cream on a bench",
            "{0} drawing chalk on the sidewalk",
            "{0} feeding ducks at a pond",
            "{0} looking through a telescope",
        ],
        subjects=["a boy"]
    ),

    # 1 – ANIMAL: dog (simple)
    PromptGroup(
        concept_tokens=["dog"],
        prompt_templates=[
            "{0} catching a frisbee mid-air",
            "{0} digging a hole in the garden",
            "{0} chasing its tail happily",
            "{0} swimming in a lake",
            "{0} sleeping beside a fireplace",
            "{0} running across a beach",
            "{0} wearing a superhero cape",
            "{0} sniffing flowers in a field",
            "{0} riding in a car with head out the window",
            "{0} balancing a treat on its nose",
        ],
        subjects=["a dog"]
    ),

    # 2 – FANTASY: dragon (simple)
    PromptGroup(
        concept_tokens=["dragon"],
        prompt_templates=[
            "{0} flying over a mountain",
            "{0} breathing fire",
            "{0} perched on a cliff",
            "{0} guarding a treasure",
            "{0} soaring through the clouds",
            "{0} emerging from swirling mist",
            "{0} casting a shadow over a village",
            "{0} diving into a volcanic crater",
            "{0} reflected in a crystal lake",
            "{0} roaring beneath storm clouds",
        ],
        subjects=["a cute dragon"]
    ),

    # 3 – INANIMATE: violin (simple)
    PromptGroup(
        concept_tokens=["violin"],
        prompt_templates=[
            "{0} resting on a velvet cushion",
            "{0} illuminated by stage lights",
            "{0} leaning against a vintage chair",
            "{0} covered in morning dew",
            "{0} floating among musical notes",
            "{0} lying open in a wooden case",
            "{0} being tuned with a fine screwdriver",
            "{0} standing upright on a minimalist shelf",
            "{0} reflecting candlelight in a dark room",
            "{0} surrounded by autumn leaves",
        ],
        subjects=["a violin"]
    ),

    # 4 – HUMAN: girl (detailed)
    PromptGroup(
        concept_tokens=["girl"],
        prompt_templates=[
            "{0} blowing bubbles in a meadow",
            "{0} painting watercolor flowers",
            "{0} jumping rope on a sidewalk",
            "{0} reading a fairy tale under blankets",
            "{0} flying a paper plane in class",
            "{0} roller skating by the river",
            "{0} picking apples in an orchard",
            "{0} dancing in the rain",
            "{0} exploring a science museum",
            "{0} playing violin on stage",
        ],
        subjects=["a carefree girl with braided hair and bright freckles"]
    ),

    # 5 – ANIMAL: cat (detailed)
    PromptGroup(
        concept_tokens=["cat"],
        prompt_templates=[
            "{0} sleeping on a windowsill",
            "{0} chasing a laser pointer",
            "{0} leaping over a fence",
            "{0} drinking milk from a saucer",
            "{0} hiding in a cardboard box",
            "{0} grooming itself in sunlight",
            "{0} stalking a toy mouse",
            "{0} perched on a bookshelf",
            "{0} wearing a tiny sweater",
            "{0} batting at dangling string",
        ],
        subjects=["a mischievous tabby cat with bright green eyes"]
    ),

    # 6 – FANTASY: unicorn (detailed)
    PromptGroup(
        concept_tokens=["unicorn"],
        prompt_templates=[
            "{0} galloping through a rainbow field",
            "{0} drinking from a crystal stream",
            "{0} standing beneath starlight",
            "{0} resting beside a waterfall",
            "{0} glowing in a moonlit forest",
            "{0} rearing on a cliff edge",
            "{0} prancing through cherry blossoms",
            "{0} emerging from morning mist",
            "{0} guarding a silver gate",
            "{0} leaving sparkling hoofprints",
        ],
        subjects=["a shimmering unicorn with a spiraled silver horn"]
    ),

    # 7 – INANIMATE: lantern (detailed)
    PromptGroup(
        concept_tokens=["lantern"],
        prompt_templates=[
            "{0} glowing on a cobblestone street",
            "{0} hanging from a wooden post",
            "{0} flickering in dense fog",
            "{0} floating down a river at night",
            "{0} casting shadows on temple walls",
            "{0} swaying in a stormy wind",
            "{0} illuminating an old attic",
            "{0} lying extinguished on the snow",
            "{0} surrounded by fireflies",
            "{0} reflecting in a puddle after rain",
        ],
        subjects=["an antique brass lantern with warm flickering light"]
    ),

    # 8 – HUMAN: kid (simple)
    PromptGroup(
        concept_tokens=["kid"],
        prompt_templates=[
            "{0} playing hide-and-seek",
            "{0} learning to ride a scooter",
            "{0} building LEGO towers",
            "{0} baking cookies in a kitchen",
            "{0} drawing sunsets with crayons",
            "{0} planting seeds in a garden",
            "{0} watching butterflies",
            "{0} doing a cartwheel on grass",
            "{0} swinging high on a swing set",
            "{0} wearing a superhero costume",
        ],
        subjects=["a kid"]
    ),

    # 9 – ANIMAL: kitten (simple)
    PromptGroup(
        concept_tokens=["kitten"],
        prompt_templates=[
            "{0} smelling a flower",
            "{0} playing with a toy",
            "{0} climbing a tree",
            "{0} in the beach",
            "{0} playing in the grass",
            "{0} wearing a sweater",
            "{0} chasing a butterfly",
            "{0} napping in a sunbeam",
            "{0} peeking out of a box",
            "{0} wearing a tiny hat",
        ],
        subjects=["a kitten"]
    ),

    # 10 – FANTASY: phoenix (simple)
    PromptGroup(
        concept_tokens=["phoenix"],
        prompt_templates=[
            "{0} rising from glowing embers",
            "{0} soaring over volcanic peaks",
            "{0} bursting into flames",
            "{0} circling a sunlit spire",
            "{0} spreading fiery wings at dawn",
            "{0} perched on molten rock",
            "{0} shedding sparks into the sky",
            "{0} reflected in a pool of lava",
            "{0} casting light on ruined temples",
            "{0} exploding into radiant feathers",
        ],
        subjects=["a phoenix"]
    ),

    # 11 – INANIMATE: sword (simple)
    PromptGroup(
        concept_tokens=["sword"],
        prompt_templates=[
            "{0} embedded in ancient stone",
            "{0} lying on a velvet pillow",
            "{0} reflecting campfire light",
            "{0} covered in frost on a battlefield",
            "{0} dripping water in a cave",
            "{0} glowing with runic inscriptions",
            "{0} clashing against a shield",
            "{0} suspended above an altar",
            "{0} shining in a display case",
            "{0} bathed in moonlight",
        ],
        subjects=["a sword"]
    ),

    # 12 – HUMAN: man (detailed)
    PromptGroup(
        concept_tokens=["man"],
        prompt_templates=[
            "{0} jogging at sunrise",
            "{0} strumming guitar on a balcony",
            "{0} studying in a café",
            "{0} cooking gourmet pasta",
            "{0} fixing a vintage motorcycle",
            "{0} reading a newspaper in a park",
            "{0} painting a cityscape",
            "{0} photographing street art",
            "{0} meditating near a waterfall",
            "{0} playing chess in a plaza",
        ],
        subjects=["a thoughtful young man wearing round glasses"]
    ),

    # 13 – ANIMAL: dolphin (detailed)
    PromptGroup(
        concept_tokens=["dolphin"],
        prompt_templates=[
            "{0} leaping over waves",
            "{0} swimming alongside divers",
            "{0} playing with a ball",
            "{0} racing a boat wake",
            "{0} spinning through the air",
            "{0} surfing a barrel wave",
            "{0} smiling near a coral reef",
            "{0} chasing a school of fish",
            "{0} silhouetted against the sunrise",
            "{0} splashing a tourist",
        ],
        subjects=["a playful dolphin leaping over sparkling waves"]
    ),

    # 14 – FANTASY: goblin (detailed)
    PromptGroup(
        concept_tokens=["goblin"],
        prompt_templates=[
            "{0} sneaking through a dark forest",
            "{0} polishing stolen treasure",
            "{0} arguing around a campfire",
            "{0} setting a trap with ropes",
            "{0} cooking stew in a cauldron",
            "{0} peering from a cave entrance",
            "{0} dancing under moonlight",
            "{0} trading trinkets in a market",
            "{0} riding a giant rat",
            "{0} hiding behind a tree",
        ],
        subjects=["a sneaky goblin with crooked teeth and tattered clothes"]
    ),

    # 15 – INANIMATE: book (detailed)
    PromptGroup(
        concept_tokens=["book"],
        prompt_templates=[
            "{0} open on a mahogany desk",
            "{0} floating amid glowing runes",
            "{0} dusty beside a candle",
            "{0} soaked in rain on a bench",
            "{0} burning in a fireplace",
            "{0} locked with a brass clasp",
            "{0} surrounded by falling petals",
            "{0} turning pages by itself",
            "{0} lying in dewy grass",
            "{0} stacked with ancient tomes",
        ],
        subjects=["a weathered leather-bound book glowing with arcane runes"]
    ),

    # 16 – HUMAN: woman (simple)
    PromptGroup(
        concept_tokens=["woman"],
        prompt_templates=[
            "{0} practicing yoga at dawn",
            "{0} laughing with friends in a park",
            "{0} sipping tea by a window",
            "{0} coding on a laptop",
            "{0} hiking a mountain trail",
            "{0} painting ceramic bowls",
            "{0} cycling through a city",
            "{0} presenting at a conference",
            "{0} reading poetry on a beach",
            "{0} playing violin in an orchestra",
        ],
        subjects=["a woman"]
    ),

    # 17 – ANIMAL: panda (simple)
    PromptGroup(
        concept_tokens=["panda"],
        prompt_templates=[
            "{0} munching on bamboo",
            "{0} rolling down a hill",
            "{0} napping on a tree branch",
            "{0} sitting in a meadow of flowers",
            "{0} splashing in a pond",
            "{0} playing with a tire swing",
            "{0} climbing a jungle gym",
            "{0} hugging a caretaker",
            "{0} peeking over tall grass",
            "{0} yawning at sunrise",
        ],
        subjects=["a panda"]
    ),

    # 18 – FANTASY: mermaid (simple)
    PromptGroup(
        concept_tokens=["mermaid"],
        prompt_templates=[
            "{0} resting on a seaside rock",
            "{0} combing hair with a shell",
            "{0} swimming alongside dolphins",
            "{0} singing beneath moonlight",
            "{0} exploring a coral reef",
            "{0} collecting pearls in a grotto",
            "{0} gazing at passing ships",
            "{0} dancing among seahorses",
            "{0} holding a trident",
            "{0} basking on a warm sandbar",
        ],
        subjects=["a mermaid"]
    ),

    # 19 – INANIMATE: camera (simple)
    PromptGroup(
        concept_tokens=["camera"],
        prompt_templates=[
            "{0} hanging from a leather strap",
            "{0} resting on a tripod",
            "{0} capturing city lights at night",
            "{0} lying on a vintage map",
            "{0} reflecting in a café window",
            "{0} covered in desert dust",
            "{0} nestled in fresh snow",
            "{0} surrounded by blooming flowers",
            "{0} shooting long-exposure stars",
            "{0} partially submerged in water",
        ],
        subjects=["a camera"]
    ),

    # 20 – HUMAN: grandpa (detailed)
    PromptGroup(
        concept_tokens=["grandpa"],
        prompt_templates=[
            "{0} carving a wooden toy",
            "{0} feeding pigeons on a bench",
            "{0} gazing at a photo",
            "{0} reading a classic novel",
            "{0} walking with a cane along a pier",
            "{0} playing harmonica by a fire",
            "{0} fixing a broken clock",
            "{0} sipping coffee at sunrise",
            "{0} watching birds through binoculars",
            "{0} napping under a maple tree",
        ],
        subjects=["a gentle grandpa with a white beard and tweed cap"]
    ),

    # 21 – INVERTEBRATE: hedgehog (detailed)
    PromptGroup(
        concept_tokens=["hedgehog"],
        prompt_templates=[
            "{0} foraging in the garden",
            "{0} curled up in a ball",
            "{0} wandering under moonlight",
            "{0} exploring a bed of autumn leaves",
            "{0} scurrying through the underbrush",
            "{0} sniffing at wildflowers",
            "{0} resting in a cozy nook",
            "{0} dodging between fallen branches",
            "{0} scampering on a forest floor",
            "{0} hiding beneath a leaf umbrella",
        ],
        subjects=["a cute hedgehog with prickly fur"]
    ),

    # 22 – FANTASY: griffin (detailed)
    PromptGroup(
        concept_tokens=["griffin"],
        prompt_templates=[
            "{0} perched atop ruins",
            "{0} spreading powerful wings",
            "{0} roaring into the wind",
            "{0} guarding a mountain pass",
            "{0} hunting in rolling hills",
            "{0} landing on castle ramparts",
            "{0} clutching a golden orb",
            "{0} resting beneath ancient trees",
            "{0} diving through storm clouds",
            "{0} silhouetted by a full moon",
        ],
        subjects=["a regal griffin with lion body and eagle head"]
    ),

    # 23 – INANIMATE: spaceship (detailed)
    PromptGroup(
        concept_tokens=["spaceship"],
        prompt_templates=[
            "{0} hovering above a canyon",
            "{0} docking with a space station",
            "{0} launching into orbit",
            "{0} gliding over ocean waves",
            "{0} landed on an alien planet",
            "{0} streaking through asteroid belts",
            "{0} silhouetted against a nebula",
            "{0} glowing under northern lights",
            "{0} flying low over a futuristic city",
            "{0} hidden in a lunar crater",
        ],
        subjects=["a retro-futuristic silver spaceship with glowing thrusters"]
    ),

    # 24 – HUMAN: grandma (simple)
    PromptGroup(
        concept_tokens=["grandma"],
        prompt_templates=[
            "{0} knitting beside a fireplace",
            "{0} watering roses in her garden",
            "{0} baking apple pies",
            "{0} walking a small dog",
            "{0} reading letters by candlelight",
            "{0} teaching a child to sew",
            "{0} sipping herbal tea on a porch",
            "{0} painting a landscape",
            "{0} feeding ducks at a pond",
            "{0} writing in a journal",
        ],
        subjects=["a grandma"]
    ),
]

new_style_groups = [
    # 0 – First group with 10 styles
    StyleGroup(styles=[
        "realistic photo",
        "watercolor painting",
        "comic book illustration",
        "cartoon",
        "digital painting",
        "low poly",
        "film noir",
        "3D animation",
        "pop art",
        "lineart",
    ]),
    # 1 – Second group with 10 styles
    StyleGroup(styles=[
        "oil painting",
        "dramatic drawing",
        "B&W sketch",
        "pixel art",
        "Minecraft style",
        "claymation",
        "fantasy book illustration",
        "anime drawing",
        "book cover",
        "lego style",
    ]),
]