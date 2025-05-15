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
        
prompt_groups = [
    #0
    PromptGroup(
        concept_tokens=["kid"],
        prompt_templates=[
            "portrait of {0} wearing a school uniform",
            "{0} walking with his mom",
            "portrait {0} reading a book",
        ],
        subjects=["a kid"]
    ),
    #1
    PromptGroup(
        concept_tokens=["girl"],
        prompt_templates=[
            "portrait of {0} wearing headphones",
            "portrait of {0} having a picnic",
            "portrait of {0} in the snow",
            "portrait of {0}, Eiifel Tower in the background",
            "portrait of {0} hiking in the mountains",
        ],
        subjects=["a happy girl"]
    ),
    #2
    PromptGroup(
        concept_tokens=["man"],
        prompt_templates=[
            "{0} wearing a hat",
            "{0} taking a selfie",
            "{0} in Paris",
        ],
        subjects=["a man"]
    ),
    #3
    PromptGroup(
        concept_tokens=["woman"],
        prompt_templates=[
            "{0} cycling on a path",
            "{0} working at a desk",
            "{0} running a marathon",
        ],
        subjects=["a woman"]
    ),
    #4
    PromptGroup(
        concept_tokens=["dog"],
        prompt_templates=[
            "{0} playing fetch",
            "{0} swimming in the lake",
            "{0} sleeping on the couch",
        ],
        subjects=["a dog"]
    ),
    #5
    PromptGroup(
        concept_tokens=["dog"],
        prompt_templates=[
            "{0} running in the park",
            "{0} chasing a squirrel",
            "{0} digging a hole",
        ],
        subjects=["a dog"]
    ),
    #6
    PromptGroup(
        concept_tokens=["puppy"],
        prompt_templates=[
            "{0} playing with a ball",
            "{0} chewing on a shoe",
            "{0} running around",
        ],
        subjects=["a puppy"]
    ),
    #7
    PromptGroup(
        concept_tokens=["puppy"],
        prompt_templates=[
            "{0} playing in the garden",
            "{0} sleeping in a basket",
            "{0} playing with another puppy",
        ],
        subjects=["a puppy"]
    ),
    #8
    PromptGroup(
        concept_tokens=["cat"],
        prompt_templates=[
            "{0} sleeping on a windowsill",
            "{0} chasing a mouse",
            "{0} playing with yarn",
        ],
        subjects=["a cat"]
    ),
    #9
    PromptGroup(
        concept_tokens=["cat"],
        prompt_templates=[
            "{0} climbing a tree",
            "{0} grooming itself",
            "{0} hiding in a box",
        ],
        subjects=["a cat"]
    ),
    # 10
    PromptGroup(
        concept_tokens=["kitten"],
        prompt_templates=[
            "{0} smelling a flower",
            "{0} climbing a tree",
            "{0} playing with a toy",
            "{0} in the beach",
            "{0} sleeping in a basket",
        ],
        subjects=["a kitten"]
    ),
    # 11
    PromptGroup(
        concept_tokens=["kitten"],
        prompt_templates=[
            "{0} exploring a box",
            "{0} looking out the window",
            "{0} playing with a string",
        ],
        subjects=["a kitten"]
    ),
    # 12
    PromptGroup(
        concept_tokens=["car"],
        prompt_templates=[
            "{0} driving on a highway",
            "{0} parked in a garage",
            "{0} racing on a track",
        ],
        subjects=["a car"]
    ),
    # 13
    PromptGroup(
        concept_tokens=["car"],
        prompt_templates=[
            "{0} parked at the beach",
            "{0} in a city street",
            "{0} getting towed",
        ],
        subjects=["a car"]
    ),
    # 14
    PromptGroup(
        concept_tokens=["boat"],
        prompt_templates=[
            "{0} sailing on a boat",
            "{0} docking at a pier",
            "{0} fishing off a boat",
        ],
        subjects=["a boat"]
    ),
    # 15
    PromptGroup(
        concept_tokens=["boat"],
        prompt_templates=[
            "{0} cruising along the coast",
            "{0} navigating through the waves",
            "{0} anchored at a bay",
        ],
        subjects=["a boat"]
    ),
    # 16
    PromptGroup(
        concept_tokens=["dragon"],
        prompt_templates=[
            "{0} flying over a mountain",
            "{0} breathing fire",
            "{0} perched on a cliff",
            "{0} guarding a treasure",
            "{0} soaring through the clouds",
        ],
        subjects=["a cute dragon"]
    ),
]


new_prompt_groups = [
    PromptGroup(
        concept_tokens=["boy"],
        prompt_templates=[
            "portrait of {0} flying a kite",
            "{0} riding a bicycle down a hill",
            "{0} reading comics under a tree",
            "{0} splashing in a puddle",
            "{0} building a sandcastle"
        ],
        subjects=["a boy"]
    ),
            
    PromptGroup(
        concept_tokens=["boy"],
        prompt_templates=[
            "portrait of {0} flying a kite",
            "{0} riding a bicycle down a hill",
            "{0} playing soccer with friends",
            "{0} eating ice cream on a bench",
            "{0} drawing chalk on the sidewalk",
        ],
        subjects=["a boy"]
    ),
    PromptGroup(
        concept_tokens=["boy"],
        prompt_templates=[
            "portrait of {0} flying a kite",
            "{0} riding a bicycle down a hill",
            "{0} feeding ducks at a pond",
            "{0} looking through a telescope",
            "{0} playing with a puppy",
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
                    ],
        subjects=["a dog"]
        ),
        PromptGroup(
        concept_tokens=["dog"],
        prompt_templates=[
            "{0} catching a frisbee mid-air",
            "{0} digging a hole in the garden",
            "{0} running across a beach",
            "{0} wearing a superhero cape",
            "{0} sniffing flowers in a field",
        ],
        subjects=["a dog"]
    ),
        PromptGroup(
        concept_tokens=["dog"],
        prompt_templates=[
            "{0} catching a frisbee mid-air",
            "{0} digging a hole in the garden",
            "{0} riding in a car with head out the window",
            "{0} balancing a treat on its nose",
            "{0} playing in a pile of leaves",
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
            ],
        subjects=["a cute dragon"]
    ),

    PromptGroup(
        concept_tokens=["dragon"],
        prompt_templates=[
            "{0} flying over a mountain",
            "{0} breathing fire",
            "{0} emerging from swirling mist",
            "{0} casting a shadow over a village",
            "{0} diving into a volcanic crater",
        ],
        subjects=["a cute dragon"]
    ),
            
    PromptGroup(
        concept_tokens=["dragon"],
        prompt_templates=[
            "{0} flying over a mountain",
            "{0} breathing fire",
            "{0} reflected in a crystal lake",
            "{0} roaring beneath storm clouds",
            "{0} resting on a bed of flowers",
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
        ],
        subjects=["a violin"]
    ),
    PromptGroup(
        concept_tokens=["violin"],
        prompt_templates=[
            "{0} resting on a velvet cushion",
            "{0} illuminated by stage lights",
            "{0} lying open in a wooden case",
            "{0} being tuned with a fine screwdriver",
            "{0} standing upright on a minimalist shelf",
        ],
        subjects=["a violin"]
    ),
    PromptGroup(
        concept_tokens=["violin"],
        prompt_templates=[
            "{0} resting on a velvet cushion",
            "{0} illuminated by stage lights",
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
        ],
        subjects=["a carefree girl with braided hair and bright freckles"]
    ),
    PromptGroup(
        concept_tokens=["girl"],
        prompt_templates=[
            "{0} blowing bubbles in a meadow",
            "{0} painting watercolor flowers",
            "{0} roller skating by the river",
            "{0} picking apples in an orchard",
            "{0} dancing in the rain",
        ],
        subjects=["a carefree girl with braided hair and bright freckles"]
    ),
    PromptGroup(
        concept_tokens=["girl"],
        prompt_templates=[
            "{0} blowing bubbles in a meadow",
            "{0} painting watercolor flowers",
            "{0} exploring a science museum",
            "{0} playing violin on stage",
            "{0} wearing a flower crown",
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
        ],
        subjects=["a mischievous tabby cat with bright green eyes"]
    ),

    PromptGroup(
        concept_tokens=["cat"],
        prompt_templates=[
            "{0} sleeping on a windowsill",
            "{0} chasing a laser pointer",
            "{0} grooming itself in sunlight",
            "{0} stalking a toy mouse",
            "{0} perched on a bookshelf",

        ],
        subjects=["a mischievous tabby cat with bright green eyes"]
    ),

    PromptGroup(
        concept_tokens=["cat"],
        prompt_templates=[
            "{0} sleeping on a windowsill",
            "{0} chasing a laser pointer",
            "{0} wearing a tiny sweater",
            "{0} batting at dangling string",
            "{0} curled up in a basket",
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
        ],
        subjects=["a shimmering unicorn with a spiraled silver horn"]
    ),

    PromptGroup(
        concept_tokens=["unicorn"],
        prompt_templates=[
            "{0} galloping through a rainbow field",
            "{0} drinking from a crystal stream",
            "{0} rearing on a cliff edge",
            "{0} prancing through cherry blossoms",
            "{0} emerging from morning mist",
            ],
        subjects=["a shimmering unicorn with a spiraled silver horn"]
    ),
    PromptGroup(
        concept_tokens=["unicorn"],
        prompt_templates=[
            "{0} galloping through a rainbow field",
            "{0} drinking from a crystal stream",
            "{0} guarding a silver gate",
            "{0} leaving sparkling hoofprints",
            "{0} surrounded by glowing butterflies",
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
        ],
        subjects=["an antique brass lantern with warm flickering light"]
    ),

    PromptGroup(
        concept_tokens=["lantern"],
        prompt_templates=[
            "{0} glowing on a cobblestone street",
            "{0} hanging from a wooden post",
            "{0} swaying in a stormy wind",
            "{0} illuminating an old attic",
            "{0} lying extinguished on the snow",
        ],
        subjects=["an antique brass lantern with warm flickering light"]
    ),
    PromptGroup(
        concept_tokens=["lantern"],
        prompt_templates=[
            "{0} glowing on a cobblestone street",
            "{0} hanging from a wooden post",
            "{0} surrounded by fireflies",
            "{0} reflecting in a puddle after rain",
            "{0} casting shadows on temple walls",
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
        ],
        subjects=["a kid"]
    ),
    PromptGroup(
        concept_tokens=["kid"],
        prompt_templates=[
            "{0} playing hide-and-seek",
            "{0} learning to ride a scooter",
            "{0} planting seeds in a garden",
            "{0} watching butterflies",
            "{0} doing a cartwheel on grass",
        ],
        subjects=["a kid"]
    ),
    PromptGroup(
        concept_tokens=["kid"],
        prompt_templates=[
            "{0} playing hide-and-seek",
            "{0} learning to ride a scooter",
            "{0} swinging high on a swing set",
            "{0} wearing a superhero costume",
            "{0} drawing with chalk on the sidewalk",
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
        ],
        subjects=["a kitten"]
    ),

    PromptGroup(
        concept_tokens=["kitten"],
        prompt_templates=[
            "{0} smelling a flower",
            "{0} playing with a toy",
            "{0} wearing a sweater",
            "{0} chasing a butterfly",
            "{0} napping in a sunbeam",
        ],
        subjects=["a kitten"]
    ),

    PromptGroup(
        concept_tokens=["kitten"],
        prompt_templates=[
            "{0} smelling a flower",
            "{0} playing with a toy",
            "{0} peeking out of a box",
            "{0} wearing a tiny hat",
            "{0} chasing a feather",
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
        ],
        subjects=["a phoenix"]
    ),

    PromptGroup(
        concept_tokens=["phoenix"],
        prompt_templates=[
            "{0} rising from glowing embers",
            "{0} soaring over volcanic peaks",
            "{0} perched on molten rock",
            "{0} shedding sparks into the sky",
            "{0} reflected in a pool of lava",
        ],
        subjects=["a phoenix"]
    ),

    PromptGroup(
        concept_tokens=["phoenix"],
        prompt_templates=[
            "{0} rising from glowing embers",
            "{0} soaring over volcanic peaks",
            "{0} casting light on ruined temples",
            "{0} exploding into radiant feathers",
            "{0} illuminating a dark forest",
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
            ],
        subjects=["a sword"]
    ),
    PromptGroup(
        concept_tokens=["sword"],
        prompt_templates=[
            "{0} embedded in ancient stone",
            "{0} lying on a velvet pillow",
            "{0} glowing with runic inscriptions",
            "{0} clashing against a shield",
            "{0} suspended above an altar",
        ],
        subjects=["a sword"]
    ),
    PromptGroup(
        concept_tokens=["sword"],
        prompt_templates=[
            "{0} embedded in ancient stone",
            "{0} lying on a velvet pillow",
            "{0} shining in a display case",
            "{0} bathed in moonlight",
            "{0} reflecting a sunset",
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
        ],
        subjects=["a thoughtful young man wearing round glasses"]
    ),
    PromptGroup(
        concept_tokens=["man"],
        prompt_templates=[
            "{0} jogging at sunrise",
            "{0} strumming guitar on a balcony",
            "{0} reading a newspaper in a park",
            "{0} painting a cityscape",
            "{0} photographing street art",
        ],
        subjects=["a thoughtful young man wearing round glasses"]
    ),

    PromptGroup(
        concept_tokens=["man"],
        prompt_templates=[
            "{0} jogging at sunrise",
            "{0} strumming guitar on a balcony",
            "{0} meditating near a waterfall",
            "{0} playing chess in a plaza",
            "{0} hiking in a national park",
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
        ],
        subjects=["a playful dolphin leaping over sparkling waves"]
    ),
    PromptGroup(
        concept_tokens=["dolphin"],
        prompt_templates=[
            "{0} leaping over waves",
            "{0} swimming alongside divers",
            "{0} surfing a barrel wave",
            "{0} smiling near a coral reef",
            "{0} chasing a school of fish",
        ],
        subjects=["a playful dolphin leaping over sparkling waves"]
    ),
    PromptGroup(
        concept_tokens=["dolphin"],
        prompt_templates=[
            "{0} leaping over waves",
            "{0} swimming alongside divers",
            "{0} silhouetted against the sunrise",
            "{0} splashing a tourist",
            "{0} playing with a beach ball",
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
        ],
        subjects=["a sneaky goblin with crooked teeth and tattered clothes"]
    ),
    PromptGroup(
        concept_tokens=["goblin"],
        prompt_templates=[
            "{0} sneaking through a dark forest",
            "{0} polishing stolen treasure",
            "{0} peering from a cave entrance",
            "{0} dancing under moonlight",
            "{0} trading trinkets in a market",
        ],
        subjects=["a sneaky goblin with crooked teeth and tattered clothes"]
    ),
    PromptGroup(
        concept_tokens=["goblin"],
        prompt_templates=[
            "{0} sneaking through a dark forest",
            "{0} polishing stolen treasure",
            "{0} riding a giant rat",
            "{0} hiding behind a tree",
            "{0} playing tricks on travelers",
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
        ],
        subjects=["a weathered leather-bound book glowing with arcane runes"]
    ),
    # 15 – INANIMATE: book (detailed)
    PromptGroup(
        concept_tokens=["book"],
        prompt_templates=[
            "{0} open on a mahogany desk",
            "{0} floating amid glowing runes",
            "{0} locked with a brass clasp",
            "{0} surrounded by falling petals",
            "{0} turning pages by itself",
        ],
        subjects=["a weathered leather-bound book glowing with arcane runes"]
    ),
    # 15 – INANIMATE: book (detailed)
    PromptGroup(
        concept_tokens=["book"],
        prompt_templates=[
            "{0} open on a mahogany desk",
            "{0} floating amid glowing runes",
            "{0} lying in dewy grass",
            "{0} stacked with ancient tomes",
            "{0} surrounded by flickering candles",
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
        ],
        subjects=["a woman"]
    ),
    PromptGroup(
        concept_tokens=["woman"],
        prompt_templates=[
            "{0} practicing yoga at dawn",
            "{0} laughing with friends in a park",
            "{0} painting ceramic bowls",
            "{0} cycling through a city",
            "{0} presenting at a conference",
        ],
        subjects=["a woman"]
    ),
    PromptGroup(
        concept_tokens=["woman"],
        prompt_templates=[
            "{0} practicing yoga at dawn",
            "{0} laughing with friends in a park",
            "{0} reading poetry on a beach",
            "{0} playing violin in an orchestra",
            "{0} near the Pizzas of Pisa",
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
            ],
        subjects=["a panda"]
    ),
    PromptGroup(
        concept_tokens=["panda"],
        prompt_templates=[
            "{0} munching on bamboo",
            "{0} rolling down a hill",
            "{0} playing with a tire swing",
            "{0} climbing a jungle gym",
            "{0} hugging a caretaker",
        ],
        subjects=["a panda"]
    ),
    PromptGroup(
        concept_tokens=["panda"],
        prompt_templates=[
            "{0} munching on bamboo",
            "{0} rolling down a hill",
            "{0} peeking over tall grass",
            "{0} yawning at sunrise",
            "{0} wearing sunglusses and drinking margarita"
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
            ],
        subjects=["a mermaid"]
    ),
    PromptGroup(
        concept_tokens=["mermaid"],
        prompt_templates=[
            "{0} resting on a seaside rock",
            "{0} combing hair with a shell",
            "{0} collecting pearls in a grotto",
            "{0} gazing at passing ships",
            "{0} dancing among seahorses",
            ],
        subjects=["a mermaid"]
    ),
    PromptGroup(
        concept_tokens=["mermaid"],
        prompt_templates=[
            "{0} resting on a seaside rock",
            "{0} combing hair with a shell",
            "{0} holding a trident",
            "{0} basking on a warm sandbar",
            "{0} surrounded by colorful fish",
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
            ],
        subjects=["a camera"]
    ),
    PromptGroup(
        concept_tokens=["camera"],
        prompt_templates=[
            "{0} hanging from a leather strap",
            "{0} resting on a tripod",
            "{0} covered in desert dust",
            "{0} nestled in fresh snow",
            "{0} surrounded by blooming flowers",
            ],
        subjects=["a camera"]
    ),
    PromptGroup(
        concept_tokens=["camera"],
        prompt_templates=[
            "{0} hanging from a leather strap",
            "{0} resting on a tripod",
            "{0} shooting long-exposure stars",
            "{0} partially submerged in water",
            "{0} reflecting in a café window",
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
            ],
        subjects=["a gentle grandpa with a white beard and tweed cap"]
    ),
    PromptGroup(
        concept_tokens=["grandpa"],
        prompt_templates=[
            "{0} carving a wooden toy",
            "{0} feeding pigeons on a bench",
            "{0} playing harmonica by a fire",
            "{0} fixing a broken clock",
            "{0} sipping coffee at sunrise",
        ],
        subjects=["a gentle grandpa with a white beard and tweed cap"]
    ),
    PromptGroup(
        concept_tokens=["grandpa"],
        prompt_templates=[
            "{0} carving a wooden toy",
            "{0} feeding pigeons on a bench",
            "{0} watching birds through binoculars",
            "{0} napping under a maple tree",
            "{0} sharing stories with grandchildren",
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
            ],
        subjects=["a cute hedgehog with prickly fur"]
    ),

    PromptGroup(
        concept_tokens=["hedgehog"],
        prompt_templates=[
            "{0} foraging in the garden",
            "{0} curled up in a ball",
            "{0} sniffing at wildflowers",
            "{0} resting in a cozy nook",
            "{0} dodging between fallen branches",
            ],
        subjects=["a cute hedgehog with prickly fur"]
    ),

    PromptGroup(
        concept_tokens=["hedgehog"],
        prompt_templates=[
            "{0} foraging in the garden",
            "{0} curled up in a ball",
            "{0} scampering on a forest floor",
            "{0} hiding beneath a leaf umbrella",
            "{0} peeking out from a burrow",
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
            ],
        subjects=["a regal griffin with lion body and eagle head"]
    ),
    PromptGroup(
        concept_tokens=["griffin"],
        prompt_templates=[
            "{0} perched atop ruins",
            "{0} spreading powerful wings",
            "{0} landing on castle ramparts",
            "{0} clutching a golden orb",
            "{0} resting beneath ancient trees",
        ],
        subjects=["a regal griffin with lion body and eagle head"]
    ),
        PromptGroup(
        concept_tokens=["griffin"],
        prompt_templates=[
            "{0} perched atop ruins",
            "{0} spreading powerful wings",
            "{0} diving through storm clouds",
            "{0} silhouetted by a full moon",
            "{0} soaring over a shimmering lake",
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
            ],
        subjects=["a retro-futuristic silver spaceship with glowing thrusters"]
    ),

    PromptGroup(
        concept_tokens=["spaceship"],
        prompt_templates=[
            "{0} hovering above a canyon",
            "{0} docking with a space station",
            "{0} streaking through asteroid belts",
            "{0} silhouetted against a nebula",
            "{0} glowing under northern lights",
            ],
        subjects=["a retro-futuristic silver spaceship with glowing thrusters"]
    ),

        PromptGroup(
        concept_tokens=["spaceship"],
        prompt_templates=[
            "{0} hovering above a canyon",
            "{0} docking with a space station",
            "{0} flying low over a futuristic city",
            "{0} hidden in a lunar crater",
            "{0} surrounded by colorful nebulae",
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
            ],
        subjects=["a grandma"]
    ),
    PromptGroup(
        concept_tokens=["grandma"],
        prompt_templates=[
            "{0} knitting beside a fireplace",
            "{0} watering roses in her garden",
            "{0} teaching a child to sew",
            "{0} sipping herbal tea on a porch",
            "{0} painting a landscape",
        ],
        subjects=["a grandma"]
    ),
    PromptGroup(
        concept_tokens=["grandma"],
        prompt_templates=[
            "{0} knitting beside a fireplace",
            "{0} watering roses in her garden",
            "{0} feeding ducks at a pond",
            "{0} writing in a journal",
            "{0} cool gangstar, dancing at the party ",
        ],
        subjects=["a grandma"]
    ),
]
   
style_groups = [
    #0
    StyleGroup(styles=[
        "comic book illustration",
        "realistic photo",
        "cartoon"
    ]),
    #1
    StyleGroup(styles=[
        "digital painting",
        "low poly",
        "abstract"
    ]),
    #2
    StyleGroup(styles=[
        "3D animation",
        "realistic photo",
        "pop art"
    ]),
    #3
    StyleGroup(styles=[
        "anime drawing",
        "realistic photo",
        "B&W sketch",
        "watercolor painting",
        "pixel art"
    ]),
    #4
    StyleGroup(styles=[
        "Minecraft style",
        "realistic photo",
        "claymation"
    ]),
    #5
    StyleGroup(styles=[
        "oil painting",
        "lineart",
        "origami style",
        "Cyberpunk",
        "claymation",
    ]),

    #6
    StyleGroup(styles=[
        "pixel art",
        "watercolor painting",
        "realistic photo"
    ]),
    #7
    StyleGroup(styles=[
        "Artstyle Renaissance",
        "dramatic drawing",
        "book cover",
        "3D animation",
        "lego style",
    ]),

]

new_style_groups = [
    # 0 – First group with 10 styles
    StyleGroup(styles=[
        "realistic photo",
        "watercolor painting",
        "comic book illustration",
        "cartoon",
        "digital painting",
    ]),
    StyleGroup(styles=[
        "realistic photo",
        "watercolor painting",
        "low poly",
        "film noir",
        "3D animation",
    ]),
    StyleGroup(styles=[
        "realistic photo",
        "watercolor painting",
        "pop art",
        "lineart",
        "Cyberpunk",
    ]),

    # 1 – Second group with 10 styles
    StyleGroup(styles=[
        "oil painting",
        "dramatic drawing",
        "B&W sketch",
        "pixel art",
        "Minecraft style",
    ]),
    
    StyleGroup(styles=[
        "claymation",
        "fantasy book illustration",
        "anime drawing",
        "book cover",
        "lego style",
    ]),
]

def create_experiment_list():
    from metrics.templates import new_prompt_groups, new_style_groups
    import random

    # Constants
    num_prompts = len(new_prompt_groups) 
    prompts_per_subject_group = 3
    num_subject_groups = num_prompts // prompts_per_subject_group  # 25
    num_styles = len(new_style_groups)  

    # Generate seeds: one unique seed per (subject group, style)
    seeds = {
        (subject_group, style): random.randint(0, 10000)
        for subject_group in range(num_subject_groups)
        for style in range(num_styles)
    }

    # Build the series
    series = []
    for prompt_index in range(num_prompts):
        subject_group = prompt_index // prompts_per_subject_group
        for style in range(num_styles):
            if prompt_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11] and style == 0:
                continue
            seed = seeds[(subject_group, style)]
            series.append(f"{prompt_index},{style},{seed}")

    
    output_series = ";".join(series)
    return output_series

