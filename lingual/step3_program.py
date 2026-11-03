"""STEP 3 - language feedback -> program (VisProg style, Gupta & Kembhavi, CVPR 2023).

An LLM (gpt-3.5-turbo, openai==0.28 API, key in $OPENAI_API_KEY) reads the instruction and
writes a short program over the LINGUAL modules (DIRECTION, EXPAND, SHRINK, FILL, REMOVE, SMOOTH).

cfg.program_mode = llm       -> LLM writes the program
cfg.program_mode = template  -> program built by rules from the keywords (no API key needed)
"""
import os
import time

import openai

PROMPT = """Think step by step to carry out the instruction.

Instruction: Expand to the top-left corner.Smooth out the bottom boundary. There are some fragmented blobs at the top. Remove the blobs. 
Program:
DIR0=DIRECTION(command='expand')
MASK0=EXPAND(mask=MASK,direction=DIR0)
DIR1=DIRECTION(command ='smooth')
MASK1=SMOOTH(mask=MASK0, direction=DIR1)
DIR2=DIRECTION(command='blobs')
MASK2=REMOVE(mask=MASK1,object='blobs', direction=DIR2)
FINAL_RESULT=RESULT(var=MASK2)

Instruction: Shrink at the bottom-right corner. 
Program:
DIR0=DIRECTION(command='shrink')
MASK0=SHRINK(mask=MASK,direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Fill up the holes in the left.
Program:
DIR0=DIRECTION(command='holes')
MASK1=FILL(mask=MASK,object='holes',direction=DIR0)
FINAL_RESULT=RESULT(var=MASK1)

Instruction: Remove the fragmented areas at the bottom.
Program:
DIR0=DIRECTION(command='fragmented areas')
MASK0=REMOVE(mask=MASK,object='fragmented areas',direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Expand to the right edge. Remove small noise artifacts.
Program:
DIR0=DIRECTION(command='expand')
MASK0=EXPAND(mask=MASK, direction=DIR0)
DIR1=DIRECTION(command='noise artifacts')
MASK1=REMOVE(mask=MASK0, object='noise artifacts', direction=DIR1)
FINAL_RESULT=RESULT(var=MASK1)


Instruction: Smooth out the rough edges at the top.
Program:
DIR0=DIRECTION(command='smooth')
MASK0=SMOOTH(mask=MASK, direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Fill small gaps in the center region.
Program:
DIR0=DIRECTION(command='small gaps')
MASK0=FILL(mask=MASK, object='small gaps', direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Remove the isolated objects at the top-left.
Program:
DIR0=DIRECTION(command='isolated objects')
MASK0=REMOVE(mask=MASK, object='isolated objects', direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Shrink at the left boundary. Smooth the edges.
Program:
DIR0=DIRECTION(command='shrink')
MASK0=SHRINK(mask=MASK,direction=DIR0)
DIR1=DIRECTION(command='smooth')
MASK1=SMOOTH(mask=MASK0,direction=DIR1)
FINAL_RESULT=RESULT(var=MASK1)

Instruction: Expand slightly on the top and bottom.
Program:
DIR0=DIRECTION(command='expand')
MASK0=EXPAND(mask=MASK,direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Smooth the inner boundaries of the shapes.
Program:
DIR0=DIRECTION(command='inner boundaries')
MASK0=SMOOTH(mask=MASK,direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Remove the tiny speckles scattered throughout the MASK.
Program:
DIR0=DIRECTION(command='tiny speckles')
MASK0=REMOVE(mask=MASK, object='tiny speckles', direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)

Instruction: Fill missing pixels in the bottom-left corner.
Program:
DIR0=DIRECTION(command='missing pixels')
MASK0=FILL(mask=MASK, object='missing pixels', direction=DIR0)
FINAL_RESULT=RESULT(var=MASK0)


Instruction: {instruction}
Program:
"""

def _chat(messages, temperature=0.7, top_p=0.5, max_retries=5, delay=3):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    for attempt in range(max_retries):
        model = "gpt-4-turbo" if attempt == max_retries - 1 else "gpt-3.5-turbo"
        try:
            return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature,
                                                max_tokens=512, top_p=top_p, n=1)
        except Exception as e:
            print(f"LLM attempt {attempt + 1} ({model}) failed: {type(e).__name__}: {e}")
            time.sleep(delay * (2 ** attempt))
    raise RuntimeError("LLM program generation failed")


def llm_program(instruction):
    response = _chat([{"role": "user", "content": PROMPT.format(instruction=instruction)}])
    return response.choices[0]["message"]["content"].strip("\n")


_TEMPLATE = {  # keyword -> (DIRECTION command, module)
    "expand": ("expand", "EXPAND"),
    "shrink": ("shrink", "SHRINK"),
    "fragments": ("fragments", "REMOVE"),
    "holes": ("holes", "FILL"),
    "smooth": ("smooth", "SMOOTH"),
}


def template_program(keywords):
    lines, mask = [], "MASK"
    for i, kw in enumerate(keywords):
        cmd, module = next(v for k, v in _TEMPLATE.items() if k in kw.lower())
        lines.append(f"DIR{i}=DIRECTION(command='{cmd}')")
        lines.append(f"MASK{i}={module}(mask={mask},direction=DIR{i})")
        mask = f"MASK{i}"
    lines.append(f"FINAL_RESULT=RESULT(var={mask})")
    return "\n".join(lines)


def make_program(keywords, cfg):
    """Returns (instruction, program)."""
    instruction = ", ".join(keywords)
    if getattr(cfg, "program_mode", "llm") == "template":
        return instruction, template_program(keywords)
    return instruction, llm_program(instruction)
