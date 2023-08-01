"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

#一个用于生成提示（prompt）的工具类，根据指定的模板生成完整的提示，并提供了获取回答的方法。
class Prompter(object):
    #是Python中的一个特殊属性，用于限制类的实例动态添加属性的能力。
    #通过定义__slots__属性，可以明确指定类实例可以拥有的属性，从而提高内存使用效率并限制属性的数量。
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
        # return output
    
if __name__ == "__main__":
    prompter =  Prompter(verbose=False)
    print(prompter.generate_prompt("123123","456456","789"))