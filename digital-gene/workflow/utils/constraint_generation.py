import torch
import re
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2Tokenizer, DynamicCache, LogitsProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field

class RespState(Enum):
    WAIT_CODE = "wait_code"                         # wait '<code>{"' (27, 1851, 88863)
    ADD_CATEGORY_KEY = "add_category_key"           # add 'category": "'
    GEN_CATEGORY_VALUE = "gen_category_value"       # generate category!
    ADD_POSITION_KEY = "add_position_key"           # add '", "pose": {"global_position":'
    GEN_POSITION_VALUE = "gen_position_value"       # generate number or ',' or '-', or ' ' or '],' (means end) ' [' or ' [-' (means start)
    ADD_ROTATION_KEY = "add_rotation_key"           # add ' "global_rotation":'
    GEN_ROTATION_VALUE = "gen_rotation_value"       # generate number or ',' or '-', or ' ' or ']},' (means end) ' [' or ' [-' (means start)
    ADD_CONCEPT_KEY = "add_concept_key"             # add ' "conceptualization": [{"template": "'
    GEN_TEMPLATE_NAME = "gen_template_name"         # generate template!
    ADD_PARAM_CON = "add_param_con"                 # add '", "parameters": {"'
    GEN_PARAM_KEY = "gen_param_key"                 # generate param key, end with '":'
    ADD_PARAM_KV_CON = "add_param_kv_con"           # add ' ['
    GEN_PARAM_VALUE = "gen_param_value"             # generate param value, range [100000, 101024) or ',' add ' '
    ADD_PARAM_VALUE_CON = "add_param_value_con"     # add '], "'
    GEN_TEMPLATE_OR_END = "gen_template_or_end"     # gen ']}' (token 13989, next template) or ']' (token 60, end)
    ADD_NEXT_TEMPLATE_CON = "add_next_template_con" # add '}, {"template": "'
    ADD_END = "add_end"                             # add '}}]}</code>'
    END = "end"                                     # nothing, just end

@dataclass
class GenerationContext:
    current_state: RespState = RespState.WAIT_CODE
    current_category: Optional[List] = field(default_factory=list)
    current_position: Optional[List] = field(default_factory=list)
    current_rotation: Optional[List] = field(default_factory=list)
    current_pose_value: Optional[List] = field(default_factory=list)
    current_commas: Optional[int] = 0
    current_template: Optional[List] = field(default_factory=list)
    current_param: Optional[List] = field(default_factory=list)
    current_param_list: Optional[List[List]] = field(default_factory=list)
    current_param_value: Optional[List] = field(default_factory=list)
    current_param_index: Optional[int] = 0

class StateTransition:
    def __init__(self):
        self.transitions: Dict = self._build_transition_table()
    
    def _build_transition_table(self) -> Dict[RespState, Dict[str, RespState]]:
        return {
            RespState.WAIT_CODE: {
                "remain": RespState.WAIT_CODE,
                "continue": RespState.ADD_CATEGORY_KEY
            },
            RespState.ADD_CATEGORY_KEY: {
                "continue": RespState.GEN_CATEGORY_VALUE
            },
            RespState.GEN_CATEGORY_VALUE: {
                "remain": RespState.GEN_CATEGORY_VALUE,
                "continue": RespState.ADD_POSITION_KEY
            },
            RespState.ADD_POSITION_KEY: {
                "continue": RespState.GEN_POSITION_VALUE
            },
            RespState.GEN_POSITION_VALUE: {
                "remain": RespState.GEN_POSITION_VALUE,
                "continue": RespState.ADD_ROTATION_KEY
            },
            RespState.ADD_ROTATION_KEY: {
                "continue": RespState.GEN_ROTATION_VALUE
            },
            RespState.GEN_ROTATION_VALUE: {
                "remain": RespState.GEN_ROTATION_VALUE,
                "continue": RespState.ADD_CONCEPT_KEY
            },
            RespState.ADD_CONCEPT_KEY: {
                "continue": RespState.GEN_TEMPLATE_NAME
            },
            RespState.GEN_TEMPLATE_NAME: {
                "remain": RespState.GEN_TEMPLATE_NAME,
                "continue": RespState.ADD_PARAM_CON
            },
            RespState.ADD_PARAM_CON: {
                "continue": RespState.GEN_PARAM_KEY
            },
            RespState.GEN_PARAM_KEY: {
                "remain": RespState.GEN_PARAM_KEY,
                "continue": RespState.ADD_PARAM_KV_CON
            },
            RespState.ADD_PARAM_KV_CON: {
                "continue": RespState.GEN_PARAM_VALUE
            },
            RespState.GEN_PARAM_VALUE: {
                "remain": RespState.GEN_PARAM_VALUE,
                "continue": RespState.ADD_PARAM_VALUE_CON,
                "wait": RespState.GEN_TEMPLATE_OR_END
            },
            RespState.ADD_PARAM_VALUE_CON: {
                "continue": RespState.GEN_PARAM_KEY
            },
            RespState.GEN_TEMPLATE_OR_END: {
                "continue": RespState.ADD_NEXT_TEMPLATE_CON,
                "end": RespState.ADD_END
            },
            RespState.ADD_NEXT_TEMPLATE_CON: {
                "continue": RespState.GEN_TEMPLATE_NAME
            },
            RespState.ADD_END: {
                "continue": RespState.END
            }
        }
    
    def get_next_state(self, current_state: RespState, action: str) -> RespState:
        state_transitions = self.transitions.get(current_state, {})
        # state transitions
        return state_transitions.get(action, current_state)

class FSM_LogitsProcessor(LogitsProcessor):
    def __init__(self, generator: "ConstrainedGenerator", prompt_len: int):
        self.generator = generator
        self.prompt_len = prompt_len
        # context
        self.context = GenerationContext()  # new context
        # fixed sequence
        self.fixed_seq_len = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_ids = input_ids[:, self.prompt_len:]
        self.context = self.generator.update_context(generated_ids, self.context)
        current_state = self.context.current_state
        if current_state == RespState.END:
            eos_token_id = self.generator.tokenizer.eos_token_id
            mask = torch.full_like(scores, -float("inf"))
            mask[:, eos_token_id] = 0
            return mask
        # get state value string
        state_value = current_state.value
        state_action = state_value.split('_')[0]
        allowed_tokens = []
        
        if state_action == "wait":
            return scores
        elif state_action == "add":
            add_token_list = self.generator.add_tokens[state_value]
            if self.fixed_seq_len < len(add_token_list):
                valid_tokens = [add_token_list[self.fixed_seq_len]]
                self.fixed_seq_len = (self.fixed_seq_len + 1) % len(add_token_list) # update or clear
                processed_scores = torch.full_like(scores, -float("inf"))
                processed_scores[:, valid_tokens] = scores[:, valid_tokens]
                return processed_scores
            else:
                raise ValueError(f"Error fixed seq len, add token list: {add_token_list}, fixed seq len: {self.fixed_seq_len}")
        else:
            assert state_action == "gen"
            # get allowed tokens
            valid_tokens = self.generator.get_allowed_tokens(current_state, self.context)
            if not valid_tokens:
                raise ValueError(f"No allowed tokens for state: {self.context.current_state}")
            processed_scores = torch.full_like(scores, -float("inf"))
            processed_scores[:, valid_tokens] = scores[:, valid_tokens]
            return processed_scores

class ConstrainedGenerator:
    def __init__(
            self,
            model_path: str,
            categories: List[str],
            param_dims: Dict,
            float_token_start: int = 100000,
            float_token_num: int = 1024,
            float_value_start: int = 2048,
            device="cuda"
        ):
        self.model_path: str = model_path
        self.device: str = device
        self.model: Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).to(device)
        self.processor: Qwen2_5_VLProcessor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.tokenizer: Qwen2Tokenizer = self.processor.tokenizer
        self.categories: List[str] = categories
        self.param_dims: Dict = param_dims
        self.float_token_start: int = float_token_start
        self.float_token_num: int = float_token_num
        self.value_range = (float_value_start, float_value_start + float_token_num)
        # state transition
        self.state_transition = StateTransition()
        # pre compute
        self._precompute_all_tokens()
    
    def _precompute_all_tokens(self):
        # 1. add tokens
        self.add_tokens = {}
        add_segments: Dict[str, str] = {
            "add_category_key": 'category": "',
            "add_position_key": '", "pose": {"global_position":',
            "add_rotation_key": ' "global_rotation":',
            "add_concept_key": ' "conceptualization": [{"template": "',
            "add_param_con": '", "parameters": {"',
            "add_param_kv_con": ' [',
            "add_param_value_con": '], "',
            "add_next_template_con": '}, {"template": "',
            "add_end": '}}]}</code>'
        }
        for add_tag, segment in add_segments.items():
            encoded = self.tokenizer.encode(segment, add_special_tokens=False)
            self.add_tokens[add_tag] = encoded
        # 2. categories id
        self.category_valid_tokens: dict = {}
        for category in self.categories:
            encoded = self.tokenizer.encode(category, add_special_tokens=False)
            for index, token in enumerate(encoded):
                seq: tuple = tuple(encoded[:index])
                if not seq in self.category_valid_tokens:
                    self.category_valid_tokens[seq] = set()
                self.category_valid_tokens[seq].add(token)
        # 4. param value
        self.value_tokens = list(range(self.float_token_start, self.float_token_start + self.float_token_num))
        # 5. wait tokens
        self.wait_tokens: dict = {}
        wait_segments: dict = {
            "wait_code": '<code>{"'
        }
        for wait_tag, segment in wait_segments.items():
            encoded = self.tokenizer.encode(segment, add_special_tokens=False)
            self.wait_tokens[wait_tag] = encoded
        # 6. gen template or end
        self.other_tokens: dict = {}
        others_segments: dict = {
            "pose_start": [' [-', ' ['],
            "position_end": ['],'],
            "rotation_end": [']},'],
            "zero": ['0'],
            "natural": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            "positive": ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            "comma": [','],
            "negative": [' -'],
            "blank": [' '],
            "param_name_end": ['":'],
            "next_template": [']}'], # ']}' means next template
            "end": [']']    # ']' means end
        }
        for other_tag, segment in others_segments.items():
            self.other_tokens[other_tag] = []
            for seg in segment:
                encoded = self.tokenizer.encode(seg, add_special_tokens=False)
                self.other_tokens[other_tag].extend(encoded)
        # 3. template info
        # template name (given category token)
        # param name (given category and template token)
        # param dims
        self.template_valid_tokens: dict = {}
        self.param_valid_tokens: dict = {}
        self.param_valid_dims: dict = {}
        for category, params in self.param_dims.items():
            # init template and param
            category_id: tuple = tuple(self.tokenizer.encode(category, add_special_tokens=False))
            self.template_valid_tokens[category_id] = {}
            self.param_valid_tokens[category_id] = {}
            self.param_valid_dims[category_id] = {}
            # template
            template_names: List[str] = params.keys()
            for template_name in template_names:
                template_name_id: tuple = tuple(self.tokenizer.encode(template_name, add_special_tokens=False))
                for index, token in enumerate(template_name_id):
                    seq: tuple = tuple(template_name_id[:index])
                    if not seq in self.template_valid_tokens[category_id]:
                        self.template_valid_tokens[category_id][seq] = set()
                    self.template_valid_tokens[category_id][seq].add(token)
                # init param
                self.param_valid_tokens[category_id][template_name_id] = {}
                self.param_valid_dims[category_id][template_name_id] = {}
                for param_name, param_dims in params[template_name].items():
                    param_name_id: tuple = tuple(self.tokenizer.encode(param_name, add_special_tokens=False))
                    for index, token in enumerate(param_name_id):
                        seq: tuple = tuple(param_name_id[:index])
                        if not seq in self.param_valid_tokens[category_id][template_name_id]:
                            self.param_valid_tokens[category_id][template_name_id][seq] = set()
                        self.param_valid_tokens[category_id][template_name_id][seq].add(token)
                    # add end tag
                    if not param_name_id in self.param_valid_tokens[category_id][template_name_id]:
                        self.param_valid_tokens[category_id][template_name_id][param_name_id] = set()
                    self.param_valid_tokens[category_id][template_name_id][param_name_id].add(self.other_tokens["param_name_end"][0])
                    self.param_valid_dims[category_id][template_name_id][param_name_id] = param_dims[-1]    # the last, TODO: all array
    
    def get_allowed_tokens(self, state: RespState, context: GenerationContext) -> List[int]:
        """generate token"""
        if state == RespState.GEN_CATEGORY_VALUE:   # catetory
            # get current context
            current_category: tuple = tuple(context.current_category)
            return list(self.category_valid_tokens.get(tuple(current_category), []))
        elif state == RespState.GEN_POSITION_VALUE or state == RespState.GEN_ROTATION_VALUE:
            # get current context (token list)
            current_pose: list = context.current_position if state == RespState.GEN_POSITION_VALUE else context.current_rotation
            if len(current_pose) == 0:
                # start
                return self.other_tokens["pose_start"]
            elif current_pose[-1] in self.other_tokens["pose_start"]:
                # first number
                return self.other_tokens["natural"]
            elif current_pose[-1] in self.other_tokens["blank"]:
                # next number
                return self.other_tokens["natural"]
            elif current_pose[-1] in self.other_tokens["comma"]:
                return self.other_tokens["blank"] + self.other_tokens["negative"]
            elif current_pose[-1] in self.other_tokens["negative"]:
                return self.other_tokens["positive"]
            else:
                allowed_tokens: list = []
                # current pose value
                # special case: the first pose value is zero
                not_last: bool = len(context.current_pose_value) < 3 and not (len(context.current_pose_value) == 1 and context.current_pose_value[0] in self.other_tokens["zero"])
                if not_last:
                    allowed_tokens += self.other_tokens["natural"]
                # if the last:
                if context.current_commas == 2:
                    allowed_tokens += self.other_tokens["position_end"] if state == RespState.GEN_POSITION_VALUE else self.other_tokens["rotation_end"]
                else:
                    allowed_tokens += self.other_tokens["comma"]
                return allowed_tokens
        elif state == RespState.GEN_TEMPLATE_NAME:
            # get current context
            current_category: tuple = tuple(context.current_category)
            assert current_category in self.template_valid_tokens
            current_template: tuple = tuple(context.current_template)
            return list(self.template_valid_tokens[current_category].get(current_template, []))
        elif state == RespState.GEN_PARAM_KEY:
            # get current conetxt
            current_category: tuple = tuple(context.current_category)
            assert current_category in self.param_valid_tokens
            current_template: tuple = tuple(context.current_template)
            assert current_template in self.param_valid_tokens[current_category]
            current_param: tuple = tuple(context.current_param)
            return list(self.param_valid_tokens[current_category][current_template].get(current_param, []))
            # TODO: remove the key already generatd
        elif state == RespState.GEN_PARAM_VALUE:
            # get current param
            current_param_value: list = context.current_param_value
            if len(current_param_value) == 0 or current_param_value[-1] in self.other_tokens["blank"]:
                return self.value_tokens
            elif self.float_token_start <= current_param_value[-1] < self.float_token_start + self.float_token_num:
                return self.other_tokens["comma"]
            elif current_param_value[-1] in self.other_tokens["comma"]:
                return self.other_tokens["blank"]
            else:
                raise ValueError(f"Error param value: {current_param_value}")
            # TODO: param dims
        elif state == RespState.GEN_TEMPLATE_OR_END:
            return self.other_tokens["next_template"] + self.other_tokens["end"]
        else:
            raise ValueError(f"Error state: {state}")
    
    def update_context(self, token_ids: torch.Tensor, context: GenerationContext):
        current_state: RespState = context.current_state
        current_state_value: str = current_state.value
        current_state_action = current_state_value.split('_')[0]    # split action
        state_transition_action: str = "continue"   # default transition
        if current_state == RespState.WAIT_CODE:
            length: int = len(self.wait_tokens["wait_code"])
            if len(token_ids[0]) < length:
                state_transition_action = "remain"
            elif token_ids[0, -length:].tolist() != self.wait_tokens["wait_code"]:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_CATEGORY_VALUE:
            # generate category, update current_category, if generation is done, contine
            # append
            context.current_category.append(token_ids[0, -1].item())  # (update current category)
            # check whether the generation process is done
            current_category: tuple = tuple(context.current_category)
            if len(self.category_valid_tokens.get(current_category, [])) != 0:
                state_transition_action = "remain"
        elif current_state == RespState.ADD_POSITION_KEY or current_state == RespState.ADD_ROTATION_KEY:
            # if add ended
            length: int = len(self.add_tokens[current_state_value])
            if len(token_ids[0]) < length:
                state_transition_action = "remain"
            elif token_ids[0, -length:].tolist() != self.add_tokens[current_state_value]:
                state_transition_action = "remain"
            else:
                # update current_commas
                context.current_commas = 0
                context.current_pose_value = []
        elif current_state == RespState.GEN_POSITION_VALUE or current_state == RespState.GEN_ROTATION_VALUE:
            # generate position, update current position, if generation is done, continue
            # append
            if current_state == RespState.GEN_POSITION_VALUE:
                context.current_position.append(token_ids[0, -1].item())  # update current position
            else:
                context.current_rotation.append(token_ids[0, -1].item())  # update current rotation
            # update commas if needed
            if token_ids[0, -1].item() in self.other_tokens["comma"]:
                context.current_commas += 1
                context.current_pose_value = []
            elif token_ids[0, -1].item() in self.other_tokens["natural"]:
                context.current_pose_value.append(token_ids[0, -1].item())
            # check whether the generation process is done
            number_tokens: list = self.other_tokens["position_end"] if current_state == RespState.GEN_POSITION_VALUE else self.other_tokens["rotation_end"]
            if not token_ids[0, -1].item() in number_tokens:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_TEMPLATE_NAME:
            # generate
            context.current_template.append(token_ids[0, -1].item())      # update current template name
            # check whether the generation process is done
            current_category: tuple = tuple(context.current_category)
            current_template: tuple = tuple(context.current_template)
            assert current_category in self.template_valid_tokens
            if len(self.template_valid_tokens[current_category].get(current_template, [])) != 0:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_PARAM_KEY:
            # if len(self.param_valid_tokens[current_category][current_template].get(current_param, [])) != 0:
            if token_ids[0, -1].item() != self.other_tokens["param_name_end"][0]:
                state_transition_action = "remain"
                # generate
                context.current_param.append(token_ids[0, -1].item())
                current_category: tuple = tuple(context.current_category)
                current_template: tuple = tuple(context.current_template)
                assert current_category in self.param_valid_tokens
                assert current_template in self.param_valid_tokens[current_category]
            else:
                current_category: tuple = tuple(context.current_category)
                current_template: tuple = tuple(context.current_template)
                assert current_category in self.param_valid_tokens
                assert current_template in self.param_valid_tokens[current_category]
                current_param: tuple = tuple(context.current_param)
                # update param list
                context.current_param_list.append(current_param)
        elif current_state == RespState.GEN_PARAM_VALUE:
            # generate
            context.current_param_value.append(token_ids[0, -1].item())
            if self.float_token_start <= context.current_param_value[-1] < self.float_token_start + self.float_token_num:
                context.current_param_index += 1
            current_category: tuple = tuple(context.current_category)
            current_template: tuple = tuple(context.current_template)
            current_param: tuple = tuple(context.current_param)
            if context.current_param_index >= self.param_valid_dims[current_category][current_template][current_param]:
                # reset param value
                context.current_param_value.clear()
                context.current_param_index = 0
                # next param or wait
                if len(context.current_param_list) == len(self.param_valid_dims[current_category][current_template].keys()):
                    # already generate all param name
                    # TODO: handle duplicated param name
                    state_transition_action = "wait"
                # next param: 'continue' (default)
            else:
                state_transition_action = "remain"
        elif current_state == RespState.GEN_TEMPLATE_OR_END:
            # update state
            if token_ids[0, -1].item() in self.other_tokens["end"]:
                state_transition_action = "end"
        elif current_state == RespState.ADD_PARAM_CON or current_state == RespState.ADD_PARAM_VALUE_CON:
            # if add ended
            length: int = len(self.add_tokens[current_state_value])
            if len(token_ids[0]) < length:
                state_transition_action = "remain"
            elif token_ids[0, -length:].tolist() != self.add_tokens[current_state_value]:
                state_transition_action = "remain"
            else:
                # clear current_param (value and name)
                context.current_param.clear()
                context.current_param_value.clear()
        elif current_state == RespState.ADD_PARAM_KV_CON:
            # if add ended
            length: int = len(self.add_tokens[current_state_value])
            if len(token_ids[0]) < length:
                state_transition_action = "remain"
            elif token_ids[0, -length:].tolist() != self.add_tokens[current_state_value]:
                state_transition_action = "remain"
            else:
                # clear current_param_value
                context.current_param_value.clear()
        elif current_state == RespState.ADD_NEXT_TEMPLATE_CON:
            # if add ended
            length: int = len(self.add_tokens[current_state_value])
            if len(token_ids[0]) < length:
                state_transition_action = "remain"
            elif token_ids[0, -length:].tolist() != self.add_tokens[current_state_value]:
                state_transition_action = "remain"
            else:
                # clear current template
                context.current_template.clear()
                context.current_param.clear()
                context.current_param_list.clear()
                context.current_param_value.clear()
                context.current_param_index = 0
        else:
            assert current_state_action == "add", f"Error state action: {current_state_action}"
            # if add ended
            length: int = len(self.add_tokens[current_state_value])
            if len(token_ids[0]) < length:
                state_transition_action = "remain"
            elif token_ids[0, -length:].tolist() != self.add_tokens[current_state_value]:
                state_transition_action = "remain"
        # update state
        context.current_state = self.state_transition.get_next_state(current_state, state_transition_action)
        return context
 
    def decode_generated_result(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
        """decode"""
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""
    
    @staticmethod
    def _prompt_template(question: str, image_path: str, system_prompt: str = "") -> List[Dict]:
        if system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                        },
                        {"type": "text", "text": question}
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
        return messages
    
    def generate(self, question: str, image_path: str, system_prompt: str = "") -> str:
        messages = ConstrainedGenerator._prompt_template(question, image_path, system_prompt)
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        # create fsm logit processor
        fsm_processor = FSM_LogitsProcessor(self, prompt_len)
        # generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False, logits_processor=[fsm_processor], eos_token_id=self.tokenizer.eos_token_id)
        return self.decode_generated_result(inputs["input_ids"], generated_ids)
