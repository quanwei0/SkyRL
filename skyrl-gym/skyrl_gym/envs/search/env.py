from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
from skyrl_gym.envs.search.utils import compute_score
from skyrl_gym.envs.search.reward import (
    compute_answer_reward,
    compute_format_reward,
    compute_retrieval_reward,
)
from skyrl_gym.tools import SearchToolGroup
import re
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass
class SearchEnvConfig:
    log_requests: bool = False
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    timeout: int = 30
    multi_reward: bool = False
    answer_reward_score: float = 1.0
    answer_reward_fail_score: float = 0.0
    format_reward_score: float = 0.2
    format_reward_fail_score: float = 0.0
    retrieval_reward_score: float = 0.5
    retrieval_reward_fail_score: float = 0.0


class SearchEnv(BaseTextEnv):
    """
    Environment for Search execution tasks.

    Based on Verl + Search-R1 integration
    """

    def __init__(self, env_config: Union[SearchEnvConfig, DictConfig], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Initialize the tools
        # name is hardcoded to "SearchToolGroup", with tool name "search"
        self.tool_group = SearchToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

        # Multi-reward (decomposed into answer + format + retrieval)
        self.multi_reward = getattr(env_config, "multi_reward", False)
        self.answer_reward_score = getattr(env_config, "answer_reward_score", 1.0)
        self.answer_reward_fail_score = getattr(env_config, "answer_reward_fail_score", 0.0)
        self.format_reward_score = getattr(env_config, "format_reward_score", 0.2)
        self.format_reward_fail_score = getattr(env_config, "format_reward_fail_score", 0.0)
        self.retrieval_reward_score = getattr(env_config, "retrieval_reward_score", 0.5)
        self.retrieval_reward_fail_score = getattr(env_config, "retrieval_reward_fail_score", 0.0)
        self._answer_r: float = 0.0
        self._format_r: float = 0.0
        self._retrieval_r: float = 0.0

    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _get_reward(self, action: str, done: bool) -> float:
        if not done:
            return 0

        chat_history_str = "".join([item["content"] for item in self.chat_history])

        if not self.multi_reward:
            return compute_score(chat_history_str, self.ground_truth)

        self._answer_r = compute_answer_reward(
            chat_history_str, self.ground_truth,
            score=self.answer_reward_score,
            fail_score=self.answer_reward_fail_score,
        )
        self._format_r = compute_format_reward(
            chat_history_str,
            score=self.format_reward_score,
            fail_score=self.format_reward_fail_score,
        )
        self._retrieval_r = compute_retrieval_reward(
            chat_history_str, self.ground_truth,
            score=self.retrieval_reward_score,
            fail_score=self.retrieval_reward_fail_score,
        )
        return self._answer_r + self._format_r + self._retrieval_r

    def get_metrics(self) -> Dict[str, Any]:
        if not self.multi_reward:
            return {}
        return {
            "answer_reward": self._answer_r,
            "format_reward": self._format_r,
            "retrieval_reward": self._retrieval_r,
        }

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)

        return "\n<information>" + tool_output + "</information>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        try:
            query = self._parse_action(action)
            observation = self._execute_tool("SearchToolGroup", "search", query)
        except Exception as e:
            error = str(e)
            observation = None

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": query,
        }

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )
