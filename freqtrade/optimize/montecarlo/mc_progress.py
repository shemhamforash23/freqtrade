from freqtrade.enums.mcptstate import MCPTState


class MCProgress:
    """
    Progress tracker for MCPT runs.

    Follows BTProgress pattern for consistency.
    Tracks progress through MCPT states and provides progress ratio.
    """

    def __init__(self) -> None:
        """Initialize MCProgress with instance-level attributes."""
        self._action: MCPTState = MCPTState.STARTUP
        self._progress: float = 0
        self._max_steps: float = 0

    def init_step(self, action: MCPTState, max_steps: float):
        """
        Initialize a new progress step.

        :param action: Current MCPT state
        :param max_steps: Maximum number of steps for this action
        """
        self._action = action
        self._max_steps = max_steps
        self._progress = 0

    def set_new_value(self, new_value: float):
        """
        Set progress to a specific value.

        :param new_value: New progress value
        """
        self._progress = new_value

    def increment(self):
        """Increment progress by 1."""
        self._progress += 1

    @property
    def progress(self) -> float:
        """
        Get progress as ratio, capped to be between 0 and 1.

        :return: Progress ratio between 0 and 1
        """
        return max(
            min(round(self._progress / self._max_steps, 5) if self._max_steps > 0 else 0, 1), 0
        )

    @property
    def action(self) -> str:
        """
        Get current action as string.

        :return: Current MCPT state as string
        """
        return str(self._action)

    @property
    def state(self) -> MCPTState:
        """
        Get current MCPT state.

        :return: Current MCPTState enum value
        """
        return self._action
