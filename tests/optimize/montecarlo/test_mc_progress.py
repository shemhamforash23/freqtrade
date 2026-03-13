from freqtrade.enums.mcptstate import MCPTState
from freqtrade.optimize.montecarlo.mc_progress import MCProgress


class TestMCProgress:
    """Test MCProgress functionality."""

    def test_init_default(self):
        """Test default initialization."""
        progress = MCProgress()

        assert progress.state == MCPTState.STARTUP
        assert progress._progress == 0
        assert progress._max_steps == 0
        assert progress.progress == 0
        assert progress.action == "startup"

    def test_init_step_basic(self):
        """Test basic step initialization."""
        progress = MCProgress()
        progress.init_step(MCPTState.DATALOAD, max_steps=10)

        assert progress.state == MCPTState.DATALOAD
        assert progress._progress == 0
        assert progress._max_steps == 10
        assert progress.progress == 0
        assert progress.action == "dataload"

    def test_set_new_value(self):
        """Test setting specific progress values."""
        progress = MCProgress()
        progress.init_step(MCPTState.PERMUTATION_TESTS, max_steps=100)

        # Set to specific value
        progress.set_new_value(25)
        assert progress._progress == 25
        assert progress.progress == 0.25

        # Set to max value
        progress.set_new_value(100)
        assert progress.progress == 1.0

        # Set beyond max (should be capped at 1.0)
        progress.set_new_value(150)
        assert progress.progress == 1.0

    def test_increment(self):
        """Test incrementing progress."""
        progress = MCProgress()
        progress.init_step(MCPTState.ORIGINAL_BACKTEST, max_steps=5)

        # Initial state
        assert progress.progress == 0

        # Increment step by step
        progress.increment()
        assert progress._progress == 1
        assert progress.progress == 0.2

        progress.increment()
        assert progress._progress == 2
        assert progress.progress == 0.4

        # Complete the step
        progress.increment()
        progress.increment()
        progress.increment()
        assert progress._progress == 5
        assert progress.progress == 1.0

    def test_progress_calculation_edge_cases(self):
        """Test progress calculation edge cases."""
        progress = MCProgress()

        # Zero max_steps should return 0
        progress.init_step(MCPTState.STARTUP, max_steps=0)
        assert progress.progress == 0

        # Negative max_steps should return 0
        progress.init_step(MCPTState.STARTUP, max_steps=-10)
        assert progress.progress == 0

        # Negative progress should be capped at 0
        progress.init_step(MCPTState.STARTUP, max_steps=10)
        progress.set_new_value(-5)
        assert progress.progress == 0

    def test_progress_rounding(self):
        """Test progress rounding to 5 decimal places."""
        progress = MCProgress()
        progress.init_step(MCPTState.VALIDATION, max_steps=3)

        # This should create a repeating decimal
        progress.set_new_value(1)
        assert progress.progress == 0.33333

        progress.set_new_value(2)
        assert progress.progress == 0.66667

    def test_all_states(self):
        """Test all MCPTState values."""
        progress = MCProgress()

        for state in MCPTState:
            progress.init_step(state, max_steps=1)
            assert progress.state == state
            assert progress.action == str(state)

            # Complete the step
            progress.increment()
            assert progress.progress == 1.0

    def test_state_transitions(self):
        """Test transitioning between different states."""
        progress = MCProgress()

        # Start with data loading
        progress.init_step(MCPTState.DATALOAD, max_steps=5)
        progress.increment()
        progress.increment()
        assert progress.state == MCPTState.DATALOAD
        assert progress.progress == 0.4

        # Switch to original backtest
        progress.init_step(MCPTState.ORIGINAL_BACKTEST, max_steps=10)
        assert progress.state == MCPTState.ORIGINAL_BACKTEST
        assert progress._progress == 0  # Reset to 0
        assert progress.progress == 0

        # Work through new step
        progress.increment()
        assert progress.progress == 0.1

    def test_action_property(self):
        """Test action property returns string representation."""
        progress = MCProgress()

        for state in MCPTState:
            progress.init_step(state, max_steps=1)
            assert progress.action == str(state)
            assert isinstance(progress.action, str)

    def test_large_numbers(self):
        """Test with large step counts."""
        progress = MCProgress()
        progress.init_step(MCPTState.PERMUTATION_TESTS, max_steps=10000)

        # Set to 50% progress
        progress.set_new_value(5000)
        assert progress.progress == 0.5

        # Set to near completion
        progress.set_new_value(9999)
        assert progress.progress == 0.9999

    def test_fractional_steps(self):
        """Test with fractional step counts."""
        progress = MCProgress()
        progress.init_step(MCPTState.REPORTING, max_steps=2.5)

        progress.set_new_value(1.25)
        assert progress.progress == 0.5

        progress.set_new_value(2.5)
        assert progress.progress == 1.0

    def test_progress_capping(self):
        """Test that progress is properly capped between 0 and 1."""
        progress = MCProgress()
        progress.init_step(MCPTState.VALIDATION, max_steps=10)

        # Test upper cap
        progress.set_new_value(15)
        assert progress.progress == 1.0

        # Test lower cap
        progress.set_new_value(-5)
        assert progress.progress == 0.0

    def test_multiple_increments_beyond_max(self):
        """Test incrementing beyond maximum steps."""
        progress = MCProgress()
        progress.init_step(MCPTState.DATALOAD, max_steps=3)

        # Increment beyond max
        for _ in range(10):
            progress.increment()

        assert progress._progress == 10
        assert progress.progress == 1.0  # Should be capped

    def test_instance_independence(self):
        """Test that each MCProgress instance has independent state."""
        progress1 = MCProgress()
        progress2 = MCProgress()

        # Modify progress1
        progress1.init_step(MCPTState.PERMUTATION_TESTS, max_steps=100)
        progress1.set_new_value(50)

        # progress2 should still be at initial state
        assert progress2.state == MCPTState.STARTUP
        assert progress2._progress == 0
        assert progress2._max_steps == 0

        # Modify progress2
        progress2.init_step(MCPTState.VALIDATION, max_steps=10)
        progress2.increment()

        # progress1 should not be affected
        assert progress1.state == MCPTState.PERMUTATION_TESTS
        assert progress1._progress == 50
        assert progress1._max_steps == 100
