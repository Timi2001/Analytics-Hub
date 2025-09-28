"""
Reinforcement Learning Agent for Autonomous Analytics Platform
Learns optimal strategies for dashboard design, analysis methods, and user experience.
"""
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.agents.user_tracker import (
    user_tracker, InteractionType, UserSentiment,
    UserInteraction, SessionMetrics
)

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of autonomous agents."""
    DASHBOARD_DESIGNER = "dashboard_designer"
    ANALYSIS_STRATEGIST = "analysis_strategist"
    REPORT_GENERATOR = "report_generator"
    META_LEARNER = "meta_learner"

class ActionType(Enum):
    """Types of actions the RL agent can take."""
    CHANGE_LAYOUT = "change_layout"
    MODIFY_CHART_TYPE = "modify_chart_type"
    ADJUST_COLOR_SCHEME = "adjust_color_scheme"
    REORDER_COMPONENTS = "reorder_components"
    ADD_REMOVE_WIDGETS = "add_remove_widgets"
    CHANGE_ANALYSIS_METHOD = "change_analysis_method"
    MODIFY_FEATURES = "modify_features"
    UPDATE_REPORT_STRUCTURE = "update_report_structure"

@dataclass
class State:
    """Current state of the environment for RL."""
    user_engagement: float
    system_performance: float
    data_characteristics: Dict[str, Any]
    recent_interactions: List[str]
    current_dashboard_config: Dict[str, Any]
    analysis_results: Dict[str, Any]
    timestamp: datetime

    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for RL algorithms."""
        # This is a simplified representation
        # In practice, this would be much more sophisticated
        vector = [
            self.user_engagement,
            self.system_performance,
            len(self.data_characteristics),
            len(self.recent_interactions),
            len(self.current_dashboard_config),
            len(self.analysis_results)
        ]
        return np.array(vector)

@dataclass
class Action:
    """An action taken by the RL agent."""
    action_type: ActionType
    parameters: Dict[str, Any]
    timestamp: datetime
    agent_type: AgentType

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'action_type': self.action_type.value,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'agent_type': self.agent_type.value
        }

@dataclass
class Reward:
    """Reward signal for RL learning."""
    value: float
    components: Dict[str, float]
    timestamp: datetime
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'value': self.value,
            'components': self.components,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason
        }

class SimpleRLAgent:
    """Basic Reinforcement Learning Agent for Autonomous Analytics."""

    def __init__(self, agent_type: AgentType, learning_rate: float = 0.1):
        self.agent_type = agent_type
        self.learning_rate = learning_rate
        self.exploration_rate = 0.2  # Epsilon for epsilon-greedy
        self.action_history: List[Action] = []
        self.reward_history: List[Reward] = []
        self.state_history: List[State] = []

        # Simple Q-learning table (in practice, would use neural networks)
        self.q_table: Dict[str, Dict[str, float]] = {}

        # Performance tracking
        self.performance_metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'average_reward': 0.0,
            'learning_progress': 0.0
        }

        logger.info(f"🤖 Initialized {agent_type.value} RL agent")

    def get_state(self) -> State:
        """Get current environment state."""
        # Get recent user interactions
        recent_interactions = [
            interaction.interaction_type.value
            for interaction in user_tracker.interaction_history[-10:]  # Last 10 interactions
        ]

        # Get dashboard engagement metrics
        engagement_metrics = user_tracker.get_dashboard_engagement_metrics()

        # Get system learning data
        learning_data = user_tracker.get_system_learning_data()

        # Create current state
        state = State(
            user_engagement=engagement_metrics.get('average_session_duration', 0.0) / 300.0,  # Normalize to 0-1
            system_performance=learning_data.get('success_rate', 0.0),
            data_characteristics={'sample_size': 100},  # Would be more sophisticated
            recent_interactions=recent_interactions,
            current_dashboard_config={'layout': 'default', 'charts': 3},
            analysis_results={'accuracy': 0.85},
            timestamp=datetime.now()
        )

        self.state_history.append(state)
        return state

    def choose_action(self, state: State) -> Action:
        """Choose an action using epsilon-greedy strategy."""

        # Exploration vs Exploitation
        if random.random() < self.exploration_rate:
            # Explore: Choose random action
            action_type = random.choice(list(ActionType))
            parameters = self._get_random_parameters(action_type)
        else:
            # Exploit: Choose best known action
            action_type, parameters = self._get_best_action(state)

        action = Action(
            action_type=action_type,
            parameters=parameters,
            timestamp=datetime.now(),
            agent_type=self.agent_type
        )

        self.action_history.append(action)
        self.performance_metrics['total_actions'] += 1

        logger.debug(f"🎯 {self.agent_type.value} chose action: {action_type.value}")
        return action

    def _get_random_parameters(self, action_type: ActionType) -> Dict[str, Any]:
        """Generate random parameters for exploration."""
        parameters = {}

        if action_type == ActionType.CHANGE_LAYOUT:
            parameters = {
                'layout_type': random.choice(['grid', 'sidebar', 'tabs', 'masonry']),
                'columns': random.randint(1, 4)
            }
        elif action_type == ActionType.MODIFY_CHART_TYPE:
            parameters = {
                'chart_type': random.choice(['bar', 'line', 'scatter', 'pie', 'heatmap']),
                'interactive': random.choice([True, False])
            }
        elif action_type == ActionType.ADJUST_COLOR_SCHEME:
            parameters = {
                'primary_color': f"#{random.randint(0, 0xFFFFFF):06x"}",
                'background_style': random.choice(['light', 'dark', 'gradient'])
            }
        elif action_type == ActionType.CHANGE_ANALYSIS_METHOD:
            parameters = {
                'algorithm': random.choice(['linear', 'forest', 'neural', 'ensemble']),
                'complexity': random.choice(['simple', 'moderate', 'complex'])
            }

        return parameters

    def _get_best_action(self, state: State) -> Tuple[ActionType, Dict[str, Any]]:
        """Get the best action based on learned Q-values."""
        # Simplified: In practice, this would use the Q-table or neural network
        best_action = random.choice(list(ActionType))
        best_params = self._get_random_parameters(best_action)
        return best_action, best_params

    def calculate_reward(self, state: State, action: Action, next_state: State) -> Reward:
        """Calculate reward for the action taken."""

        # Base reward components
        reward_components = {
            'user_engagement': next_state.user_engagement * 0.3,
            'system_performance': next_state.system_performance * 0.25,
            'interaction_success': 1.0 if self._was_action_successful(action) else -0.5,
            'user_satisfaction': self._get_user_satisfaction_score() * 0.2,
            'system_efficiency': self._calculate_efficiency_score(action) * 0.1
        }

        # Calculate total reward
        total_reward = sum(reward_components.values())

        reward = Reward(
            value=total_reward,
            components=reward_components,
            timestamp=datetime.now(),
            reason=f"Reward for {action.action_type.value} action"
        )

        self.reward_history.append(reward)

        # Update performance metrics
        if total_reward > 0:
            self.performance_metrics['successful_actions'] += 1

        # Update average reward
        if self.reward_history:
            self.performance_metrics['average_reward'] = (
                sum(r.value for r in self.reward_history) / len(self.reward_history)
            )

        logger.debug(f"💰 Calculated reward: {total_reward:.3f} for {action.action_type.value}")
        return reward

    def _was_action_successful(self, action: Action) -> bool:
        """Check if an action was successful."""
        # Simplified: In practice, would check actual outcomes
        return random.random() > 0.2  # 80% success rate for now

    def _get_user_satisfaction_score(self) -> float:
        """Get current user satisfaction score."""
        # Get recent sentiment data
        recent_interactions = user_tracker.interaction_history[-20:]  # Last 20 interactions

        if not recent_interactions:
            return 0.5  # Neutral if no data

        # Convert sentiments to scores
        sentiment_scores = {
            UserSentiment.VERY_DISSATISFIED.value: 0.0,
            UserSentiment.DISSATISFIED.value: 0.25,
            UserSentiment.NEUTRAL.value: 0.5,
            UserSentiment.SATISFIED.value: 0.75,
            UserSentiment.VERY_SATISFIED.value: 1.0
        }

        scores = [
            sentiment_scores.get(interaction.sentiment.value, 0.5)
            for interaction in recent_interactions
            if interaction.sentiment
        ]

        return sum(scores) / len(scores) if scores else 0.5

    def _calculate_efficiency_score(self, action: Action) -> float:
        """Calculate efficiency score for the action."""
        # Simplified: In practice, would measure actual resource usage
        base_efficiency = 0.8

        # Adjust based on action complexity
        complexity_penalty = {
            ActionType.CHANGE_LAYOUT: 0.1,
            ActionType.MODIFY_CHART_TYPE: 0.05,
            ActionType.ADJUST_COLOR_SCHEME: 0.02,
            ActionType.REORDER_COMPONENTS: 0.08,
            ActionType.CHANGE_ANALYSIS_METHOD: 0.15
        }

        penalty = complexity_penalty.get(action.action_type, 0.05)
        return max(0.0, base_efficiency - penalty)

    def learn_from_experience(self, state: State, action: Action, reward: Reward, next_state: State):
        """Update the agent's knowledge based on experience."""

        # Simplified Q-learning update
        state_key = self._state_to_key(state)
        action_key = action.action_type.value

        # Initialize Q-table entries if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0

        # Q-learning update rule
        current_q = self.q_table[state_key][action_key]
        max_next_q = self._get_max_q_value(next_state)

        # Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
        updated_q = current_q + self.learning_rate * (
            reward.value + 0.9 * max_next_q - current_q
        )

        self.q_table[state_key][action_key] = updated_q

        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)

        logger.debug(f"🧠 Updated Q-value for {action_key}: {updated_q:.3f}")

    def _state_to_key(self, state: State) -> str:
        """Convert state to a hashable key for Q-table."""
        # Simplified: In practice, would use more sophisticated state representation
        return f"engagement_{state.user_engagement".2f"}"

    def _get_max_q_value(self, state: State) -> float:
        """Get maximum Q-value for a state."""
        state_key = self._state_to_key(state)

        if state_key not in self.q_table:
            return 0.0

        return max(self.q_table[state_key].values()) if self.q_table[state_key] else 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance."""
        summary = {
            'agent_type': self.agent_type.value,
            'total_actions': self.performance_metrics['total_actions'],
            'success_rate': (
                self.performance_metrics['successful_actions'] /
                max(1, self.performance_metrics['total_actions'])
            ),
            'average_reward': self.performance_metrics['average_reward'],
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.q_table),
            'total_rewards': len(self.reward_history),
            'learning_progress': min(1.0, len(self.q_table) / 100)  # Normalize to 0-1
        }

        return summary

    def suggest_improvement(self) -> Dict[str, Any]:
        """Suggest a specific improvement based on learning."""

        # Analyze recent performance
        recent_rewards = self.reward_history[-50:]  # Last 50 rewards

        if not recent_rewards:
            return {'suggestion': 'Continue exploring different approaches', 'confidence': 0.5}

        average_reward = sum(r.value for r in recent_rewards) / len(recent_rewards)

        # Find best performing action types
        action_performance = {}
        for action in self.action_history[-50:]:
            action_type = action.action_type.value
            if action_type not in action_performance:
                action_performance[action_type] = {'total': 0, 'rewards': []}

            action_performance[action_type]['total'] += 1

        # This is simplified - in practice, would correlate actions with rewards
        best_action = max(action_performance.keys(),
                         key=lambda x: action_performance[x]['total'])

        return {
            'suggestion': f'Focus on {best_action} actions as they show promise',
            'confidence': min(0.9, average_reward + 0.5),
            'based_on_recent_performance': True
        }

class AutonomousAnalyticsSystem:
    """Main autonomous analytics system coordinating multiple RL agents."""

    def __init__(self):
        self.agents: Dict[AgentType, SimpleRLAgent] = {}
        self.is_learning = False
        self.last_learning_cycle = None
        self.learning_interval = 300  # 5 minutes between learning cycles

        # Initialize agents
        self._initialize_agents()

        logger.info("🚀 Autonomous Analytics System initialized")

    def _initialize_agents(self):
        """Initialize all RL agents."""
        agent_types = [
            AgentType.DASHBOARD_DESIGNER,
            AgentType.ANALYSIS_STRATEGIST,
            AgentType.REPORT_GENERATOR
        ]

        for agent_type in agent_types:
            self.agents[agent_type] = SimpleRLAgent(agent_type)

        logger.info(f"🤖 Initialized {len(self.agents)} autonomous agents")

    def start_learning_cycle(self):
        """Start an autonomous learning cycle."""
        if self.is_learning:
            return

        self.is_learning = True
        logger.info("🧠 Starting autonomous learning cycle")

        try:
            # Get current state
            current_state = self._get_current_state()

            # Let each agent choose and execute actions
            for agent_type, agent in self.agents.items():
                # Choose action
                action = agent.choose_action(current_state)

                # Execute action (simplified for now)
                self._execute_action(action)

                # Get next state
                next_state = self._get_current_state()

                # Calculate reward
                reward = agent.calculate_reward(current_state, action, next_state)

                # Learn from experience
                agent.learn_from_experience(current_state, action, reward, next_state)

            self.last_learning_cycle = datetime.now()

        except Exception as e:
            logger.error(f"❌ Error in learning cycle: {e}")
        finally:
            self.is_learning = False

    def _get_current_state(self) -> State:
        """Get current system state."""
        # This would integrate with the actual dashboard and analysis systems
        return State(
            user_engagement=0.7,
            system_performance=0.85,
            data_characteristics={'sample': 'current'},
            recent_interactions=['dashboard_view', 'chart_interaction'],
            current_dashboard_config={'layout': 'grid', 'charts': 3},
            analysis_results={'accuracy': 0.85},
            timestamp=datetime.now()
        )

    def _execute_action(self, action: Action):
        """Execute an action chosen by an RL agent."""
        logger.info(f"🎯 Executing {action.agent_type.value} action: {action.action_type.value}")

        # In practice, this would modify the actual dashboard/analysis system
        # For now, just log the action
        action_data = action.to_dict()
        logger.debug(f"📝 Action details: {action_data}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and learning progress."""
        status = {
            'system_active': True,
            'total_agents': len(self.agents),
            'learning_active': self.is_learning,
            'last_learning_cycle': self.last_learning_cycle.isoformat() if self.last_learning_cycle else None,
            'agents_status': {},
            'overall_performance': 0.0
        }

        # Get status from each agent
        agent_performances = []
        for agent_type, agent in self.agents.items():
            agent_summary = agent.get_performance_summary()
            status['agents_status'][agent_type.value] = agent_summary
            agent_performances.append(agent_summary.get('average_reward', 0.0))

        # Calculate overall performance
        if agent_performances:
            status['overall_performance'] = sum(agent_performances) / len(agent_performances)

        return status

    def get_autonomous_insights(self) -> Dict[str, Any]:
        """Get insights generated by the autonomous system."""
        insights = {
            'system_maturity': 'early_learning',
            'recommended_improvements': [],
            'learning_progress': {},
            'autonomous_suggestions': []
        }

        # Get suggestions from each agent
        for agent_type, agent in self.agents.items():
            suggestion = agent.suggest_improvement()
            insights['autonomous_suggestions'].append({
                'agent': agent_type.value,
                'suggestion': suggestion
            })

            # Get learning progress
            progress = agent.get_performance_summary()
            insights['learning_progress'][agent_type.value] = progress

        return insights

# Global autonomous system instance
autonomous_system = AutonomousAnalyticsSystem()
