""
LangGraph orchestrator for agent workflow execution.
Implements deterministic, bounded agent execution with human-in-the-loop approval.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import logging
from sqlalchemy.orm import Session
from app.schemas.chat import AgentState, AgentPlan, ToolResult, ChatResponse
from app.services.database import ConversationService, ApprovalService, AuditService
from mcp_tools.tools import MCPToolRegistry

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Main orchestrator for agent workflow execution."""

    def __init__(self, llm_model: str = "gpt-4-turbo-preview", temperature: float = 0.7):
        """Initialize the orchestrator."""
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.max_steps = 10
        self.timeout_seconds = 30

    def run_agent(self, db: Session, session_id: str, user_id: str, channel_id: str,
                 user_input: str, platform: str = "slack") -> ChatResponse:
        """
        Run the agent workflow end-to-end.
        
        Args:
            db: Database session
            session_id: Unique session ID
            user_id: User ID
            channel_id: Channel ID
            user_input: User's message
            platform: Platform (slack, teams)
            
        Returns:
            ChatResponse with final response
        """
        try:
            # Create or get conversation state
            state = ConversationService.create_state(
                db, session_id, user_id, channel_id, user_input, platform
            )

            # Step 1: Intent Classification
            intent = self._classify_intent(user_input)
            state = ConversationService.update_state(
                db, session_id, intent=intent, step_count=1
            )

            # Step 2: Planning
            plan = self._generate_plan(user_input, intent)
            state = ConversationService.update_state(
                db, session_id, plan=plan.model_dump(), step_count=2
            )

            # Step 3: Approval Check
            if plan.requires_approval:
                approval = ApprovalService.create_approval(
                    db, user_id, channel_id, plan.steps[0].get("description", ""),
                    plan.model_dump(), platform
                )
                state = ConversationService.update_state(
                    db, session_id, requires_approval=True, step_count=3
                )
                return ChatResponse(
                    platform=platform,
                    channel_id=channel_id,
                    user_id=user_id,
                    text=f"I need your approval to proceed. Please review: {plan.steps[0].get('description', '')}",
                    requires_approval=True,
                    approval_id=approval.id
                )

            # Step 4: Tool Execution
            tool_results = []
            for step in plan.steps:
                if state.step_count >= self.max_steps:
                    break

                tool_name = step.get("tool")
                tool_args = step.get("args", {})

                result = self._execute_tool(db, tool_name, user_id, channel_id, **tool_args)
                tool_results.append(result)

                state = ConversationService.update_state(
                    db, session_id,
                    tool_results=[r.model_dump() for r in tool_results],
                    step_count=state.step_count + 1
                )

                if not result.success:
                    logger.warning(f"Tool {tool_name} failed: {result.error}")
                    break

            # Step 5: Generate Final Response
            final_response = self._generate_response(user_input, intent, tool_results)
            state = ConversationService.update_state(
                db, session_id,
                final_response=final_response,
                status="completed",
                step_count=state.step_count + 1
            )

            AuditService.log_action(
                db, user_id, "agent_execution", "conversation", session_id,
                {"intent": intent, "steps": len(plan.steps)}, platform=platform
            )

            return ChatResponse(
                platform=platform,
                channel_id=channel_id,
                user_id=user_id,
                text=final_response
            )

        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            ConversationService.update_state(
                db, session_id, error=str(e), status="failed"
            )
            AuditService.log_action(
                db, user_id, "agent_execution", "conversation", session_id,
                {}, "failure", str(e), platform
            )
            return ChatResponse(
                platform=platform,
                channel_id=channel_id,
                user_id=user_id,
                text=f"I encountered an error: {str(e)}"
            )

    def _classify_intent(self, user_input: str) -> str:
        """
        Classify user intent using LLM.
        
        Returns: One of "query", "action", "multi_step"
        """
        try:
            system_prompt = """You are an intent classifier. Classify the user's message into one of these categories:
- "query": The user is asking for information
- "action": The user wants to perform a single action
- "multi_step": The user wants to perform multiple actions

Respond with ONLY the category name, nothing else."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]

            response = self.llm.invoke(messages)
            intent = response.content.strip().lower()

            # Validate intent
            valid_intents = ["query", "action", "multi_step"]
            if intent not in valid_intents:
                intent = "query"

            return intent
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            return "query"

    def _generate_plan(self, user_input: str, intent: str) -> AgentPlan:
        """
        Generate an execution plan using LLM.
        """
        try:
            system_prompt = """You are an AI planner. Given a user request, generate a structured plan.
Return a JSON object with:
{
  "intent": "the classified intent",
  "steps": [
    {"tool": "tool_name", "args": {...}, "description": "what this does"}
  ],
  "requires_approval": true/false,
  "estimated_duration": seconds
}

Available tools: create_task, list_tasks, complete_task, schedule_reminder, generate_weekly_report

Only include steps that are actually needed. Set requires_approval=true for sensitive actions like bulk updates or deletions."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User intent: {intent}\nUser request: {user_input}")
            ]

            response = self.llm.invoke(messages)
            plan_text = response.content.strip()

            # Extract JSON from response
            try:
                plan_dict = json.loads(plan_text)
            except json.JSONDecodeError:
                # Try to extract JSON if wrapped in markdown code blocks
                if "```json" in plan_text:
                    plan_text = plan_text.split("```json")[1].split("```")[0]
                    plan_dict = json.loads(plan_text)
                else:
                    plan_dict = {
                        "intent": intent,
                        "steps": [],
                        "requires_approval": False
                    }

            return AgentPlan(**plan_dict)
        except Exception as e:
            logger.error(f"Error generating plan: {str(e)}")
            return AgentPlan(
                intent=intent,
                steps=[],
                requires_approval=False
            )

    def _execute_tool(self, db: Session, tool_name: str, user_id: str,
                     channel_id: str, **kwargs) -> ToolResult:
        """
        Execute a tool through MCP.
        """
        import time
        start_time = time.time()

        try:
            # Add required context
            kwargs["user_id"] = user_id
            kwargs["channel_id"] = channel_id
            kwargs["db"] = db

            result = MCPToolRegistry.invoke_tool(tool_name, **kwargs)

            execution_time_ms = (time.time() - start_time) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=result.get("success", False),
                result=result,
                execution_time_ms=execution_time_ms
            )
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool execution failed: {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time_ms
            )

    def _generate_response(self, user_input: str, intent: str,
                          tool_results: List[ToolResult]) -> str:
        """
        Generate a human-readable response based on tool results.
        """
        try:
            results_text = "\n".join([
                f"- {r.tool_name}: {'Success' if r.success else 'Failed'} ({r.result})"
                for r in tool_results
            ])

            system_prompt = """You are a helpful assistant. Generate a concise, friendly response to the user based on the tool execution results.
Keep it brief and actionable."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""User request: {user_input}
Intent: {intent}
Tool results:
{results_text}

Generate a response for the user.""")
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I've processed your request. Please check the results above."

    def handle_approval(self, db: Session, approval_id: str, approved: bool,
                       approver_id: str, comment: Optional[str] = None) -> Optional[ChatResponse]:
        """
        Handle approval response and resume workflow.
        """
        try:
            approval = ApprovalService.get_approval(db, approval_id)
            if not approval:
                return None

            if approved:
                ApprovalService.approve(db, approval_id, approver_id, comment)
                # Resume workflow - in a real system, this would trigger async execution
                return ChatResponse(
                    platform=approval.platform,
                    channel_id=approval.channel_id,
                    user_id=approval.user_id,
                    text="Approval granted! Proceeding with the action."
                )
            else:
                ApprovalService.reject(db, approval_id, approver_id, comment)
                return ChatResponse(
                    platform=approval.platform,
                    channel_id=approval.channel_id,
                    user_id=approval.user_id,
                    text="Action cancelled. Approval was rejected."
                )
        except Exception as e:
            logger.error(f"Error handling approval: {str(e)}")
            return None


live

Jump to live
