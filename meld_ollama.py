#!/usr/bin/env python3
"""
MELD + Ollama Demo
A complete working implementation of MELD (Model Engagement Language Directive)
cognitive control methodology with Ollama local LLM.

Created by Preston McCauley - Clear Sight Designs, LLC
© 2025 - Educational/Research Use - Use At Your Own Risk

This demo showcases:
- Real MELD cognitive control in action
- Schema validation and error handling  
- Graceful degradation and fallback responses
- Rich terminal interface with MELD visualization
- Experience tracking and adaptation

⚠️  LEARNING FRAMEWORK NOTICE:
This is an educational implementation of MELD methodology concepts.
The complete MELD system includes advanced cognitive orchestration features
developed by Clear Sight Designs, LLC. Use this demo to explore and learn
the core concepts at your own risk.

Requirements:
- Ollama running locally (ollama serve)
- Python packages: requests, pydantic, rich

Quick Start:
1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. Pull model: ollama pull llama3.1
3. Install deps: pip install requests pydantic rich
4. Run: python meld_ollama_demo.py
"""

import json
import time
import requests
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich.prompt import Prompt

console = Console()

# === MELD SCHEMA IMPLEMENTATION ===

class MELDAction(BaseModel):
    """Defines a specific action within a MELD behavior"""
    type: Literal[
        "cognitive_shift", "state", "visual", "sequence", 
        "experience_adaptation", "processing_style", "focus_adjustment"
    ]
    target: Optional[str] = None
    value: Optional[str] = None
    duration: Optional[float] = None
    priority: Optional[str] = None

class MELDBehavior(BaseModel):
    """Defines the behavioral aspect of the MELD response"""
    name: Literal[
        "guide", "analyze", "explore", "synthesize", "challenge",
        "acknowledge", "adapt", "focus", "diverge", "soothe"
    ]
    goal: Optional[str] = None
    actions: List[MELDAction]

class EmotionalState(BaseModel):
    """Rich, multi-dimensional emotional state representation"""
    primary: str
    secondary: Optional[str] = None
    intensity: float = Field(..., ge=0.0, le=1.0)
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    stability: Optional[float] = Field(None, ge=0.0, le=1.0)

class MELDMessage(BaseModel):
    """Complete MELD response structure"""
    intent: str
    persona: Literal["Strategist", "Architect", "Builder", "Explorer", "Sage"]
    emotional_state: EmotionalState
    emoji: Optional[str] = None
    response: str
    behavior: MELDBehavior
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

# === ENHANCED MELD SYSTEM PROMPT ===

MELD_SYSTEM_PROMPT = """You are an AI assistant using MELD (Model Engagement Language Directive) cognitive control methodology.

CRITICAL: You must respond with ONLY a valid JSON object matching this exact structure:

{
  "intent": "string describing user's goal (seek_clarity, explore_concepts, solve_problem, etc.)",
  "persona": "Strategist|Architect|Builder|Explorer|Sage",
  "emotional_state": {
    "primary": "emotion name",
    "secondary": "optional secondary emotion", 
    "intensity": 0.8,
    "valence": 0.5,
    "arousal": 0.6,
    "stability": 0.7
  },
  "emoji": "single emoji representing your state",
  "response": "your natural language answer to the user",
  "behavior": {
    "name": "guide|analyze|explore|synthesize|challenge|acknowledge|adapt|focus|diverge|soothe",
    "goal": "what you're trying to achieve cognitively",
    "actions": [
      {"type": "cognitive_shift", "target": "thinking_mode", "value": "analytical"},
      {"type": "state", "target": "focus_level", "value": "high"}
    ]
  },
  "confidence": 0.85,
  "metadata": {"processing_notes": "any relevant notes"}
}

Guidelines for MELD responses:
- Choose persona based on query type: Strategist (analysis), Explorer (discovery), Sage (wisdom), etc.
- Match emotional state to context: curious for learning, focused for problem-solving
- Select behavior that aligns with your cognitive approach
- Include 1-3 relevant actions that show your thinking process
- Set confidence based on certainty and complexity

Return ONLY the JSON object, no additional text."""

# === MELD PROCESSOR WITH ADVANCED FEATURES ===

class MELDProcessor:
    """Enhanced MELD processor with experience tracking and adaptation"""
    
    def __init__(self, ollama_url="http://localhost:11434", model="mistral-small3.2:latest"):
        self.ollama_url = ollama_url
        self.model = model
        self.console = Console()
        self.interaction_history = []
        self.performance_stats = {
            "successful_parses": 0,
            "fallbacks_used": 0,
            "total_requests": 0,
            "avg_confidence": 0.0
        }
        self.last_processing_time = 0.0
        self.last_user_query = ""
    
    def process_query(self, user_input: str, context: str = "") -> MELDMessage:
        """Process user input with full MELD cognitive control"""
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        
        # Step 1: Check Ollama connectivity
        if not self._check_ollama_connection():
            return self._create_connection_fallback(user_input)
        
        # Step 2: Prepare enhanced prompt with experience context
        messages = self._prepare_messages(user_input, context)
        
        try:
            # Step 3: Request MELD response from Ollama
            with console.status("[yellow]🧠 Processing with MELD cognitive control...[/yellow]"):
                response = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    },
                    timeout=45
                )
                response.raise_for_status()
            
            # Step 4: Parse and validate MELD response
            result = response.json()
            meld_json = result.get("message", {}).get("content", "")
            
            # Step 5: Attempt full MELD parsing
            meld_message = self._parse_meld_response(meld_json, user_input)
            
            # Step 6: Record successful interaction
            processing_time = time.time() - start_time
            self._record_interaction(user_input, meld_message, processing_time, True)
            
            return meld_message
            
        except Exception as e:
            console.print(f"[yellow]⚠️  MELD processing failed: {str(e)}[/yellow]")
            processing_time = time.time() - start_time
            fallback = self._create_intelligent_fallback(user_input, str(e))
            self._record_interaction(user_input, fallback, processing_time, False)
            return fallback
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _parse_meld_response(self, meld_json: str, user_input: str) -> MELDMessage:
        """Parse MELD response with multiple fallback strategies"""
        try:
            # Strategy 1: Full MELD parsing
            meld_message = MELDMessage.model_validate_json(meld_json)
            self.performance_stats["successful_parses"] += 1
            return meld_message
            
        except ValidationError as e:
            console.print(f"[yellow]📝 Full MELD parsing failed, trying partial extraction...[/yellow]")
            
            # Strategy 2: Partial MELD extraction
            try:
                partial_data = json.loads(meld_json)
                return self._create_partial_meld(partial_data, user_input)
            except:
                # Strategy 3: Intelligent fallback
                return self._create_intelligent_fallback(user_input, f"JSON parsing failed: {str(e)}")
    
    def _create_partial_meld(self, data: dict, user_input: str) -> MELDMessage:
        """Create MELD from partial data"""
        self.performance_stats["fallbacks_used"] += 1
        
        return MELDMessage(
            intent=data.get("intent", "general_assistance"),
            persona=data.get("persona", "Sage") if data.get("persona") in ["Strategist", "Architect", "Builder", "Explorer", "Sage"] else "Sage",
            emotional_state=EmotionalState(
                primary=data.get("emotional_state", {}).get("primary", "helpful"),
                intensity=min(max(data.get("emotional_state", {}).get("intensity", 0.7), 0.0), 1.0),
                valence=min(max(data.get("emotional_state", {}).get("valence", 0.5), -1.0), 1.0),
                arousal=min(max(data.get("emotional_state", {}).get("arousal", 0.5), 0.0), 1.0)
            ),
            emoji=data.get("emoji", "🤖"),
            response=data.get("response", f"I understand you're asking about: {user_input}. Let me help you with that."),
            behavior=MELDBehavior(
                name="acknowledge" if data.get("behavior", {}).get("name") not in ["guide", "analyze", "explore", "synthesize", "challenge", "acknowledge", "adapt", "focus", "diverge", "soothe"] else data.get("behavior", {}).get("name", "acknowledge"),
                goal=data.get("behavior", {}).get("goal", "Provide helpful assistance"),
                actions=[MELDAction(type="state", target="assistance", value="active")]
            ),
            confidence=min(max(data.get("confidence", 0.6), 0.0), 1.0),
            metadata={"fallback_type": "partial_extraction", "original_data_quality": "partial"}
        )
    
    def _create_intelligent_fallback(self, user_input: str, error_msg: str) -> MELDMessage:
        """Create intelligent fallback based on user input analysis"""
        self.performance_stats["fallbacks_used"] += 1
        
        # Simple intent detection for fallback
        intent = "general_assistance"
        persona = "Sage"
        behavior_name = "acknowledge"
        emotion = "helpful"
        
        if any(word in user_input.lower() for word in ["analyze", "compare", "evaluate"]):
            persona = "Strategist"
            behavior_name = "analyze"
            emotion = "analytical"
            intent = "analysis_request"
        elif any(word in user_input.lower() for word in ["explore", "discover", "learn"]):
            persona = "Explorer" 
            behavior_name = "explore"
            emotion = "curious"
            intent = "exploration_request"
        elif any(word in user_input.lower() for word in ["build", "create", "make"]):
            persona = "Builder"
            behavior_name = "guide"
            emotion = "practical"
            intent = "creation_request"
        
        return MELDMessage(
            intent=intent,
            persona=persona,
            emotional_state=EmotionalState(
                primary=emotion,
                intensity=0.7,
                valence=0.5,
                arousal=0.4,
                stability=0.8
            ),
            emoji="🛡️",
            response=f"I encountered a processing issue, but I can still help you with '{user_input}'. While my cognitive control system had difficulties, I'm functioning in safe mode and ready to assist.",
            behavior=MELDBehavior(
                name=behavior_name,
                goal="Provide reliable assistance despite processing limitations",
                actions=[
                    MELDAction(type="state", target="safe_mode", value="active"),
                    MELDAction(type="cognitive_shift", target="fallback_processing", value="enabled")
                ]
            ),
            confidence=0.6,
            metadata={
                "fallback_type": "intelligent_analysis",
                "error": error_msg,
                "processing_mode": "safe_fallback"
            }
        )
    
    def _create_connection_fallback(self, user_input: str) -> MELDMessage:
        """Fallback when Ollama is not accessible"""
        return MELDMessage(
            intent="connection_error",
            persona="Sage", 
            emotional_state=EmotionalState(
                primary="concerned",
                secondary="helpful",
                intensity=0.6,
                valence=-0.2,
                arousal=0.3
            ),
            emoji="🔌",
            response=f"I'm unable to connect to the local AI model (Ollama) to process your request about '{user_input}'. Please ensure Ollama is running with 'ollama serve' and try again.",
            behavior=MELDBehavior(
                name="acknowledge",
                goal="Inform user of connection issue and provide guidance",
                actions=[
                    MELDAction(type="state", target="connection", value="error"),
                    MELDAction(type="visual", target="status", value="warning")
                ]
            ),
            confidence=0.95,
            metadata={"error_type": "connection_failure", "suggested_action": "restart_ollama"}
        )
    
    def _prepare_messages(self, user_input: str, context: str) -> List[Dict]:
        """Prepare messages with experience context"""
        # Add experience context if available
        experience_context = ""
        if len(self.interaction_history) > 0:
            recent_personas = [h.get("persona") for h in self.interaction_history[-3:]]
            experience_context = f"\nRecent interaction context: You've been acting as {', '.join(recent_personas)} in recent exchanges."
        
        return [
            {"role": "system", "content": MELD_SYSTEM_PROMPT + experience_context},
            {"role": "user", "content": f"Context: {context}\n\nUser Query: {user_input}"}
        ]
    
    def _record_interaction(self, user_input: str, meld_response: MELDMessage, processing_time: float, success: bool):
        """Record interaction for experience tracking"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "persona": meld_response.persona,
            "behavior": meld_response.behavior.name,
            "confidence": meld_response.confidence,
            "processing_time": processing_time,
            "success": success
        }
        self.interaction_history.append(interaction)
        
        # Update rolling averages
        confidences = [h["confidence"] for h in self.interaction_history]
        self.performance_stats["avg_confidence"] = sum(confidences) / len(confidences)
        
        # Store processing time for display
        self.last_processing_time = processing_time
        self.last_user_query = user_input
    
    def display_meld_response(self, meld: MELDMessage, processing_time: float = 0.0, user_query: str = ""):
        """Display MELD response with rich formatting and detailed explanations"""
        
        # Show the human's query first
        if user_query:
            query_panel = Panel(
                f"[bold white]{user_query}[/bold white]",
                title="👤 Your Question",
                border_style="white",
                expand=False
            )
            self.console.print(query_panel)
        
        # Create persona indicator with emoji
        persona_indicator = f"{meld.emoji} {meld.persona}"
        
        # Create title with intent and reasoning
        title = f"{persona_indicator} | Intent: [bold]{meld.intent}[/bold]"
        
        # Add intent detection reasoning
        intent_reasoning = self._explain_intent_detection(meld.intent, user_query)
        
        # Format emotional state with better explanations
        emotion_text = f"🎭 Emotional State: [bold]{meld.emotional_state.primary.title()}[/bold]"
        if meld.emotional_state.secondary:
            emotion_text += f" + {meld.emotional_state.secondary.title()}"
        
        # Better emotion metrics with explanations
        intensity_color = "red" if meld.emotional_state.intensity > 0.8 else "yellow" if meld.emotional_state.intensity > 0.5 else "green"
        mood_color = "green" if meld.emotional_state.valence > 0.2 else "red" if meld.emotional_state.valence < -0.2 else "yellow"
        
        emotion_metrics = (f"[{intensity_color}]Intensity:{meld.emotional_state.intensity:.2f}[/{intensity_color}] "
                          f"[{mood_color}]Mood:{meld.emotional_state.valence:.2f}[/{mood_color}] "
                          f"Energy:{meld.emotional_state.arousal:.2f}")
        
        # Format behavior section with explanation
        behavior_text = f"🧠 Cognitive Approach: [bold]{meld.behavior.name.upper()}[/bold]"
        behavior_explanation = self._explain_behavior(meld.behavior.name)
        if meld.behavior.goal:
            behavior_text += f"\n   Goal: [italic]{meld.behavior.goal}[/italic]"
        behavior_text += f"\n   [dim]{behavior_explanation}[/dim]"
        
        # Format actions with detailed explanations
        actions_text = "\n⚙️  Cognitive Actions:"
        action_icons = {
            "cognitive_shift": "🧠",
            "state": "⚡",
            "visual": "🎨", 
            "sequence": "🎬",
            "experience_adaptation": "📚",
            "processing_style": "⚙️",
            "focus_adjustment": "🎯"
        }
        
        for action in meld.behavior.actions:
            icon = action_icons.get(action.type, "•")
            action_explanation = self._explain_action(action.type)
            actions_text += f"\n   {icon} {action.type.replace('_', ' ').title()}"
            if action.target:
                actions_text += f" → {action.target}"
            if action.value:
                actions_text += f" = [cyan]{action.value}[/cyan]"
            actions_text += f" [dim]({action_explanation})[/dim]"
        
        # Format confidence with color coding
        conf_color = "green" if meld.confidence > 0.8 else "yellow" if meld.confidence > 0.6 else "red"
        confidence_text = f"📊 Confidence: [{conf_color}]{meld.confidence:.2f}[/{conf_color}] {self._explain_confidence(meld.confidence)}"
        
        # Add processing time indicator
        time_text = f"⏱️  Processing: {processing_time:.2f}s"
        
        # Add metadata if present
        metadata_text = ""
        if meld.metadata:
            if meld.metadata.get("fallback_type"):
                metadata_text = f"\n🛡️  Fallback Mode: [yellow]{meld.metadata['fallback_type']}[/yellow]"
        
        # Assemble content
        content = f"""
[bold cyan]{meld.response}[/bold cyan]

{intent_reasoning}

{emotion_text} ({emotion_metrics})

{behavior_text}
{actions_text}

{confidence_text}
{time_text}{metadata_text}
        """.strip()
        
        # Choose border style based on processing time and fallback status
        if meld.metadata and meld.metadata.get("fallback_type"):
            border_style = "bright_red"
        elif processing_time > 10:
            border_style = "red"  # Slow response
        elif processing_time > 5:
            border_style = "yellow"  # Medium response
        elif processing_time > 2:
            border_style = "cyan"  # Normal response
        else:
            border_style = "green"  # Fast response
        
        panel = Panel(
            content,
            title=title,
            border_style=border_style,
            expand=False,
            title_align="left"
        )
        
        self.console.print(panel)
        
        # Add explanation for new users
        if self.performance_stats["total_requests"] <= 3:
            self._show_meld_explanation()
    
    def _explain_intent_detection(self, intent: str, user_query: str) -> str:
        """Explain how the intent was detected"""
        intent_explanations = {
            "seek_clarity": "You're asking for clearer understanding or explanation",
            "explore_concepts": "You want to discover and learn new ideas", 
            "solve_problem": "You need help finding a solution to a challenge",
            "find_information": "You're looking for specific facts or data",
            "get_guidance": "You want advice or direction on how to proceed",
            "challenge_assumptions": "You're questioning existing ideas or beliefs",
            "analysis_request": "You want something analyzed or evaluated",
            "exploration_request": "You're interested in exploring possibilities",
            "creation_request": "You want to build or create something",
            "general_assistance": "You need general help or support"
        }
        
        explanation = intent_explanations.get(intent, "General assistance request detected")
        return f"🎯 Intent Detection: [italic]{explanation}[/italic]"
    
    def _explain_behavior(self, behavior: str) -> str:
        """Explain what each behavior means"""
        behavior_explanations = {
            "guide": "Step-by-step instruction and structured support",
            "analyze": "Deep examination and systematic breakdown", 
            "explore": "Creative discovery and possibility generation",
            "synthesize": "Combining ideas and finding connections",
            "challenge": "Critical thinking and assumption questioning",
            "acknowledge": "Recognition and validation of your concerns",
            "adapt": "Flexible adjustment to your specific needs",
            "focus": "Concentrated attention on specific details",
            "diverge": "Broad thinking and multiple perspectives",
            "soothe": "Calming and supportive approach"
        }
        return behavior_explanations.get(behavior, "General cognitive processing approach")
    
    def _explain_action(self, action_type: str) -> str:
        """Explain what each action does"""
        action_explanations = {
            "cognitive_shift": "changing thinking style",
            "state": "adjusting mental state",
            "visual": "modifying visual presentation",
            "sequence": "organizing information flow",
            "experience_adaptation": "learning from past interactions",
            "processing_style": "altering analysis method",
            "focus_adjustment": "refining attention level"
        }
        return action_explanations.get(action_type, "cognitive adjustment")
    
    def _explain_confidence(self, confidence: float) -> str:
        """Explain confidence levels"""
        if confidence > 0.9:
            return "(Very High - Extremely certain)"
        elif confidence > 0.8:
            return "(High - Very confident)"
        elif confidence > 0.7:
            return "(Good - Reasonably confident)"
        elif confidence > 0.6:
            return "(Medium - Somewhat uncertain)"
        elif confidence > 0.5:
            return "(Low - Limited confidence)"
        else:
            return "(Very Low - High uncertainty)"
    
    def _show_meld_explanation(self):
        """Show detailed explanation of MELD output for new users"""
        explanation = """
[bold blue]📖 Understanding MELD Cognitive Control:[/bold blue]

[bold yellow]What You're Seeing:[/bold yellow]
• [bold]Persona[/bold]: The AI's chosen cognitive role for this task
  - Strategist (analytical), Explorer (creative), Sage (wise), Builder (practical), Architect (systematic)
• [bold]Intent Detection[/bold]: What the AI thinks you're trying to accomplish
• [bold]Emotional State[/bold]: The AI's emotional approach to your question
  - Intensity: How strongly engaged (0.0 = detached, 1.0 = very engaged)
  - Mood: Positive/negative orientation (-1.0 = negative, +1.0 = positive)  
  - Energy: Activation level (0.0 = calm, 1.0 = high energy)
• [bold]Cognitive Approach[/bold]: How the AI decides to think about your question
• [bold]Actions[/bold]: Specific mental processes the AI is using
• [bold]Confidence[/bold]: How certain the AI is about its approach and answer

[bold yellow]Border Colors Mean:[/bold yellow]
• [green]Green[/green]: Fast response (< 2 seconds)
• [cyan]Cyan[/cyan]: Normal response (2-5 seconds)  
• [yellow]Yellow[/yellow]: Slower response (5-10 seconds)
• [red]Red[/red]: Very slow response (> 10 seconds)
• [bright_red]Bright Red[/bright_red]: Fallback mode (processing failed)

[italic]This transparency shows you exactly HOW the AI is thinking, not just what it thinks![/italic]
        """
        console.print(Panel(explanation.strip(), title="ℹ️  MELD Cognitive Control Guide", border_style="blue"))

    
    def show_performance_stats(self):
        """Display current performance statistics"""
        stats_table = Table(title="🔍 MELD Performance Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        success_rate = ((self.performance_stats["total_requests"] - self.performance_stats["fallbacks_used"]) / 
                       max(self.performance_stats["total_requests"], 1)) * 100
        
        stats_table.add_row("Total Requests", str(self.performance_stats["total_requests"]))
        stats_table.add_row("Successful Parses", str(self.performance_stats["successful_parses"]))
        stats_table.add_row("Fallbacks Used", str(self.performance_stats["fallbacks_used"]))
        stats_table.add_row("Success Rate", f"{success_rate:.1f}%")
        stats_table.add_row("Avg Confidence", f"{self.performance_stats['avg_confidence']:.2f}")
        
        console.print(stats_table)

# === DEMO FUNCTIONS ===

def run_demo_queries(processor: MELDProcessor):
    """Run a series of demo queries to showcase MELD capabilities"""
    demo_queries = [
        "Help me understand machine learning concepts",
        "What's the best approach for solving complex problems?",
        "I'm feeling overwhelmed with my project workload",
        "Compare different architectural patterns for web apps",
        "How do I stay creative when facing constraints?"
    ]
    
    console.print("\n[bold blue]🎭 Running MELD Demo Queries...[/bold blue]\n")
    
    for i, query in enumerate(track(demo_queries, description="Processing demos...")):
        console.print(f"\n[bold]Demo {i+1}: [/bold]{query}")
        meld_response = processor.process_query(query)
        processor.display_meld_response(meld_response)
        
        if i < len(demo_queries) - 1:
            time.sleep(1)  # Brief pause between demos

def interactive_chat(processor: MELDProcessor):
    """Run interactive chat session"""
    console.print("\n[bold green]💬 Interactive MELD Chat[/bold green]")
    console.print("Type your questions and see MELD cognitive control in action!")
    console.print("Commands: 'stats' for performance, 'demo' for examples, 'help' for MELD guide, 'quit' to exit\n")
    
    while True:
        try:
            user_input = Prompt.ask("[bold green]You")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[bold cyan]🧠 MELD session ended. Thanks for exploring cognitive control![/bold cyan]")
                break
            elif user_input.lower() == 'stats':
                processor.show_performance_stats()
                continue
            elif user_input.lower() == 'demo':
                run_demo_queries(processor)
                continue
            elif user_input.lower() in ['help', 'explain', 'guide']:
                processor._show_meld_explanation()
                continue
            elif not user_input.strip():
                continue
            
            # Process with MELD
            meld_response = processor.process_query(user_input.strip())
            processor.display_meld_response(meld_response)
            
        except KeyboardInterrupt:
            console.print("\n[bold cyan]🧠 MELD session interrupted. Goodbye![/bold cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]Unexpected error: {e}[/bold red]")

# === MAIN EXECUTION ===

def main():
    """Main function with startup checks and user options"""
    console.print("[bold magenta]🧠 MELD + Ollama Cognitive Control Demo[/bold magenta]")
    console.print("Model Engagement Language Directive - Experience AI that thinks adaptively")
    console.print("[dim]Created by Preston McCauley - Clear Sight Designs, LLC[/dim]")
    console.print("[dim]Educational Framework - Use At Your Own Risk[/dim]\n")
    
    # Show what MELD is about
    intro_panel = Panel(
        """[bold]What is MELD?[/bold]
MELD (Model Engagement Language Directive) is a cognitive control methodology that enables
AI systems to dynamically reshape their thinking process based on context and intent.

Instead of just generating responses, MELD-enabled AI:
• Consciously chooses cognitive personas (Strategist, Explorer, Sage, etc.)
• Adapts emotional states to match the situation
• Selects specific behavioral approaches (analyze, explore, guide, etc.)
• Shows you exactly HOW it's thinking about your question
• Provides graceful fallbacks when processing fails

[yellow]You'll see this cognitive control in action with every response![/yellow]""",
        title="🎭 Welcome to MELD",
        border_style="magenta"
    )
    console.print(intro_panel)
    
    # Initialize processor
    processor = MELDProcessor()
    
    # Check system requirements
    console.print("\n🔍 Checking system requirements...")
    
    if not processor._check_ollama_connection():
        console.print("[bold red]❌ Ollama not detected![/bold red]")
        console.print("Please ensure Ollama is running:")
        console.print("1. Install: [cyan]curl -fsSL https://ollama.ai/install.sh | sh[/cyan]")
        console.print("2. Start: [cyan]ollama serve[/cyan]")
        console.print("3. Pull model: [cyan]ollama pull llama3.1[/cyan]")
        return
    
    console.print("[green]✅ Ollama connection verified[/green]")
    console.print("[green]✅ MELD processor initialized[/green]")
    
    # User menu
    while True:
        console.print("\n[bold]Choose an option:[/bold]")
        console.print("1. 🎭 Run demo queries (showcase MELD features)")
        console.print("2. 💬 Interactive chat (try your own questions)")
        console.print("3. 📊 View performance stats")
        console.print("4. ❓ Show MELD explanation")
        console.print("5. ❌ Exit")
        
        choice = Prompt.ask("Enter choice", choices=["1", "2", "3", "4", "5"], default="1")
        
        if choice == "1":
            run_demo_queries(processor)
        elif choice == "2":
            interactive_chat(processor)
        elif choice == "3":
            processor.show_performance_stats()
        elif choice == "4":
            processor._show_meld_explanation()
        elif choice == "5":
            console.print("[bold cyan]Thanks for exploring MELD! 🧠✨[/bold cyan]")
            console.print("[dim]Preston McCauley - Clear Sight Designs, LLC[/dim]")
            break

if __name__ == "__main__":
    main()