import os
import uuid
import json
from typing import Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults

## Environment Variable
OPENAI_API_KEY="sk-proj-xxxxx" # https://platform.openai.com/account/api-keys
TAVILY_API_KEY="tvly-xxxx" # https://tavily.com/account/api-keys
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY


#define model (gpt-4o) - try different models
model = ChatOpenAI(model='gpt-4o')

class State(TypedDict):
    messages: Annotated[list, add_messages]


# Here, we are building an agent using LangGraph. In LangGraph, the State class represents the schema of the graph state. Graph state is a shared data structure that holds the data necessary for the nodes in the graph to operate and communicate with each other.
# For this example, we have created two tools: add and Tavily Search. Toolexectutor(tools) that initializes a ToolExecutor with the list of tools. The ToolExecutor is responsible for managing and executing these tools as needed. model.bind_tools(tools) method binds the list of tools to the model. This means that the model is now aware of the available tools and can utilize them during its operations.

## define two tools- internet search and simply add
@tool
def add(x,y):
    "adding two numbers"
    return x+y

tools = [TavilySearchResults(max_results=1), add]

#
tool_executor = ToolExecutor(tools)
model = model.bind_tools(tools)

# Before building a graph, let’s define the different components first. The first function, should_continue, determines whether the process should end or continue to other nodes based on the last message (messages[-1]). If tool_calls is not available, the process will end; otherwise, it will continue. The call_model function allows us to invoke the previously defined model and return the response. The third function, call_tool, efficiently handles the execution of a tool based on the latest message in the state. It extracts the most recent message to identify the tool call, constructs an invocation for the specified tool, executes the tool using a tool executor, and creates a new message with the tool’s response. This new message is then returned to be added to the existing conversation state. This function is essential for enabling dynamic tool invocation and response handling within the LangGraph framework.

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}


# Next, let’s defines a new graph using LangGraph, setting up a workflow with two nodes: agent and action. The agent node is defined to call a model (call_model), and the action node is defined to call a tool (call_tool). The entry point is set to the agent node, making it the initial node in the workflow. A conditional edge is added from the agent node, using the should_continue function to determine the next node: either continuing to the action node or ending the workflow. Finally, a normal edge is added from the action node back to the agent node, creating a cycle between these two nodes.

# Define a new graph
workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# The following two lines are crucial for the Human-in-the-Loop framework. To ensure persistence, include a checkpoint when compiling the graph, which is necessary to support interrupts. We use SqliteSaver for an in-memory SQLite database to save the state. To consistently interrupt before a specific node, we should provide the node’s name to the compile method:

memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# This means that when we provide input to the graph, the execution begins. Once the agent selects a tool, the run will be interrupted before reaching the action node, allowing a human to provide input. We manually update the graph’s state with the human input and then resume the graph’s execution. After the graph is interrupted, we perform the following steps, which occur only if a tool call is generated before the interruption:
# Append a verification AIMessage (output from the helper function generate_verification_message) to the state asking for user approval.
# Receive user input and append it to the state as a HumanMessage.
# If approved, append the tool call message to the state and resume execution.
# Otherwise, resume execution from the new user input.

# Helper function to construct message asking for verification
def generate_verification_message(message: AIMessage) -> None:
    """Generate "verification message" from message with tool calls."""
    serialized_tool_calls = json.dumps(
        message.tool_calls,
        indent=2,
    )
    return AIMessage(
        content=(
            "I plan to invoke the following tools, do you approve?\n\n"
            "Type 'y' if you do, anything else to stop.\n\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )

# Helper function to stream output from the graph
def stream_app_catch_tool_calls(inputs, thread) -> Optional[AIMessage]:
    """Stream app, catching tool calls."""
    tool_call_message = None
    for event in app.stream(inputs, thread, stream_mode="values"):
        message = event["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_message = message
        else:
            #print(message)
            message.pretty_print()
            if isinstance(message, AIMessage):
                st.write(message.content)

    return tool_call_message

# The helper function generate_verification_message assists in creating the verification message, while stream_app_catch_tool_calls extracts the tool call message.
# In a human-in-the-loop scenario, threads and memory play a crucial role in tracking and managing the flow of conversations, ensuring continuity and context retention. Threads are sequences of traces that represent a single conversation or interaction within the system. Threads help group related traces together, allowing for a coherent view of a conversation or dialogue. On the other hand, memory in LangGraph refers to the checkpointed state of the conversation, capturing all modifications and inputs made during the interaction.


# The following code snippet performs these steps:
# The user provides an input (problem statement), and based on this input, tool selection occurs (tool_call_message), and we display the tool_name.
# A verification message is generated using the generate_verification_message function.
# The user is asked to provide a tool approval message (“yes” or another response).
# When the user provides a approval message, both the verification message generated in step two and the user’s message from step three are added to the graph’s state. This is done by the line of code snapshot.values[“messages”] += [verification_message, input_message].
# If the user provides “yes”, we further add the tool_call_message to the graph’s state. This indicates that we approve this tool and agree to proceed with the graph execution using the selected tool.
# Execute the rest of the graph with the modified state. If the user provides a response other than “yes,” the graph execution proceeds without the tool_call_message.


st.title('Human in The Loop - Agent')

user_input = st.text_input("Enter your question:", key="input1")
#if st.button("Submit Question"):

if user_input:
    thread = {"configurable": {"thread_id": "5"}}
    #inputs = [HumanMessage(content="what's the weather in sf now?")]

    inputs = [HumanMessage(content=user_input)]
    # for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    #     event["messages"][-1].pretty_print()

    tool_call_message = stream_app_catch_tool_calls(
        {"messages": inputs},
        thread,
    )

    # tool name:
    tool_name=tool_call_message.tool_calls[-1]['name']
    #st.write(tool_call_message.tool_calls[-1])
    st.write(f":blue[tool invoked]: {tool_name} ")

    st.write(":green[Please approve the tool picked up by the agent - select either 'yes' or 'no' ]")

    verification_message = generate_verification_message(tool_call_message)
    #verification_message.pretty_print()

    #st.write(verification_message)

    #human_input=input("Please provide your response")
    human_input = st.text_input('Please provide your response', key='keyname')
    if human_input:

        input_message = HumanMessage(human_input)
        # if input_message.content == "exit":
        #     break

        #st.write(input_message)
        #input_message.pretty_print()

        # First we update the state with the verification message and the input message.
        # note that `generate_verification_message` sets the message ID to be the same
        # as the ID from the original tool call message. Updating the state with this
        # message will overwrite the previous tool call.
        snapshot = app.get_state(thread)
        snapshot.values["messages"] += [verification_message, input_message]

        if input_message.content == "yes":
            tool_call_message.id = str(uuid.uuid4())
            # If verified, we append the tool call message to the state
            # and resume execution.
            snapshot.values["messages"] += [tool_call_message]
            app.update_state(thread, snapshot.values, as_node="agent")
            tool_call_message = stream_app_catch_tool_calls(None, thread)
        else:
            # Otherwise, resume execution from the input message.
            app.update_state(thread, snapshot.values, as_node="__start__")
            tool_call_message = stream_app_catch_tool_calls(None, thread)