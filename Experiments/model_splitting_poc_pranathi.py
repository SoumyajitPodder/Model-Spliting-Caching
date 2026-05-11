from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"
prompt = "Hello World"

# Split after layer 15 (1-indexed), resume at layer 16.
stopping_layer = 15
starting_layer = stopping_layer + 1

ATOL = 1e-5
RTOL = 1e-4



def _normalize_hidden(hidden):
    if hidden.dim() == 2:
        return hidden.unsqueeze(0)
    return hidden


def _register_all_layer_hooks(model_obj):
    layer_outputs = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(_module, _input, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            layer_outputs[layer_idx] = _normalize_hidden(hidden.detach().clone())

        return hook_fn

    for idx, layer in enumerate(model_obj.model.layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    return layer_outputs, handles


def _remove_handles(handles):
    for handle in handles:
        handle.remove()


def run_full_path(model_obj, input_ids):
    layer_outputs, handles = _register_all_layer_hooks(model_obj)
    try:
        with torch.no_grad():
            outputs = model_obj(input_ids=input_ids, return_dict=True, use_cache=False)
    finally:
        _remove_handles(handles)

    return layer_outputs, outputs.logits.detach().clone()


def run_split_path(model_obj, input_ids):
    layer_outputs, handles = _register_all_layer_hooks(model_obj)
    bridge = {}
    stop_idx = stopping_layer - 1

    def stop_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        bridge["stop_hidden"] = _normalize_hidden(hidden.detach().clone())
        layer_outputs[stop_idx] = bridge["stop_hidden"]
        raise StopIteration

    def position_hook(_module, _args, kwargs):
        cos, sin = kwargs.get("position_embeddings")
        bridge["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
        bridge["position_ids"] = kwargs.get("position_ids").detach().clone()

    stop_handle = model_obj.model.layers[stop_idx].register_forward_hook(stop_hook)
    pos_handle = model_obj.model.layers[stop_idx].register_forward_pre_hook(
        position_hook, with_kwargs=True
    )

    try:
        with torch.no_grad():
            model_obj(input_ids=input_ids, return_dict=True, use_cache=False)
    except StopIteration:
        pass
    finally:
        stop_handle.remove()
        pos_handle.remove()

    x = bridge["stop_hidden"]
    position_ids = bridge["position_ids"]
    position_embeddings = bridge["position_embeddings"]

    with torch.no_grad():
        for layer_idx in range(starting_layer - 1, len(model_obj.model.layers)):
            x = model_obj.model.layers[layer_idx](
                x,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
            x = _normalize_hidden(x)

        x = model_obj.model.norm(x)
        split_logits = model_obj.lm_head(x).detach().clone()

    _remove_handles(handles)
    return layer_outputs, split_logits


def print_layer_metrics(full_outputs, split_outputs, total_layers):
    matched_layers = 0
    compared_layers = 0

    print("\n========== Per-layer Hidden State Metrics ==========")
    print("Layer | allclose")
    print("-----------------")

    for layer_idx in range(total_layers):
        full_hidden = full_outputs.get(layer_idx)
        split_hidden = split_outputs.get(layer_idx)

        if full_hidden is None or split_hidden is None:
            print(f"{layer_idx + 1:>5} | missing")
            continue

        compared_layers += 1
        is_close = torch.allclose(full_hidden, split_hidden, atol=ATOL, rtol=RTOL)
        matched_layers += int(is_close)
        print(f"{layer_idx + 1:>5} | {str(is_close):>7}")

    layer_accuracy = (matched_layers / compared_layers * 100.0) if compared_layers else 0.0
    print("\n========== Aggregate Metrics ==========")
    print(f"Compared layers: {compared_layers}/{total_layers}")
    print(f"Exact-match layers (allclose): {matched_layers}/{compared_layers}")
    print(f"Layer accuracy: {layer_accuracy:.2f}%")


def print_generation_metrics(tokenizer_obj, full_logits, split_logits):
    full_next_token = torch.argmax(full_logits[:, -1, :], dim=-1)
    split_next_token = torch.argmax(split_logits[:, -1, :], dim=-1)
    token_match = bool(torch.equal(full_next_token, split_next_token))

    full_decoded = tokenizer_obj.decode(full_next_token)
    split_decoded = tokenizer_obj.decode(split_next_token)

    print("\n========== Generation-level Metrics ==========")
    print(f"Next-token IDs match: {token_match}")
    print(f"Full next token id/text: {full_next_token.item()} / {repr(full_decoded)}")
    print(f"Split next token id/text: {split_next_token.item()} / {repr(split_decoded)}")


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    full_layer_outputs, full_logits = run_full_path(model, input_ids)
    split_layer_outputs, split_logits = run_split_path(model, input_ids)

    print_layer_metrics(full_layer_outputs, split_layer_outputs, len(model.model.layers))
    print_generation_metrics(tokenizer, full_logits, split_logits)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"
prompt = "Hello World"

# Split after layer 15 (1-indexed), resume at layer 16.
stopping_layer = 15
starting_layer = stopping_layer + 1

ATOL = 1e-5
RTOL = 1e-4


def _normalize_hidden(hidden):
    if hidden.dim() == 2:
        return hidden.unsqueeze(0)
    return hidden


def _register_all_layer_hooks(model_obj):
    layer_outputs = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(_module, _input, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            layer_outputs[layer_idx] = _normalize_hidden(hidden.detach().clone())

        return hook_fn

    for idx, layer in enumerate(model_obj.model.layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    return layer_outputs, handles


def _remove_handles(handles):
    for handle in handles:
        handle.remove()


def run_full_path(model_obj, input_ids):
    layer_outputs, handles = _register_all_layer_hooks(model_obj)
    try:
        with torch.no_grad():
            outputs = model_obj(input_ids=input_ids, return_dict=True, use_cache=False)
    finally:
        _remove_handles(handles)

    return layer_outputs, outputs.logits.detach().clone()


def run_split_path(model_obj, input_ids):
    layer_outputs, handles = _register_all_layer_hooks(model_obj)
    bridge = {}
    stop_idx = stopping_layer - 1

    def stop_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        bridge["stop_hidden"] = _normalize_hidden(hidden.detach().clone())
        layer_outputs[stop_idx] = bridge["stop_hidden"]
        raise StopIteration

    def position_hook(_module, _args, kwargs):
        cos, sin = kwargs.get("position_embeddings")
        bridge["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
        bridge["position_ids"] = kwargs.get("position_ids").detach().clone()

    stop_handle = model_obj.model.layers[stop_idx].register_forward_hook(stop_hook)
    pos_handle = model_obj.model.layers[stop_idx].register_forward_pre_hook(
        position_hook, with_kwargs=True
    )

    try:
        with torch.no_grad():
            model_obj(input_ids=input_ids, return_dict=True, use_cache=False)
    except StopIteration:
        pass
    finally:
        stop_handle.remove()
        pos_handle.remove()

    x = bridge["stop_hidden"]
    position_ids = bridge["position_ids"]
    position_embeddings = bridge["position_embeddings"]

    with torch.no_grad():
        for layer_idx in range(starting_layer - 1, len(model_obj.model.layers)):
            x = model_obj.model.layers[layer_idx](
                x,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
            x = _normalize_hidden(x)

        x = model_obj.model.norm(x)
        split_logits = model_obj.lm_head(x).detach().clone()

    _remove_handles(handles)
    return layer_outputs, split_logits


def _cosine_similarity(a, b):
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()


def print_layer_metrics(full_outputs, split_outputs, total_layers):
    matched_layers = 0
    compared_layers = 0

    print("\n========== Per-layer Hidden State Metrics ==========")
    print("Layer | allclose | mean_abs_diff | max_abs_diff | cosine_sim")
    print("---------------------------------------------------------------")

    for layer_idx in range(total_layers):
        full_hidden = full_outputs.get(layer_idx)
        split_hidden = split_outputs.get(layer_idx)

        if full_hidden is None or split_hidden is None:
            print(f"{layer_idx + 1:>5} | missing")
            continue

        compared_layers += 1
        is_close = torch.allclose(full_hidden, split_hidden, atol=ATOL, rtol=RTOL)
        matched_layers += int(is_close)

        delta = (full_hidden - split_hidden).abs()
        mean_abs_diff = delta.mean().item()
        max_abs_diff = delta.max().item()
        cosine_sim = _cosine_similarity(full_hidden, split_hidden)

        print(
            f"{layer_idx + 1:>5} | {str(is_close):>7} | "
            f"{mean_abs_diff:>13.6e} | {max_abs_diff:>12.6e} | {cosine_sim:>10.6f}"
        )

    layer_accuracy = (matched_layers / compared_layers * 100.0) if compared_layers else 0.0
    print("\n========== Aggregate Metrics ==========")
    print(f"Compared layers: {compared_layers}/{total_layers}")
    print(f"Exact-match layers (allclose): {matched_layers}/{compared_layers}")
    print(f"Layer accuracy: {layer_accuracy:.2f}%")


def print_generation_metrics(tokenizer_obj, full_logits, split_logits):
    full_next_token = torch.argmax(full_logits[:, -1, :], dim=-1)
    split_next_token = torch.argmax(split_logits[:, -1, :], dim=-1)
    token_match = bool(torch.equal(full_next_token, split_next_token))

    logits_delta = (full_logits[:, -1, :] - split_logits[:, -1, :]).abs()
    mean_abs_logit_diff = logits_delta.mean().item()
    max_abs_logit_diff = logits_delta.max().item()
    logits_cosine = _cosine_similarity(full_logits[:, -1, :], split_logits[:, -1, :])

    full_decoded = tokenizer_obj.decode(full_next_token)
    split_decoded = tokenizer_obj.decode(split_next_token)

    print("\n========== Generation-level Metrics ==========")
    print(f"Next-token IDs match: {token_match}")
    print(f"Full next token id/text: {full_next_token.item()} / {repr(full_decoded)}")
    print(f"Split next token id/text: {split_next_token.item()} / {repr(split_decoded)}")
    print(f"Mean abs logit diff (last step): {mean_abs_logit_diff:.6e}")
    print(f"Max abs logit diff (last step):  {max_abs_logit_diff:.6e}")
    print(f"Logit cosine similarity:          {logits_cosine:.6f}")


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    full_layer_outputs, full_logits = run_full_path(model, input_ids)
    split_layer_outputs, split_logits = run_split_path(model, input_ids)

    print_layer_metrics(full_layer_outputs, split_layer_outputs, len(model.model.layers))
    print_generation_metrics(tokenizer, full_logits, split_logits)

"""
VALIDATION STRATEGY: Hidden State Equivalence
--------------------------------------------------------------------------------
Why we only compare output[0] (the 'Hidden State'):

In a Transformer model, data flows through layers like a relay race. 
The 'Hidden State' is the 'baton' a 3D tensor containing the model's 
compressed conceptual understanding of the input at that specific layer.

1. The 'Source of Truth':
   Each layer in the Transformer is mathematically defined to receive 
   the hidden state from the previous layer as its primary input. 
   If our split is successful, the input to Layer 15 (our 'partial pass') 
   must be identical to the input Layer 15 would have received in a 
   standard, continuous execution.

2. Defining Equivalence:
   Because the model is a deterministic chain of mathematical operations, 
   the 'Hidden State' is the sole driver of the next layer's output. If the 
   Hidden State at Layer 14 is identical (within floating-point tolerance), 
   then the output of Layer 15, 16, and all subsequent layers is guaranteed 
   to be identical, regardless of whether the model was run in one piece 
   or split into chunks.

3. Ignoring Auxiliary Data:
   While each layer returns a tuple (a package containing hidden states, 
   attention weights, and cache objects), only the hidden state (index 0) 
   feeds into the next layer's 'forward' function. The auxiliary data acts 
   as 'spectator' information for this specific workflow. Comparing the 
   Hidden State alone is sufficient to prove that our model-splitting logic 
   maintains the mathematical integrity of the entire inference pipeline.
--------------------------------------------------------------------------------
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path)
#Loading model 
tokenizer = AutoTokenizer.from_pretrained(model_path)
#Loading Tokenizer
prompt = "Hello World"
#Loading prompt
inputs = tokenizer(prompt, return_tensors="pt")
#Tokenizing prompt into input tensors
model.eval()
#neural network enters evaluation mode so it behaves predictably
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
"""
Defining Split Boundary: 

Stop Layer defines the end of the first split
Start Layer defines the beginning of the second split

"""

"write each layer output into a file"
"Generation"


stopping_layer = 15 
#Defining Stop Layer 

starting_layer = 16 
#Defining start layer

def capture_full_pass():
    """
    Full forward pass through all 28 layers.
    Captures layer 14 output and layer 15 output using hooks.
    This is the ground truth we compare everything else against.
    """
    captured = {}
    #Empty Dictionary to store stopping and starting layer outputs

    def hook_layer_stopping(module, input, output):
        """
        Forward hook to capture output of stopping layer

        What is a forward hook?
            A user-defined function that allows us to "register" a layer
            PyTorch is instructed to call this function every time a layer calls the forward() method

        Two types of forward hooks 
            Forward pre-hook: Executes before the layer does its math we can see the input data
            
            Forward hook: Executes after the layer completes its math but before the data is 
            passed into the next layer
        
        Args:
            module: The PyTorch layer the hook is attached to.
            input: The input tuple to the layer.
            output: The output tuple returned by the layer.

        Details:
        - Intercepts the forward pass at the specified layer.
        - Extracts the primary hidden state (index 0).
        - Calls .detach() to disconnect from the gradient computation graph.
        - Calls .clone() to create an independent memory copy of the tensor.

        output[0] is the hidden state (multidimensional tensors) calculated by the current layer
        .detach() frees up memory from PyTorch. Removes the history tracking for each tensor 
        .clone() Ensures the data we save doesnt get overwritten by in place operations 
        """

        captured["stopping"] = output[0].detach().clone()
        # Captures hidden state data

        if output[0].dim() == 2:
            captured["stopping"] = captured["stopping"].unsqueeze(0)
        
        # Fixes dimensions of the output 

    def hook_layer_starting(module, input, output):
        """
        Forward hook to capture output of stopping layer

        What is a forward hook?
            A user-defined function that allows us to "register" a layer
            PyTorch is instructed to call this function every time a layer calls the forward() method

        Two types of forward hooks 
            Forward pre-hook: Executes before the layer does its math we can see the input data
            
            Forward hook: Executes after the layer completes its math but before the data is 
            passed into the next layer
        
        Args:
            module: The PyTorch layer the hook is attached to.
            input: The input tuple to the layer.
            output: The output tuple returned by the layer.

        Details:
        - Intercepts the forward pass at the specified layer.
        - Extracts the primary hidden state (index 0).
        - Calls .detach() to disconnect from the gradient computation graph.
        - Calls .clone() to create an independent memory copy of the tensor.

        output[0] is the hidden state (multidimensional tensors) calculated by the current layer
        .detach() frees up memory from PyTorch. Removes the history tracking for each tensor 
        .clone() Ensures the data we save doesnt get overwritten by in place operations 
        """

        captured["starting"] = output[0].detach().clone()
        # Captures hidden state data 

        if output[0].dim() == 2:
            captured["starting"] = captured["starting"].unsqueeze(0) 
        # Fixes dimensions of the output 
        # Input of the next layer wants a 3 dimensional shape but hook returns a 2D shape
        # unsqueeze simply adds 1 to the batch dimension
        # [batch, tokens, dimensions]
        # sometimes the data will only contain [tokens, dimensions]

        #print(captured["starting"].shape)

    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_layer_stopping)
    # We call model.model.layer[x] to access a specific model layer
    # We call register_forward_hook to attach the hook we defined to a specific layer
    # Stopping_layer - 1 because of 0 indexing 

    h2 = model.model.layers[starting_layer - 1].register_forward_hook(hook_layer_starting)
    # We call model.model.layer[x] to access a specific model layer
    # we call register_forward_hook to attach the hook we defined to a specific layer
    # Starting_layer - 1 because of 0 indexing

    #print(len(model.model.layers))

    with torch.no_grad():
        model(**inputs)
    # Starting the forward pass 
    # Denoted with **to turn the input into a dictionary where every key is an 
    # argument name  

    h1.remove()
    # removing hook 1
    h2.remove()
    # removing hook 2
    
    #print("====== stopping layer ======")
    #print(captured["stopping"])

    #print("====== starting layer ======")
    #print(captured["starting"])

    return captured["stopping"], captured["starting"]


def capture_stopped_pass():
    """
    Executes the model up to the specified stopping layer and captures the 
    state required to resume execution downstream.

    This function interrupts the model's forward pass after layer 14 finishes,
    preventing the execution of all subsequent layers (15-28). It extracts
    the 'hand-off' package containing the hidden state and the necessary 
    positional context to ensure the resumption remains mathematically 
    consistent with the original pass.

    Rationale for Positional Context:
    Transformers require positional information (position_ids and 
    position_embeddings) to interpret the sequence of tokens. This metadata 
    is computed globally at Layer 0 and is static throughout the model. 
    Because Layer 15 (the resumption point) is decoupled from the initial 
    input process, it lacks access to this original context. We must 
    explicitly capture and inject these constants to ensure the attention 
    mechanisms and RoPE (Rotary Position Embeddings) calculations function 
    correctly when the partial pass resumes.

    Returns:
        tuple: A triplet containing:
            - hidden_state (Tensor): The 3D hidden state output from layer 14.
            - position_ids (Tensor): The original positional identifiers.
            - position_embeddings (tuple): The (cos, sin) cache tables.

    """
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0].detach().clone()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        captured["stopping"] = hidden
        raise StopIteration
        # Stops iteration when this hook is launched

    def hook_pos(module, args, kwargs):
        cos, sin = kwargs.get("position_embeddings")
        # Llama-3B defines these position embeddings as cos, sin and performs RoPE - Rotary Position Embeddings
        captured["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
        # Same .detach .clone logic as seen before
        captured["position_ids"] = kwargs.get("position_ids")
        

    h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)
    # position ids and embeddings are stored in the arguments for the input of each layer sp we need forward prehook to access
    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            model(**inputs)
    except StopIteration:
        pass
    # gracefully handles the stopping

    h1.remove()
    h2.remove()

    return captured["stopping"], captured["position_ids"], captured["position_embeddings"]



def capture_partial_pass(stop_layer_output, position_ids, position_embeddings):
    """
    Performs forward pass only on the starting layer
    We use to confirm if the input package works

    Args:
        stop_layer_output (Tensor): The 3D hidden state captured from the 
            stopping layer.
        position_ids (Tensor): The positional identifiers captured from 
            the global model state.
        position_embeddings (tuple): The (cos, sin) RoPE tables required 
            for rotary positional calculations.

    Returns:
        Tensor: The 3D hidden state resulting from the forward pass of the 
            resumption layer.

    """
    
    with torch.no_grad():
        for i in range(starting_layer - 1, starting_layer):
            x = model.model.layers[i](
                stop_layer_output,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
            if x.dim() == 2:
                x = x.unsqueeze(0)
    return x

if __name__ == "__main__":
    full_stopping_layer, full_starting_layer = capture_full_pass()
    stopped_stop_layer, pos_ids, pos_emb = capture_stopped_pass()
    partial_start_layer = capture_partial_pass(stopped_stop_layer, pos_ids, pos_emb)

    print("Stopping Layer match:", torch.allclose(full_stopping_layer, stopped_stop_layer))
    # torch.allclose is a boolean method which checks to see if two tensors are mathematically identical
    print("Starting Layer match:", torch.allclose(full_starting_layer, partial_start_layer))
    # torch.allclose is a boolean method which checks to see if two tensors are mathematically identical

    