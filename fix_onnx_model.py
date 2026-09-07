import onnx
from onnx import helper


INPUT_MODEL = "assets/models/malnutrition_model.onnx"
OUTPUT_MODEL = "assets/models/malnutrition_model_fixed.onnx"


print("========================================")
print("LOADING ONNX MODEL")
print("========================================")

model = onnx.load(INPUT_MODEL)

graph = model.graph

print("Model loaded successfully.")
print()
print("NODES:")
for node in graph.node:
    print(
        f"  {node.name or '<unnamed>'} "
        f"| {node.op_type} "
        f"| inputs={list(node.input)} "
        f"| outputs={list(node.output)}"
    )

print()
print("========================================")
print("SEARCHING FOR ZIPMAP")
print("========================================")


zipmap_nodes = [
    node
    for node in graph.node
    if node.op_type == "ZipMap"
]

if not zipmap_nodes:
    print("No ZipMap node was found.")
    print(
        "The model may already be using tensor probabilities."
    )

    print()
    print("Saving a copy anyway...")

    onnx.save(model, OUTPUT_MODEL)

    print()
    print("Saved:")
    print(OUTPUT_MODEL)

    raise SystemExit


print(
    f"Found {len(zipmap_nodes)} ZipMap node(s)."
)

for node in zipmap_nodes:
    print(
        "ZipMap node:",
        node.name or "<unnamed>",
    )
    print(
        "Input:",
        list(node.input),
    )
    print(
        "Output:",
        list(node.output),
    )


# ============================================================
# WE EXPECT ONE ZIPMAP NODE FOR CLASS PROBABILITIES
# ============================================================

zipmap = zipmap_nodes[0]

zipmap_input = zipmap.input[0]
zipmap_output = zipmap.output[0]

print()
print("========================================")
print("ZIPMAP DETAILS")
print("========================================")

print("ZipMap input :", zipmap_input)
print("ZipMap output:", zipmap_output)


# ============================================================
# FIND THE GRAPH OUTPUT THAT CURRENTLY USES ZIPMAP
# ============================================================

probability_output = None

for output in graph.output:
    if output.name == zipmap_output:
        probability_output = output
        break


if probability_output is None:
    print()
    print(
        "WARNING: ZipMap output was not directly "
        "declared as a graph output."
    )


# ============================================================
# REMOVE ZIPMAP
# ============================================================

print()
print("========================================")
print("REMOVING ZIPMAP")
print("========================================")

new_nodes = []

for node in graph.node:
    if node is not zipmap:
        new_nodes.append(node)

graph.ClearField("node")

graph.node.extend(new_nodes)


# ============================================================
# REPLACE THE ZIPMAP GRAPH OUTPUT
#
# The tensor going INTO ZipMap contains:
#
# [probability_class_0, probability_class_1]
#
# We expose that tensor directly.
# ============================================================

print()
print("========================================")
print("CREATING TENSOR PROBABILITY OUTPUT")
print("========================================")


# Remove old ZipMap output from graph outputs.

remaining_outputs = []

for output in graph.output:
    if output.name != zipmap_output:
        remaining_outputs.append(output)


graph.ClearField("output")

graph.output.extend(remaining_outputs)


# ============================================================
# CREATE NEW TENSOR OUTPUT
# ============================================================

probability_tensor_output = helper.make_tensor_value_info(
    "probabilities",
    onnx.TensorProto.FLOAT,
    [None, 2],
)

graph.output.append(probability_tensor_output)


# ============================================================
# RENAME THE INPUT OF THE ZIPMAP AS "probabilities"
#
# Instead of:
#
# classifier -> ZipMap -> probability map
#
# we want:
#
# classifier -> probabilities tensor
# ============================================================

for node in graph.node:
    for i in range(len(node.output)):
        if node.output[i] == zipmap_input:
            node.output[i] = "probabilities"

    for i in range(len(node.input)):
        if node.input[i] == zipmap_input:
            node.input[i] = "probabilities"


# ============================================================
# CHECK MODEL
# ============================================================

print()
print("========================================")
print("CHECKING FIXED MODEL")
print("========================================")

onnx.checker.check_model(model)

print("ONNX checker: PASSED")


# ============================================================
# SAVE
# ============================================================

print()
print("========================================")
print("SAVING FIXED MODEL")
print("========================================")

onnx.save(
    model,
    OUTPUT_MODEL,
)

print()
print("SUCCESS!")
print()
print(
    "Fixed model saved to:"
)
print(
    OUTPUT_MODEL
)

print()
print("========================================")
print("FIXED MODEL OUTPUTS")
print("========================================")

for output in graph.output:
    print(
        output.name,
        "|",
        output.type,
    )

print()
print("========================================")
print("DONE")
print("========================================")