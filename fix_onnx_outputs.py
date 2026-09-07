import onnx
from onnx import helper


INPUT_MODEL = "assets/models/malnutrition_model.onnx"
OUTPUT_MODEL = "assets/models/malnutrition_model_fixed.onnx"


print("========================================")
print("FIXING ONNX MODEL OUTPUTS")
print("========================================")

print()
print("Loading:")
print(INPUT_MODEL)

model = onnx.load(INPUT_MODEL)

graph = model.graph


# ============================================================
# SHOW CURRENT OUTPUTS
# ============================================================

print()
print("CURRENT GRAPH OUTPUTS")
print("----------------------------------------")

for output in graph.output:
    print(output.name)


# ============================================================
# CHECK WHETHER PROBABILITIES ALREADY EXISTS
# ============================================================

probabilities_exists = False

for output in graph.output:
    if output.name == "probabilities":
        probabilities_exists = True


if probabilities_exists:
    print()
    print("probabilities is already a graph output.")
    print("No modification required.")


else:

    print()
    print("probabilities is NOT a graph output.")
    print("Adding it now...")


    # ========================================================
    # ADD probabilities AS A FLOAT TENSOR OUTPUT
    #
    # TreeEnsembleClassifier binary classification produces:
    #
    # [probability_class_0, probability_class_1]
    #
    # for each input row.
    #
    # Therefore shape = [None, 2]
    # ========================================================

    probability_output = helper.make_tensor_value_info(
        "probabilities",
        onnx.TensorProto.FLOAT,
        [None, 2],
    )


    graph.output.append(
        probability_output
    )


    print()
    print("Added graph output:")
    print("Name: probabilities")
    print("Type: FLOAT")
    print("Shape: [None, 2]")


# ============================================================
# CHECK MODEL
# ============================================================

print()
print("========================================")
print("CHECKING MODEL")
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
print("Saved fixed model:")
print(OUTPUT_MODEL)


# ============================================================
# VERIFY
# ============================================================

print()
print("========================================")
print("FINAL GRAPH OUTPUTS")
print("========================================")

for output in graph.output:
    print(
        "Name:",
        output.name,
    )

    print(
        "Type:",
        output.type,
    )

    if output.type.HasField("tensor_type"):

        tensor_type = output.type.tensor_type

        print(
            "Element type:",
            tensor_type.elem_type,
        )

        if tensor_type.HasField("shape"):
            print(
                "Shape:",
                tensor_type.shape,
            )

    print()


print("========================================")
print("DONE")
print("========================================")